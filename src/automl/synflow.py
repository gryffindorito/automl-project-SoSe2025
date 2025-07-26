import torch
import torch.nn as nn

def compute_synflow_score(model, input_size=(1, 1, 28, 28), device='cuda'):
    """
    Compute the SynFlow score for the model.

    Args:
        model: The PyTorch model.
        input_size: Size of dummy input.
        device: 'cuda' or 'cpu'.

    Returns:
        SynFlow score (float).
    """
    model = model.to(device)
    model.eval()

    # Step 1: Replace ReLU with Identity (linearize)
    def linearize(model):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
                module.forward = lambda x: x

    # Step 2: Restore ReLU after computation
    def nonlinearize(model):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.forward = nn.ReLU(inplace=True).forward

    linearize(model)

    # Step 3: Ensure all weights are positive and require gradients
    for param in model.parameters():
        param.requires_grad_(True)
        param.data = param.data.abs()

    # Step 4: Create input that also requires gradient
    input_tensor = torch.ones(input_size, device=device, requires_grad=True)
    output = model(input_tensor).sum()
    output.backward()

    # Step 5: SynFlow = sum over abs(w * grad)
    score = 0.0
    for param in model.parameters():
        if param.grad is not None:
            score += (param.data * param.grad).abs().sum().item()

    # Step 6: Cleanup
    nonlinearize(model)
    model.zero_grad()

    return score
