import torch
import torch.nn as nn

@torch.no_grad()
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

    # Linearize model (replace ReLU with identity)
    def linearize(model):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False
                module.forward = lambda x: x

    def nonlinearize(model):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.forward = nn.ReLU(inplace=True).forward

    linearize(model)

    # Save original weights and mask gradients
    for param in model.parameters():
        param.requires_grad_(True)
        param.data = param.data.abs()

    input_tensor = torch.ones(input_size).to(device)
    output = model(input_tensor).sum()
    output.backward()

    score = 0.0
    for param in model.parameters():
        if param.grad is not None:
            score += (param.data * param.grad).abs().sum().item()

    # Undo changes
    nonlinearize(model)
    model.zero_grad()

    return score
