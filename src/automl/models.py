import torch.nn as nn
import torchvision.models as models

def build_model(name: str, num_classes: int, in_channels: int = 3):
    if name == "resnet18":
        model = models.resnet18(pretrained=False)
        if in_channels == 1:
            model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif name == "mobilenet_v2":
        model = models.mobilenet_v2(pretrained=False)
        if in_channels == 1:
            model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=False)
        if in_channels == 1:
            model.features[0][0] = nn.Conv2d(1, model.features[0][0].out_channels,
                                             kernel_size=model.features[0][0].kernel_size,
                                             stride=model.features[0][0].stride,
                                             padding=model.features[0][0].padding,
                                             bias=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    else:
        raise ValueError(f"Unknown model: {name}")

    return model
