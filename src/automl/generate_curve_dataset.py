import os
import torch
from torchvision import transforms
from torchvision.datasets import ImageFolder
from .models import build_model
from .trainer import train

def get_dataloaders_from_folder(dataset_path, image_size, grayscale, split_ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.Grayscale(num_output_channels=1) if grayscale else transforms.Lambda(lambda x: x),
        transforms.ToTensor()
    ])

    full_dataset = ImageFolder(dataset_path, transform=transform)
    total_size = len(full_dataset)
    train_size = int(total_size * split_ratio)
    val_size = total_size - train_size
    return torch.utils.data.random_split(full_dataset, [train_size, val_size])

def generate_curve_dataset(
    dataset_name: str,
    model_names: list,
    dataset_dir: str = "data",
    output_path: str = "curve_dataset.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
):
    dataset_info = {
        "fashion": {"img_size": 28, "channels": 1, "classes": 10},
        "emotions": {"img_size": 48, "channels": 1, "classes": 7},
        "flowers": {"img_size": 512, "channels": 3, "classes": 102},
    }

    assert dataset_name in dataset_info, f"Unknown dataset: {dataset_name}"

    dataset_path = os.path.join(dataset_dir, dataset_name, "images_train")
    image_size = dataset_info[dataset_name]["img_size"]
    grayscale = dataset_info[dataset_name]["channels"] == 1
    num_classes = dataset_info[dataset_name]["classes"]

    results = []

    for model_name in model_names:
        print(f"Training {model_name} on {dataset_name}")
        model = build_model(model_name, num_classes)
        train_data, val_data = get_dataloaders_from_folder(dataset_path, image_size, grayscale)
        curve = train(model, train_data, val_data, device=device)
        results.append((model_name, curve))

    torch.save(results, output_path)
    print(f"Saved learning curves to {output_path}")
