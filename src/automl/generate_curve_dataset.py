import os
import torch
import time
from .models import build_model
from .trainer import train
from .dataloader_utils import get_dataloaders
from torch.utils.data import random_split
from tqdm import tqdm  # ✅ Added for progress indication

def generate_curve_dataset(
    dataset_name: str,
    model_names: list,
    dataset_dir: str = "data",
    output_path: str = "curve_dataset.pt",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    n_epochs: int = 20
):
    """
    Trains a list of models on a dataset and saves their learning curves.

    Args:
        dataset_name: One of ["fashion", "flowers", "emotions"]
        model_names: List of model names (e.g. ["resnet18", "mobilenet_v2"])
        output_path: Where to save the curves
        device: "cuda" or "cpu"
        n_epochs: Number of epochs to train each model
    """
    dataset_info = {
        "fashion": {"classes": 10},
        "emotions": {"classes": 7},
        "flowers": {"classes": 102},
    }

    assert dataset_name in dataset_info, f"Unknown dataset: {dataset_name}"
    num_classes = dataset_info[dataset_name]["classes"]

    # Get dataloaders (not datasets!)
    train_loader, val_loader, _ = get_dataloaders(dataset_name, root=dataset_dir)

    results = []

    for model_name in tqdm(model_names, desc="Training Models", unit="model"):  # ✅ Progress bar for models
        print(f"\n Training {model_name} on {dataset_name} for {n_epochs} epochs...")

        model = build_model(model_name, num_classes=num_classes)

        start_time = time.time()  # ✅ Start timing

        curve = train(model, train_loader, val_loader, device=device, epochs=n_epochs)

        duration = time.time() - start_time  # ✅ End timing

        print(f"✅ Finished training {model_name} in {duration:.2f} seconds.\n")

        results.append((model_name, curve))

    torch.save(results, output_path)
    print(f" Saved learning curves to {output_path}")
