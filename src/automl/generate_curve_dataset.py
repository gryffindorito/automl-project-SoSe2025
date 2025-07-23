import os
import torch
import time
from .models import build_model
from .trainer import train
from .dataloader_utils import get_dataloaders
from torch.utils.data import random_split
from tqdm import tqdm  # ✅ Progress bar for models

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
    Now saves learning curves incrementally after every epoch to avoid loss.
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

    for model_name in tqdm(model_names, desc="Training Models", unit="model"):
        print(f"\n Training {model_name} on {dataset_name} for {n_epochs} epochs...")

        model = build_model(model_name, num_classes=num_classes)

        curve_so_far = []

        def on_epoch_end(epoch, val_accs):
            # Update local curve and write full file
            curve_so_far.clear()
            curve_so_far.extend(val_accs)
            temp_results = results + [(model_name, curve_so_far)]
            torch.save(temp_results, output_path)
            print(f" Saved checkpoint with {len(temp_results)} models to {output_path}")

        start_time = time.time()

        train(
            model,
            train_loader,
            val_loader,
            device=device,
            epochs=n_epochs,
            on_epoch_end=on_epoch_end
        )

        duration = time.time() - start_time
        print(f" Finished training {model_name} in {duration:.2f} seconds.\n")

        results.append((model_name, curve_so_far))

    # Final save (redundant but safe)
    torch.save(results, output_path)
    print(f" Final saved learning curves to {output_path}")
