import os
import torch
from tqdm import tqdm
from src.automl.models import get_model
from src.automl.synflow import compute_synflow_score
from src.automl.curve_predictor import train_and_record_curve
from src.automl.dataloader_utils import get_dataloaders

# ✅ Per-model HPO config
model_hpo = {
    "resnet18": {
        "lr": 1e-3,
        "wd": 1e-4,
        "opt": "adamw",
        "sched": "cosine",
    },
    "mobilenet_v2": {
        "lr": 5e-4,
        "wd": 5e-4,
        "opt": "adamw",
        "sched": "step",
    },
    "efficientnet_b0": {
        "lr": 1e-4,
        "wd": 1e-3,
        "opt": "adamw",
        "sched": None,
    }
}


def generate_curve_dataset(
    model_names, dataset_names, save_path="curve_dataset.pt",
    dataset_dir="automl_data", device="cuda", max_epoch=20
):
    # Load existing progress
    if os.path.exists(save_path):
        print(f"📂 Found existing dataset at {save_path}, loading...")
        curve_data = torch.load(save_path)
    else:
        curve_data = []

    # Make (model, dataset) → entry mapping
    entry_lookup = {
        (entry['model'], entry['dataset']): entry for entry in curve_data
    }

    for dataset_name in dataset_names:
        for model_name in model_names:
            print(f"\n🚀 Processing {model_name} on {dataset_name}")
            key = (model_name, dataset_name)

            # Remove previous entry if exists — no resume
            if key in entry_lookup:
                print(f"🗑 Removing previous entry for {model_name} on {dataset_name}")
                del entry_lookup[key]

            # Get dataloaders
            train_loader, val_loader, _ = get_dataloaders(
                dataset_name, root=dataset_dir, batch_size=64
            )

            print("✅ Dataset split complete.")
            print("Train:", len(train_loader.dataset), "| Val:", len(val_loader.dataset))

            # Get input/output info
            first_batch = next(iter(train_loader))[0]
            in_channels = first_batch.shape[1]

            base = train_loader.dataset
            base = base.dataset if hasattr(base, "dataset") else base
            labels = [int(base[i][1]) for i in range(len(base))]
            num_classes = len(set(labels))

            # Fresh model
            model = get_model(model_name, num_classes, in_channels).to(device)

            # SynFlow
            input_shape = tuple(first_batch.shape)
            synflow = compute_synflow_score(model, input_shape, device=device)

            # HPO config
            config = model_hpo.get(model_name, {})

            # Train from scratch
            model, curve, _ = train_and_record_curve(
                model=model,
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=max_epoch,
                device=device,
                lr=config.get("lr", 1e-3),
                wd=config.get("wd", 1e-4),
                optimizer_type=config.get("opt", "adamw"),
                scheduler_type=config.get("sched", None),
                curve_path=save_path,
                model_name=model_name,
                dataset_name=dataset_name
            )

            # Save entry
            entry_lookup[key] = {
                "model": model_name,
                "dataset": dataset_name,
                "curve": curve,
                "synflow": synflow
            }
            torch.save(list(entry_lookup.values()), save_path)
            print(f"✅ Completed {model_name} on {dataset_name} | Final Acc: {curve[-1]:.4f}")

    print("\n🎉 All curve data generated.")
