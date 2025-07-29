import os
import torch
import numpy as np
from tqdm import tqdm
from src.automl.models import get_model
from src.automl.synflow import compute_synflow_score
from src.automl.curve_predictor import get_curve
from src.automl.dataloader_utils import get_dataloaders  # Updated import

def generate_curve_dataset(model_names, dataset_names, save_path="curve_dataset.pt",
                           dataset_dir="automl_data", device="cuda", max_epoch=20):
    curve_data = []

    # 🔄 Resume if previous results exist
    if os.path.exists(save_path):
        print(f"📂 Found existing dataset at {save_path}, loading...")
        curve_data = torch.load(save_path)
        done_pairs = {(entry['model'], entry['dataset']) for entry in curve_data}
    else:
        done_pairs = set()

    for dataset_name in dataset_names:
        for model_name in model_names:
            if (model_name, dataset_name) in done_pairs:
                print(f"⏩ Skipping {model_name} on {dataset_name}, already done.")
                continue

            print(f"🚀 Processing {model_name} on {dataset_name}")

            # 🔌 Load data
            train_loader, val_loader, _ = get_dataloaders(dataset_name, dataset_dir)

            first_batch = next(iter(train_loader))[0]
            in_channels = first_batch.shape[1]
            num_classes = getattr(train_loader.dataset, 'num_classes', None)
            if num_classes is None:
                try:
                    num_classes = len(train_loader.dataset.classes)
                except:
                    raise ValueError("Could not determine number of classes for dataset.")

            # 🏗️ Build model
            model = get_model(model_name, num_classes, in_channels).to(device)

            # 🔢 SynFlow
            synflow = compute_synflow_score(model, dummy_input=first_batch.to(device))

            # 📈 Learning curve
            _, curve = get_curve(model, train_loader, val_loader, epochs=max_epoch, device=device)

            curve_data.append({
                "model": model_name,
                "dataset": dataset_name,
                "curve": curve,
                "synflow": synflow,
            })

            # 💾 Save progress
            torch.save(curve_data, save_path)
            print(f"✅ Saved curve entry for {model_name} on {dataset_name} to {save_path}")

    print("🎉 All curve data generated.")
