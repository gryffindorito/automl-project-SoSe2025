import os
import torch
from tqdm import tqdm
from src.automl.models import get_model
from src.automl.synflow import compute_synflow_score
from src.automl.curve_predictor import get_curve, train_and_record_curve
from src.automl.dataloader_utils import get_dataloaders


def generate_curve_dataset(
    model_names, dataset_names, save_path="curve_dataset.pt",
    dataset_dir="automl_data", device="cuda", max_epoch=20
):
    curve_data = []

    # Resume if existing
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

            print(f"\n🚀 Processing {model_name} on {dataset_name}")

            # Load data
            train_loader, val_loader, _ = get_dataloaders(
                dataset_name, root=dataset_dir, batch_size=64
            )

            print("✅ Dataset split complete.")
            print("Train:", len(train_loader.dataset), "| Val:", len(val_loader.dataset))

            # Get input shape
            first_batch = next(iter(train_loader))[0]
            in_channels = first_batch.shape[1]

            # Infer number of classes
            dataset_obj = train_loader.dataset
            base = dataset_obj.dataset if hasattr(dataset_obj, 'dataset') else dataset_obj
            try:
                labels = [int(base[i][1]) for i in range(len(base))]
                num_classes = len(set(labels))
            except Exception as e:
                raise ValueError(f"❌ Could not determine number of classes: {e}")

            # Get model
            model = get_model(model_name, num_classes, in_channels).to(device)

            # Compute SynFlow
            input_shape = tuple(first_batch.shape)
            synflow = compute_synflow_score(model, input_shape, device=device)

            # Train and get curve
            _, metrics = train_and_record_curve(model, train_loader, val_loader, num_epochs=max_epoch, device=device)
            curve = get_curve(metrics)


            # Save entry
            curve_data.append({
                "model": model_name,
                "dataset": dataset_name,
                "curve": curve,
                "synflow": synflow,
            })

            torch.save(curve_data, save_path)
            print(f"✅ Saved curve entry for {model_name} on {dataset_name}")

    print("\n🎉 All curve data generated.")
