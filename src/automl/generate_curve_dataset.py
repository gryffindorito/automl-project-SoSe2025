import os
import torch
import copy
from tqdm import tqdm
from src.automl.models import get_model
from src.automl.synflow import compute_synflow_score
from src.automl.curve_predictor import train_and_record_curve
from src.automl.dataloader_utils import get_dataloaders

# Per-model HPO config
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
    curve_data = []

    # 📂 Load existing data if resuming
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

            # Detect input channels
            first_batch = next(iter(train_loader))[0]
            in_channels = first_batch.shape[1]

            # Detect number of classes manually
            dataset_obj = train_loader.dataset
            base = dataset_obj.dataset if hasattr(dataset_obj, 'dataset') else dataset_obj

            try:
                labels = [int(base[i][1]) for i in range(len(base))]
                num_classes = len(set(labels))
            except Exception as e:
                raise ValueError(f"❌ Could not determine number of classes for dataset '{dataset_name}': {e}")

            # Get model
            model = get_model(model_name, num_classes, in_channels).to(device)

            # Compute SynFlow
            input_shape = tuple(first_batch.shape)
            synflow = compute_synflow_score(model, input_shape, device=device)

            # Get model-specific HPO config
            config = model_hpo.get(model_name, {})

            # Epoch-by-epoch training + saving
            metrics = {}
            for epoch in range(max_epoch):
                _, epoch_metrics = train_and_record_curve(
                    model, train_loader, val_loader,
                    num_epochs=1,  # just one epoch per call
                    device=device,
                    lr=config.get("lr", 1e-3),
                    wd=config.get("wd", 1e-4),
                    optimizer_type=config.get("opt", "adamw"),
                    scheduler_type=config.get("sched", None)
                )
                metrics.update(epoch_metrics)

                # Create updated entry
                entry = {
                    "model": model_name,
                    "dataset": dataset_name,
                    "curve": [metrics[e] for e in sorted(metrics)],
                    "synflow": synflow,
                }

                # Replace previous partial or existing entry
                curve_data = [e for e in curve_data if not (e['model'] == model_name and e['dataset'] == dataset_name)]
                curve_data.append(copy.deepcopy(entry))
                torch.save(curve_data, save_path)
                print(f"💾 Saved progress after epoch {epoch+1}/{max_epoch}")

    print("\n🎉 All curve data generated.")
