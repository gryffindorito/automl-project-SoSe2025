import os
import torch

from automl.curve_predictor import train_and_record_curve  # ✅ Absolute import for Colab
from automl.utils import set_seed
from automl.dataloader_utils import get_dataloaders
from automl.models import get_model

# 🧠 Best hyperparameters per model from Optuna (regardless of dataset)
model_hpo = {
    "resnet18": {
        "lr": 0.0003498223657876835,
        "weight_decay": 0.0007347274364999834,
        "optimizer_type": "adamw",
        "scheduler_type": "none"
    },
    "mobilenet_v2": {
        "lr": 0.0028309852006291976,
        "weight_decay": 0.0001829693458682074,
        "optimizer_type": "adamw",
        "scheduler_type": "none"
    },
    "efficientnet_b0": {
        "lr": 0.0007415721317706288,
        "weight_decay": 8.387790933340183e-05,
        "optimizer_type": "adamw",
        "scheduler_type": "none"
    }
}

def run_curve_mode(args):
    # 🎯 Always use fixed seed for reproducibility
    set_seed(42)

    config = model_hpo[args.model]
    os.makedirs(args.curve_dir, exist_ok=True)
    curve_path = os.path.join(args.curve_dir, f"curve_dataset_{args.dataset}_{args.model}.pt")

    # 📦 Load data and model
    train_loader, val_loader, _ = get_dataloaders(args.dataset, root=args.data_dir, batch_size=64)
    first_batch = next(iter(train_loader))[0]
    in_channels = first_batch.shape[1]

    base = train_loader.dataset
    base = base.dataset if hasattr(base, "dataset") else base
    labels = [int(base[i][1]) for i in range(len(base))]
    num_classes = len(set(labels))

    model = get_model(args.model, num_classes=num_classes, in_channels=in_channels).to(args.device)

    # 🧪 Train and record learning curve (both accuracy and loss)
    results = train_and_record_curve(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.curve_epochs,
        device=args.device,
        lr=config["lr"],
        wd=config["weight_decay"],
        optimizer_type=config["optimizer_type"],
        scheduler_type=config["scheduler_type"],
    )

    # 📊 Save results
    torch.save({
        "model": args.model,
        "dataset": args.dataset,
        "val_accuracies": results["val_accuracies"],
        "val_losses": results["val_losses"]
    }, curve_path)
    print(f"✅ Curve saved to {curve_path}")
