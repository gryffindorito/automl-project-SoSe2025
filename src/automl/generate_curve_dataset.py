import os
import torch

from automl.curve_predictor import train_and_record_curve  # ✅ Absolute import for Colab
from automl.utils import set_seed

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

    train_and_record_curve(
        model_name=args.model,
        dataset_name=args.dataset,
        lr=config["lr"],
        weight_decay=config["weight_decay"],
        optimizer_type=config["optimizer_type"],
        scheduler_type=config["scheduler_type"],
        epochs=50,
        save_path=curve_path
    )
