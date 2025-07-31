import optuna
import torch
from src.automl.dataloader_utils import get_dataloaders
from src.automl.models import get_model
from src.automl.curve_predictor import train_and_record_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_optuna_study(model_name, dataset_name, n_trials=15, max_epoch=20, data_dir="automl_data"):
    print(f"🎯 Starting Optuna HPO for {model_name} on {dataset_name}")

    def objective(trial):
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        wd = trial.suggest_float("weight_decay", 1e-5, 5e-3, log=True)

        train_loader, val_loader, _ = get_dataloaders(
            dataset_name, root=data_dir, batch_size=64
        )

        # Extract info
        in_channels = next(iter(train_loader))[0].shape[1]
        base = train_loader.dataset
        base = base.dataset if hasattr(base, "dataset") else base
        labels = [int(base[i][1]) for i in range(len(base))]
        num_classes = len(set(labels))

        model = get_model(model_name, num_classes, in_channels).to(DEVICE)

        _, curve, _ = train_and_record_curve(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=max_epoch,
            device=DEVICE,
            lr=lr,
            wd=wd,
            optimizer_type="adamw",
            scheduler_type=None,
            curve_path=None
        )

        return curve[-1]  # Return final val acc

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"✅ Best Params for {model_name} on {dataset_name}: {study.best_params}")
    print(f"🏁 Best Val Accuracy: {study.best_value:.4f}")

    return study.best_params, study.best_value
