import optuna
import torch
from src.automl.dataloader_utils import get_dataloaders
from src.automl.models import get_model
from src.automl.curve_predictor import train_and_record_curve

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def run_optuna_study(model_name, dataset_name, base_lr, base_wd,
                     n_trials=5, max_epoch=10, data_dir="automl_data"):
    print(f"🎯 Starting focused Optuna HPO for {model_name} on {dataset_name}")

    results = []

    def objective(trial):
        lr = trial.suggest_float("lr", 0.5 * base_lr, 1.5 * base_lr, log=True)
        wd = trial.suggest_float("weight_decay", 0.3 * base_wd, 2.0 * base_wd, log=True)

        print(f"🧪 Trial Params → lr: {lr:.6f}, weight_decay: {wd:.6f}")

        train_loader, val_loader, _ = get_dataloaders(
            dataset_name, root=data_dir, batch_size=64
        )

        in_channels = next(iter(train_loader))[0].shape[1]
        base = train_loader.dataset
        base = base.dataset if hasattr(base, "dataset") else base
        num_classes = len(set(int(base[i][1]) for i in range(len(base))))

        model = get_model(model_name, num_classes, in_channels).to(DEVICE)

        _, acc_curve, loss_curve = train_and_record_curve(
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

        results.append({
            "model": model_name,
            "dataset": dataset_name,
            "acc_curve": acc_curve,
            "loss_curve": loss_curve,
            "lr": lr,
            "weight_decay": wd,
            "in_channels": in_channels,
            "num_classes": num_classes
        })

        return acc_curve[-1]  # Use final val acc as score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)

    print(f"✅ Best Params: {study.best_params}")
    print(f"🏁 Best Final Val Acc: {study.best_value:.4f}")

    return results  # list of 5 trial dicts
