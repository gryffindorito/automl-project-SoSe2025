import torch
import numpy as np
import joblib
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import torch.nn.functional as F
import os

MODEL_MAP = {"resnet18": 0, "mobilenet_v2": 1, "efficientnet_b0": 2}

# 🧠 Best hyperparameters per model from HPO
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


def safe_polyfit(y_values):
    if len(y_values) < 2:
        return 0.0
    if np.all(np.isclose(y_values, y_values[0])):
        return 0.0  # flat curve
    try:
        return np.polyfit(np.arange(1, len(y_values)+1), y_values, 1)[0]
    except:
        return 0.0


def estimate_num_classes(item):
    return 10  # fallback if not present in item


def train_per_model_regressors(data, save_dir="model_regressors"):
    os.makedirs(save_dir, exist_ok=True)
    by_model = {"resnet18": [], "mobilenet_v2": [], "efficientnet_b0": []}
    for item in data:
        if item["model"] in by_model:
            by_model[item["model"]].append(item)

    for model_name, model_data in by_model.items():
        if not model_data:
            print(f"⚠️ No data for {model_name}, skipping.")
            continue
        print(f"🎓 Training regressor for {model_name} on {len(model_data)} samples")
        model = Ridge(alpha=1.0)
        X = [extract_features(item) for item in model_data]
        y = [item["acc_curve"][49] for item in model_data]
        model.fit(X, y)
        path = os.path.join(save_dir, f"{model_name}_regressor.pkl")
        joblib.dump(model, path)
        print(f"✅ Saved {model_name} regressor to {path}")


def train_and_record_curve(
    model, train_loader, val_loader,
    num_epochs=20, device="cuda",
    lr=1e-3, wd=1e-4, optimizer_type="adamw", scheduler_type=None,
    curve_path=None, model_name=None, dataset_name=None
):
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    model.to(device)
    model.train()
    val_accuracies = []
    val_losses = []

    if curve_path and not os.path.exists(curve_path):
        torch.save([], curve_path)

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        model.eval()
        correct, total, total_loss = 0, 0, 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
                total_loss += F.cross_entropy(logits, labels, reduction="sum").item()

        acc = correct / total
        avg_loss = total_loss / total
        val_accuracies.append(acc)
        val_losses.append(avg_loss)

        print(f"📊 Epoch {epoch+1}/{num_epochs} | Val Acc: {acc:.4f} | Val Loss: {avg_loss:.4f}")
        model.train()

        if curve_path:
            record = {
                "model": model_name,
                "dataset": dataset_name,
                "acc_curve": val_accuracies.copy(),
                "loss_curve": val_losses.copy()
            }
            try:
                data = torch.load(curve_path)
            except:
                data = []

            data = [d for d in data if not (d["model"] == model_name and d["dataset"] == dataset_name)]
            data.append(record)
            torch.save(data, curve_path)
            print(f"💾 Saved progress to {curve_path} after epoch {epoch+1}/{num_epochs}")

    if curve_path:
        weights_path = curve_path.replace(".pt", "_weights.pt")
        torch.save(model.state_dict(), weights_path)
        print(f"💾 Saved model weights to {weights_path}")

    return model, val_accuracies, val_losses


def extract_features(item):
    acc = item["acc_curve"][:10]
    loss = item["loss_curve"][:10]

    acc_deltas = [acc[i+1] - acc[i] for i in range(len(acc)-1)]
    loss_deltas = [loss[i+1] - loss[i] for i in range(len(loss)-1)]

    acc_slope = safe_polyfit(acc)
    loss_slope = safe_polyfit(loss)
    plateau = 1.0 if len(acc_deltas) > 0 and acc_deltas[-1] < 0.001 else 0.0
    loss_drop = loss[-1] - loss[0] if len(loss) >= 2 else 0.0

    acc_last5_mean = (
        np.mean(item["acc_curve"][45:50])
        if len(item["acc_curve"]) >= 50
        else 0.0
    )
    acc_max_first10 = max(acc) if acc else 0.0
    acc_std_first10 = np.std(acc) if acc else 0.0

    model_one_hot = [0, 0, 0]
    if item["model"] in MODEL_MAP:
        model_one_hot[MODEL_MAP[item["model"]]] = 1

    in_channels = item.get("in_channels", 0)
    num_classes = item.get("num_classes", 0)

    return np.array(
        acc +
        loss +
        acc_deltas +
        loss_deltas +
        [acc_slope, loss_slope, loss_drop, plateau] +
        [acc_last5_mean, acc_max_first10, acc_std_first10] +
        model_one_hot +
        [in_channels, num_classes]
    )


def train_regressor(data, save_path="regressor.pkl"):
    X, y = [], []
    for item in data:
        if "in_channels" not in item:
            item["in_channels"] = 3  # default
        if "num_classes" not in item:
            item["num_classes"] = estimate_num_classes(item)

        features = extract_features(item)
        X.append(features)
        y.append(item["acc_curve"][49])
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    joblib.dump(model, save_path)
    print(f"✅ Ridge regressor saved to {save_path}")
    return model


def predict_final_accuracy(model_path, item):
    model = joblib.load(model_path)
    X = extract_features(item).reshape(1, -1)
    return model.predict(X)[0]


def evaluate_regressor(model_path, data):
    model = joblib.load(model_path)
    X, y = [], []
    for item in data:
        if "in_channels" not in item:
            item["in_channels"] = 3
        if "num_classes" not in item:
            item["num_classes"] = estimate_num_classes(item)
        X.append(extract_features(item))
        y.append(item["acc_curve"][49])
    preds = model.predict(X)
    if len(y) < 2:
        print("⚠️ Only one sample — R² score is not well-defined.")
        return None
    return r2_score(y, preds)


def run_full_automl(dataset_name, regressor_path, device='cuda', data_dir='/content/automl_data'):
    import os
    from .models import get_model
    from .dataloader_utils import get_dataloaders
    from .curve_predictor import train_and_record_curve, predict_final_accuracy

    print(f"\n🤖 AutoML selecting best model for {dataset_name}...")

    model_names = ["resnet18", "mobilenet_v2", "efficientnet_b0"]
    best_model, best_score = None, -1
    all_records = []

    dataset_path = os.path.join(data_dir, dataset_name)
    print(f"📂 Loading dataset from: {dataset_path}")

    for model_name in model_names:
        print(f"\n🚀 Training {model_name} for 10 epochs...")

        try:
            train_loader, val_loader, _ = get_dataloaders(
                dataset_name=dataset_name,
                root=data_dir,
                batch_size=64
            )
        except FileNotFoundError as e:
            print(f"❌ Dataset loading error for {model_name}: {e}")
            continue

        in_channels = next(iter(train_loader))[0].shape[1]
        base = train_loader.dataset
        base = base.dataset if hasattr(base, 'dataset') else base
        num_classes = len(set(int(base[i][1]) for i in range(len(base))))

        model = get_model(model_name, num_classes=num_classes, in_channels=in_channels).to(device)
        config = model_hpo[model_name]

        _, acc_curve, loss_curve = train_and_record_curve(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=10,
            device=device,
            lr=config["lr"],
            wd=config["weight_decay"],
            optimizer_type=config["optimizer_type"],
            scheduler_type=config["scheduler_type"],
            curve_path=None,
            model_name=model_name,
            dataset_name=dataset_name
        )

        item = {
            "model": model_name,
            "dataset": dataset_name,
            "acc_curve": acc_curve,
            "loss_curve": loss_curve,
            "in_channels": in_channels,
            "num_classes": num_classes
        }
        all_records.append(item)

        try:
            pred_acc = predict_final_accuracy(regressor_path, item)
            print(f"📈 {model_name} → Predicted acc50: {pred_acc:.4f}")
            if pred_acc > best_score:
                best_score = pred_acc
                best_model = model_name
        except Exception as e:
            print(f"❌ Error in prediction for {model_name}: {e}")

    out_path = f"curve_dataset_{dataset_name}.pt"
    torch.save(all_records, out_path)
    print(f"\n💾 Saved evaluation curves to {out_path}")
    print(f"\n🏆 Best model: {best_model} with predicted acc50 = {best_score:.4f}")
    return best_model, best_score


__all__ = [
    "train_and_record_curve",
    "train_regressor",
    "predict_final_accuracy",
    "evaluate_regressor",
    "run_full_automl"
]
