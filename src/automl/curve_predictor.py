import torch
import numpy as np
import joblib
from sklearn.metrics import r2_score
from sklearn.linear_model import Ridge
import torch.nn.functional as F
import os

MODEL_MAP = {"resnet18": 0, "mobilenet_v2": 1, "efficientnet_b0": 2}

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

    return model, val_accuracies, val_losses

def extract_features(item):
    acc = item["acc_curve"][:10]
    loss = item["loss_curve"][:10]
    acc_deltas = [acc[i+1] - acc[i] for i in range(9)]
    loss_deltas = [loss[i+1] - loss[i] for i in range(9)]

    # Slopes via linear fit
    epochs = np.arange(1, 11)
    acc_slope = np.polyfit(epochs, acc, 1)[0]
    loss_slope = np.polyfit(epochs, loss, 1)[0]

    # Plateau detection: final acc delta very low
    plateau = 1.0 if acc_deltas[-1] < 0.001 else 0.0

    # Loss drop (loss[10] - loss[1])
    loss_drop = loss[-1] - loss[0]

    # Model one-hot
    model_one_hot = [0, 0, 0]
    if item["model"] in MODEL_MAP:
        model_one_hot[MODEL_MAP[item["model"]]] = 1

    return np.array(acc + loss + acc_deltas + loss_deltas + [acc_slope, loss_slope, loss_drop, plateau] + model_one_hot)

def train_regressor(data, save_path="regressor.pkl"):
    X, y = [], []
    for item in data:
        features = extract_features(item)
        X.append(features)
        y.append(item["acc_curve"][49])  # Epoch 50
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
        X.append(extract_features(item))
        y.append(item["acc_curve"][49])
    preds = model.predict(X)
    if len(y) < 2:
        print("⚠️ Only one sample — R² score is not well-defined.")
        return None
    return r2_score(y, preds)

__all__ = [
    "train_and_record_curve",
    "train_regressor",
    "predict_final_accuracy",
    "evaluate_regressor"
]
