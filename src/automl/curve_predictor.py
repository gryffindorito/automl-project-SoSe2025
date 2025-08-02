import torch
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch.nn.functional as F
import os

def get_curve(metrics_dict):
    return [metrics_dict[epoch] for epoch in sorted(metrics_dict)]

def train_and_record_curve(
    model, train_loader, val_loader,
    num_epochs=20, device="cuda",
    lr=1e-3, wd=1e-4, optimizer_type="adamw", scheduler_type=None,
    curve_path=None, model_name=None, dataset_name=None
):
    # Optimizer
    if optimizer_type == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif optimizer_type == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

    # Scheduler
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    # Init
    model.to(device)
    model.train()
    val_accuracies = []
    val_losses = []

    for epoch in range(num_epochs):
        # Train loop
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # Eval loop
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

    # Save curve record
    if curve_path:
        record = {
            "model": model_name,
            "dataset": dataset_name,
            "acc_curve": val_accuracies,
            "loss_curve": val_losses
        }

        if os.path.exists(curve_path):
            data = torch.load(curve_path)
            data.append(record)
        else:
            data = [record]

        torch.save(data, curve_path)
        print(f"💾 Saved curve data to {curve_path}")

    return model, val_accuracies, val_losses

def build_feature_vector(synflow_score, acc_prefix, loss_prefix):
    return np.array([synflow_score] + acc_prefix + loss_prefix)

def train_regressor(data, save_path="regressor.pkl"):
    X, y = [], []
    for item in data:
        feature = build_feature_vector(
            item["synflow"],
            item["acc_curve"][:10],
            item["loss_curve"][:10]
        )
        X.append(feature)
        y.append(item["acc_curve"][-1])
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        verbosity=0
    )
    model.fit(X, y)
    joblib.dump(model, save_path)
    print(f"✅ XGBoost regressor saved to {save_path}")
    return model

def predict_final_accuracy(model_path, synflow_score, acc_prefix, loss_prefix):
    model = joblib.load(model_path)
    X = build_feature_vector(synflow_score, acc_prefix, loss_prefix).reshape(1, -1)
    return model.predict(X)[0]

def evaluate_regressor(model_path, data):
    model = joblib.load(model_path)
    X, y = [], []
    for item in data:
        feature = build_feature_vector(
            item["synflow"],
            item["acc_curve"][:10],
            item["loss_curve"][:10]
        )
        X.append(feature)
        y.append(item["acc_curve"][-1])
    preds = model.predict(X)
    if len(y) < 2:
        print("⚠️ Only one sample — R² score is not well-defined.")
        return None
    return r2_score(y, preds)

__all__ = [
    "get_curve",
    "train_and_record_curve",
    "build_feature_vector",
    "train_regressor",
    "predict_final_accuracy",
    "evaluate_regressor"
]
