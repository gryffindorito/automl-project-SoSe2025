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

    # Load previous curve file if exists
    existing_data = []
    if curve_path and os.path.exists(curve_path):
        existing_data = torch.load(curve_path)
        for item in existing_data:
            if item["model"] == model_name and item["dataset"] == dataset_name:
                curve = item["curve"]
                break
        else:
            curve = []
            existing_data.append({"model": model_name, "dataset": dataset_name, "curve": curve})
    else:
        curve = []
        existing_data.append({"model": model_name, "dataset": dataset_name, "curve": curve})

    model.to(device)
    model.train()

    for epoch in range(len(curve), num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        if scheduler:
            scheduler.step()

        # Eval
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        curve.append(acc)
        print(f"📊 Epoch {epoch+1}/{num_epochs} | Val Acc: {acc:.4f}")

        # Save
        if curve_path:
            torch.save(existing_data, curve_path)
            print(f"💾 Saved progress after epoch {epoch+1}/{num_epochs}")

        model.train()

    return model, curve, existing_data


def build_feature_vector(synflow_score, curve_prefix):
    return np.array([synflow_score] + curve_prefix)

def train_regressor(X, y, save_path="regressor.pkl"):
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

def predict_final_accuracy(model_path, synflow_score, curve_prefix):
    model = joblib.load(model_path)
    X = build_feature_vector(synflow_score, curve_prefix).reshape(1, -1)
    return model.predict(X)[0]

def evaluate_regressor(model_path, X_test, y_test):
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    if len(y_test) < 2:
        print("⚠️ Only one sample — R² score is not well-defined.")
        return None
    return r2_score(y_test, preds)


__all__ = [
    "get_curve",
    "train_and_record_curve",
    "build_feature_vector",
    "train_regressor",
    "predict_final_accuracy",
    "evaluate_regressor"
]
