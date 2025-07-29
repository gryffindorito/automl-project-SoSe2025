import torch
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score
from tqdm import tqdm
import torch.nn.functional as F

def get_curve(metrics_dict):
    """
    Extracts accuracy values across epochs from a metrics dictionary.
    Assumes dictionary format: {epoch_num: accuracy}.
    """
    return [metrics_dict[epoch] for epoch in sorted(metrics_dict)]


def train_and_record_curve(model, train_loader, val_loader, num_epochs=20, device="cuda"):
    """
    Train the model and record validation accuracy at each epoch.
    Returns the model and a metrics dict of val accuracies.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    metrics = {}

    model.to(device)
    model.train()

    for epoch in range(num_epochs):
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            loss.backward()
            optimizer.step()

        # Eval after epoch
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        acc = correct / total
        metrics[epoch] = acc
        print(f"📊 Epoch {epoch+1}/{num_epochs} | Val Acc: {acc:.4f}")
        model.train()

    return model, metrics


def build_feature_vector(synflow_score, curve_prefix):
    """
    Combine SynFlow + partial learning curve into one feature vector.
    """
    return np.array([synflow_score] + curve_prefix)


def train_regressor(X, y, save_path="regressor.pkl"):
    """
    Train XGBoost regressor on SynFlow + curve prefix to predict final accuracy.
    """
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
    """
    Use trained regressor to predict final accuracy.
    """
    model = joblib.load(model_path)
    X = build_feature_vector(synflow_score, curve_prefix).reshape(1, -1)
    return model.predict(X)[0]


def evaluate_regressor(model_path, X_test, y_test):
    """
    Evaluate the regressor on test data. R² requires at least 2 points.
    """
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
