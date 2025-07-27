import torch
import numpy as np
import xgboost as xgb
import joblib
from sklearn.metrics import r2_score

def build_feature_vector(synflow_score, curve_prefix):
    """
    Combine SynFlow + partial learning curve into one feature vector.
    """
    return np.array([synflow_score] + curve_prefix)

def train_regressor(X, y, save_path="regressor.pkl"):
    """
    Train basic XGBoost regressor on SynFlow + curve prefix to predict final accuracy.
    """
    model = xgb.XGBRegressor(
        n_estimators=50,
        max_depth=3,
        learning_rate=0.1,
        verbosity=0
    )
    model.fit(X, y)
    joblib.dump(model, save_path)
    print(f"XGBoost regressor saved to {save_path}")
    return model

def predict_final_accuracy(model_path, synflow_score, curve_prefix):
    """
    Use trained regressor to predict final validation accuracy.
    """
    model = joblib.load(model_path)
    X = build_feature_vector(synflow_score, curve_prefix).reshape(1, -1)
    return model.predict(X)[0]

def evaluate_regressor(model_path, X_test, y_test):
    """
    Evaluate regressor on test data. Note: With <2 samples, R² is meaningless.
    """
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    if len(y_test) < 2:
        print("  Only 1 sample — R² score is not well-defined.")
        return None
    return r2_score(y_test, preds)
