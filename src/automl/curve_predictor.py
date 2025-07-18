import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import numpy as np
import joblib

def build_feature_vector(synflow_score, curve_prefix):
    """
    Combine SynFlow + partial learning curve into one feature vector.
    """
    return np.array([synflow_score] + curve_prefix)

def train_regressor(X, y, save_path="regressor.pkl"):
    """
    Train Ridge regressor to predict final accuracy from partial curves + SynFlow.
    """
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    joblib.dump(model, save_path)
    print(f"Regressor saved to {save_path}")
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
    Evaluate R² of trained regressor.
    """
    model = joblib.load(model_path)
    preds = model.predict(X_test)
    return r2_score(y_test, preds)
