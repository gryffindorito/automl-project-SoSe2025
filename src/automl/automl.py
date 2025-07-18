import torch
import numpy as np
from pathlib import Path

from .models import build_model
from .trainer import train
from .synflow import compute_synflow_score
from .curve_predictor import predict_final_accuracy

class AutoML:
    def __init__(self, dataset_name, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dataset_name = dataset_name
        self.num_classes = {
            "fashion": 10,
            "flowers": 102,
            "emotions": 7
        }[dataset_name]

        # Load trained regressor
        self.regressor_path = Path("regressor.pkl")
        if not self.regressor_path.exists():
            raise FileNotFoundError("You must first train the regressor before using AutoML.")

    def predict_best_model(self, dataloaders, model_names):
        """
        Pick the best model based on extrapolated performance.
        """
        best_score = -1e9
        best_model_name = None

        for model_name in model_names:
            print(f"Evaluating {model_name}...")
            model = build_model(model_name, self.num_classes).to(self.device)

            # Compute SynFlow score
            synflow = compute_synflow_score(model, input_size=(1, 1, 28, 28), device=self.device)

            # Get learning curve prefix (e.g., first 5 epochs)
            _, val_curve = train(model, dataloaders, epochs=5, device=self.device)
            pred_acc = predict_final_accuracy(self.regressor_path, synflow, val_curve[:5])

            print(f"Predicted final accuracy for {model_name}: {pred_acc:.4f}")

            if pred_acc > best_score:
                best_score = pred_acc
                best_model_name = model_name

        print(f"\n✅ Selected model: {best_model_name} with predicted acc {best_score:.4f}")
        return best_model_name
