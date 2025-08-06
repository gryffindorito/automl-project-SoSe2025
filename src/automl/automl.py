import os
import torch
import numpy as np
from .models import build_model
from .trainer import train
from .curve_predictor import predict_final_accuracy
from .dataloader_utils import get_dataloaders

class AutoML:
    def __init__(self, dataset_name, device='cpu', data_dir='conteny/automl_data',
                 curve_path=None, regressor_path=None):
        self.dataset_name = dataset_name
        self.device = device
        self.data_dir = data_dir

        self.curve_path = curve_path or f"curve_dataset_{dataset_name}.pt"
        self.regressor_path = regressor_path or f"regressor_{dataset_name}.pkl"

        if not os.path.exists(self.curve_path):
            raise FileNotFoundError(f"Missing curve dataset: {self.curve_path}")
        if not os.path.exists(self.regressor_path):
            raise FileNotFoundError(f"Missing trained regressor: {self.regressor_path}")

        self.num_classes = {
            "fashion": 10,
            "flowers": 102,
            "emotions": 7,
            "intel": 6
        }.get(dataset_name, None)

    def run(self):
        print(f"\n🔍 Running AutoML for dataset: {self.dataset_name}")
        dataloaders = get_dataloaders(self.dataset_name, root=self.data_dir, batch_size=64)
        best_model, best_score = self.predict_best_model(dataloaders, [
            "resnet18", "mobilenet_v2", "efficientnet_b0"
        ])
        return best_model, best_score

    def predict_best_model(self, dataloaders, model_names):
        train_loader, val_loader, _ = dataloaders

        best_score = -1e9
        best_model_name = None

        # Load curve dataset
        print(f" Loading curve dataset from {self.curve_path}")
        data = torch.load(self.curve_path)

        for model_name in model_names:
            print(f"\n Evaluating {model_name}...")

            entry = next((d for d in data if d["model"] == model_name), None)
            if not entry:
                print(f" No curve data for {model_name}, skipping.")
                continue

            # Prepare item for regressor input
            item = {
                "model": model_name,
                "acc_curve": entry["acc_curve"],
                "loss_curve": entry["loss_curve"]
            }

            try:
                pred_acc = predict_final_accuracy(self.regressor_path, item)
                print(f" Predicted final accuracy: {pred_acc:.4f}")
            except Exception as e:
                print(f" Failed to predict for {model_name}: {e}")
                continue

            if pred_acc > best_score:
                best_score = pred_acc
                best_model_name = model_name

        print(f"\n Selected model: {best_model_name} with predicted acc {best_score:.4f}")
        return best_model_name, best_score
