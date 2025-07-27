import os
import torch
import numpy as np
from .models import build_model
from .trainer import train
from .synflow import compute_synflow_score
from .curve_predictor import predict_final_accuracy
from .dataloader_utils import get_dataloaders


class AutoML:
    def __init__(self, dataset_name, device='cpu', data_dir='data',
                 curve_path=None, regressor_path=None):
        self.dataset_name = dataset_name
        self.device = device
        self.data_dir = data_dir

        # Default to expected filenames if not provided
        self.curve_path = curve_path or f"curve_dataset_{dataset_name}.pt"
        self.regressor_path = regressor_path or f"regressor_{dataset_name}.pkl"

        if not os.path.exists(self.curve_path):
            raise FileNotFoundError(f"Missing curve dataset: {self.curve_path}")
        if not os.path.exists(self.regressor_path):
            raise FileNotFoundError(f"Missing trained regressor: {self.regressor_path}")

        self.num_classes = {
            "fashion": 10,
            "flowers": 102,
            "emotions": 7
        }[dataset_name]

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

        # Load SynFlow+curve dataset
        print(f"📦 Loading curve dataset from {self.curve_path}")
        data = torch.load(self.curve_path)
        synflow_dict = {model_name: synflow for model_name, _, synflow in data}
        curve_dict = {model_name: curve for model_name, curve, _ in data}

        for model_name in model_names:
            print(f"\n🔧 Evaluating {model_name}...")

            if model_name not in curve_dict or model_name not in synflow_dict:
                print(f"❌ Missing curve or synflow for {model_name}, skipping.")
                continue

            curve = curve_dict[model_name]
            synflow = synflow_dict[model_name]

            pred_acc = predict_final_accuracy(self.regressor_path, synflow, curve[:5])
            print(f"📈 Predicted final accuracy: {pred_acc:.4f}")

            if pred_acc > best_score:
                best_score = pred_acc
                best_model_name = model_name

        print(f"\n✅ Selected model: {best_model_name} with predicted acc {best_score:.4f}")
        return best_model_name, best_score
