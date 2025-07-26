import argparse
import torch
import os
from src.automl.generate_curve_dataset import generate_curve_dataset
from src.automl.curve_predictor import train_regressor, evaluate_regressor
from src.automl.automl import AutoML
from src.automl.synflow import compute_synflow_score
from src.automl.models import build_model
import joblib
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['curve', 'synflow', 'train_regressor', 'eval_regressor', 'full_automl'],
                        help='Select mode')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashion', 'flowers', 'emotions'],
                        help='Select dataset')
    parser.add_argument('--models', nargs='+',
                        default=['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
                        help='Model names')
    parser.add_argument('--output', type=str, default='curve_dataset.pt',
                        help='Output file path')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--curve_epochs', type=int, default=20,
                        help='Number of epochs for learning curves')
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Root of dataset directory")
    parser.add_argument('--curve_path', default="curve_dataset.pt", help="Path to curve dataset (for regressor steps)")
    
    # NEW: Set regressor path based on dataset name
    args = parser.parse_args()
    args.regressor_path = f"regressor_{args.dataset}.pkl"

    if args.mode == "curve":
        generate_curve_dataset(
            dataset_name=args.dataset,
            model_names=args.models,
            dataset_dir=args.dataset_dir,
            output_path=args.output,
            device=args.device,
            n_epochs=args.curve_epochs
        )

    elif args.mode == "synflow":
        print(f"\n Calculating SynFlow scores for models in {args.curve_path}...\n")
        data = torch.load(args.curve_path)
        results = []

        for model_name, curve in data:
            in_channels = 1 if args.dataset == 'fashion' else 3
            model = build_model(model_name,
                    num_classes={"fashion": 10, "emotions": 7, "flowers": 102}[args.dataset],
                    in_channels=in_channels)
            input_size = {
              'fashion': (1, 1, 28, 28),
              'emotions': (1, 3, 224, 224),
              'flowers': (1, 3, 224, 224)
            }[args.dataset]
            syn_score = compute_synflow_score(model, input_size=input_size, device=args.device)

            results.append((model_name, curve, syn_score))
            print(f" {model_name} SynFlow score: {syn_score:.2e}")

        torch.save(results, args.curve_path)
        print(f"\n SynFlow scores saved back to {args.curve_path}")

    elif args.mode == "train_regressor":
        print(f"\n Training regressor using curve data from {args.curve_path}...\n")
        data = torch.load(args.curve_path)
        X, y = [], []

        for item in data:
            if len(item) == 3:
                model_name, curve, synflow = item
            else:
                raise ValueError("SynFlow score missing from curve data.")
            feature = np.array([synflow] + curve[:5])  # Use prefix length 5
            X.append(feature)
            y.append(curve[-1])  # Final epoch accuracy

        train_regressor(X, y, save_path=args.regressor_path)
        print(f"\n Regressor saved to {args.regressor_path}")

    elif args.mode == "eval_regressor":
        print(f"\n Evaluating regressor at {args.regressor_path}...\n")
        data = torch.load(args.curve_path)
        X, y = [], []

        for item in data:
            if len(item) == 3:
                model_name, curve, synflow = item
            else:
                raise ValueError("SynFlow score missing from curve data.")
            feature = np.array([synflow] + curve[:5])
            X.append(feature)
            y.append(curve[-1])

        r2 = evaluate_regressor(args.regressor_path, X, y)
        print(f" R² Score: {r2:.4f}")

    elif args.mode == "full_automl":
        automl = AutoML(device=args.device)
        best_model, best_score = automl.run(dataset_name=args.dataset)
        print(f" Best model: {best_model} with predicted final accuracy: {best_score:.4f}")

if __name__ == "__main__":
    main()
