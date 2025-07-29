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
                        help='Select mode to run')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashion', 'flowers', 'emotions'],
                        help='Dataset to use')
    parser.add_argument('--models', nargs='+',
                        default=['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
                        help='List of models to process')
    parser.add_argument('--output', type=str, default='curve_dataset.pt',
                        help='Output file path for curve dataset')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--curve_epochs', type=int, default=20,
                        help='Number of epochs to train models for learning curves')
    parser.add_argument('--data_dir', type=str, default='/content/automl_data',
                        help='Path to data directory')
    parser.add_argument('--curve_path', default="curve_dataset.pt",
                        help="Path to saved curve dataset (.pt file)")

    args = parser.parse_args()
    args.regressor_path = f"regressor_{args.dataset}.pkl"

    if args.mode == "curve":
        print(f"\n📈 Generating learning curves for dataset={args.dataset} | epochs={args.curve_epochs}")
        generate_curve_dataset(
            model_names=args.models,
            dataset_names=[args.dataset],
            save_path=args.output,
            dataset_dir=args.data_dir,
            device=args.device,
            max_epoch=args.curve_epochs
        )

    elif args.mode == "synflow":
        print(f"\n🔬 Calculating SynFlow scores for models in {args.curve_path}...\n")
        data = torch.load(args.curve_path)
        results = []

        for item in data:
            model_name = item["model"]
            curve = item["curve"]
            dataset_name = item["dataset"]

            in_channels = 1 if dataset_name == 'fashion' else 3
            model = build_model(
                model_name,
                num_classes={"fashion": 10, "emotions": 7, "flowers": 102}[dataset_name],
                in_channels=in_channels
            )
            input_size = {
                'fashion': (1, 1, 28, 28),
                'emotions': (1, 3, 224, 224),
                'flowers': (1, 3, 224, 224)
            }[dataset_name]

            syn_score = compute_synflow_score(model, input_size=input_size, device=args.device)
            item["synflow"] = syn_score
            results.append(item)
            print(f"✅ {model_name} on {dataset_name} → SynFlow score: {syn_score:.2e}")

        torch.save(results, args.curve_path)
        print(f"\n💾 SynFlow scores saved to {args.curve_path}")

    elif args.mode == "train_regressor":
        print(f"\n🎓 Training regressor using curve data from {args.curve_path}...\n")
        data = torch.load(args.curve_path)
        X, y = [], []

        for item in data:
            if "synflow" not in item:
                raise ValueError("SynFlow score missing from curve data.")
            feature = np.array([item["synflow"]] + item["curve"][:5])
            X.append(feature)
            y.append(item["curve"][-1])

        train_regressor(X, y, save_path=args.regressor_path)
        print(f"✅ Regressor saved to {args.regressor_path}")

    elif args.mode == "eval_regressor":
        print(f"\n📊 Evaluating regressor at {args.regressor_path}...\n")
        data = torch.load(args.curve_path)
        X, y = [], []

        for item in data:
            if "synflow" not in item:
                raise ValueError("SynFlow score missing from curve data.")
            feature = np.array([item["synflow"]] + item["curve"][:5])
            X.append(feature)
            y.append(item["curve"][-1])

        r2 = evaluate_regressor(args.regressor_path, X, y)
        print(f"📈 R² Score: {r2:.4f}")

    elif args.mode == "full_automl":
        print(f"\n🤖 Running full AutoML pipeline on {args.dataset}...\n")
        automl = AutoML(
            dataset_name=args.dataset,
            device=args.device,
            data_dir=args.data_dir,
            curve_path=args.curve_path,
            regressor_path=args.regressor_path
        )
        best_model, best_score = automl.run()
        print(f"\n🏆 Best model: {best_model} with predicted accuracy: {best_score:.4f}")

if __name__ == "__main__":
    main()
