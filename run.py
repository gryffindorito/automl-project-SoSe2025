import argparse
import torch
import os
import numpy as np
import joblib

from src.automl.generate_curve_dataset import generate_curve_dataset
from src.automl.curve_predictor import train_regressor, evaluate_regressor
from src.automl.automl import AutoML
from src.automl.synflow import compute_synflow_score
from src.automl.models import build_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['curve', 'synflow', 'train_regressor', 'eval_regressor', 'full_automl'],
                        help='Select mode to run')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashion', 'flowers', 'emotions'],
                        help='Dataset name')
    parser.add_argument('--models', nargs='+',
                        default=['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
                        help='List of model architectures')
    parser.add_argument('--curve_path', type=str, default='curve_dataset.pt',
                        help='Path to curve dataset file')
    parser.add_argument('--regressor_path', type=str, default=None,
                        help='Path to save or load regressor model')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use')
    parser.add_argument('--n_epochs', type=int, default=20,
                        help='Max epochs to run for curve generation')
    parser.add_argument("--data_dir", type=str, default='/content/automl_data',
                        help="Path to dataset root folder")

    args = parser.parse_args()
    
    if args.regressor_path is None:
        args.regressor_path = f"regressor_{args.dataset}.pkl"

    if args.mode == "curve":
        print(f"\n📈 Generating learning curves for dataset={args.dataset} | epochs={args.n_epochs}")
        generate_curve_dataset(
        dataset_names=[args.dataset],  # 👈 wrap in list
        model_names=args.models,
        dataset_dir=args.data_dir,
        save_path=args.output,
        device=args.device,
        max_epoch=args.curve_epochs
    )

    elif args.mode == "synflow":
        print(f"\n🔍 Calculating SynFlow scores for models in {args.curve_path}...\n")
        data = torch.load(args.curve_path)
        results = []

        for model_name, curve in data:
            in_channels = 1 if args.dataset == 'fashion' else 3
            model = build_model(
                model_name,
                num_classes={"fashion": 10, "emotions": 7, "flowers": 102}[args.dataset],
                in_channels=in_channels
            )
            input_size = {
                'fashion': (1, 1, 28, 28),
                'emotions': (1, 1, 48, 48),
                'flowers': (1, 3, 224, 224)
            }[args.dataset]

            score = compute_synflow_score(model, input_size=input_size, device=args.device)
            results.append((model_name, curve, score))
            print(f" ✅ {model_name} SynFlow Score: {score:.2e}")

        torch.save(results, args.curve_path)
        print(f"\n💾 SynFlow scores saved to {args.curve_path}")

    elif args.mode == "train_regressor":
        print(f"\n🎯 Training XGBoost regressor using curve data from {args.curve_path}")
        data = torch.load(args.curve_path)
        X, y = [], []

        for item in data:
            if len(item) == 3:
                model_name, curve, synflow = item
            else:
                raise ValueError("Missing SynFlow score in curve dataset.")
            X.append(np.array([synflow] + curve[:5]))
            y.append(curve[-1])  # Final epoch accuracy

        train_regressor(X, y, save_path=args.regressor_path)
        print(f"✅ Regressor saved to {args.regressor_path}")

    elif args.mode == "eval_regressor":
        print(f"\n🧪 Evaluating regressor from {args.regressor_path}...")
        data = torch.load(args.curve_path)
        X, y = [], []

        for item in data:
            if len(item) == 3:
                model_name, curve, synflow = item
            else:
                raise ValueError("Missing SynFlow score in curve dataset.")
            X.append(np.array([synflow] + curve[:5]))
            y.append(curve[-1])

        r2 = evaluate_regressor(args.regressor_path, X, y)
        print(f"📊 R² Score: {r2:.4f}")

    elif args.mode == "full_automl":
        print(f"\n🤖 Launching full AutoML pipeline for dataset: {args.dataset}")
        automl = AutoML(
            dataset_name=args.dataset,
            device=args.device,
            data_dir=args.data_dir,
            curve_path=args.curve_path,
            regressor_path=args.regressor_path
        )
        best_model, best_score = automl.run()
        print(f"\n🏆 Best model: {best_model} with predicted final accuracy: {best_score:.4f}")

if __name__ == "__main__":
    main()
