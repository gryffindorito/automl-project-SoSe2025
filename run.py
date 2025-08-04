import argparse
import torch
import os
from src.automl.generate_curve_dataset import run_curve_mode, train_and_record_curve
from src.automl.curve_predictor import train_regressor, evaluate_regressor
from src.automl.curve_predictor import run_full_automl
from src.automl.automl import AutoML
from src.automl.models import build_model, get_model
from src.automl.dataloader_utils import get_dataloaders

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['curve', 'synflow', 'train_regressor', 'eval_regressor', 'full_automl', 'test_hpo'],
                        help='Select mode to run')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashion', 'flowers', 'emotions','intel','all'],
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
    parser.add_argument('--data_dir', type=str, default='/content/automl_data')
    parser.add_argument('--curve_path', default="curve_dataset.pt",
                        help="Path to saved curve dataset (.pt file)")
    parser.add_argument("--curve-dir", type=str, default="curve_data/", help="Directory to save curve dataset files")
    parser.add_argument('--regressor_path', type=str, default=None,
                    help="Optional override path for regressor .pkl file")

    args = parser.parse_args()
    if args.mode in ["full_automl", "train_regressor", "eval_regressor"]:
        args.regressor_path = args.regressor_path or f"regressor_{args.dataset}.pkl"
        print(f"📎 Using regressor: {args.regressor_path}")


    if args.mode == "curve":
        print(f"\n📈 Generating learning curves for dataset={args.dataset} | epochs={args.curve_epochs}")
        for model in args.models:
            args.model = model
            run_curve_mode(args)

    elif args.mode == "train_regressor":
        print(f"\n🎓 Training regressor using curve data from {args.curve_path}...\n")
        data = torch.load(args.curve_path)
        train_regressor(data, save_path=args.regressor_path)
        print(f"✅ Regressor saved to {args.regressor_path}")

    elif args.mode == "eval_regressor":
        print(f"\n📊 Evaluating regressor at {args.regressor_path}...\n")
        data = torch.load(args.curve_path)
        r2 = evaluate_regressor(args.regressor_path, data)
        print(f"📈 R² Score: {r2:.4f}")

    elif args.mode == "test_hpo":
        print(f"🔍 Running 10-epoch HPO test on dataset: {args.dataset}")

        for model_name in args.models:
            print(f"\n🚀 Testing {model_name} with HPO settings")

            train_loader, val_loader, _ = get_dataloaders(
                dataset_name=args.dataset,
                root=args.data_dir,
                batch_size=64
            )

            in_channels = next(iter(train_loader))[0].shape[1]
            dataset_obj = train_loader.dataset
            base = dataset_obj.dataset if hasattr(dataset_obj, 'dataset') else dataset_obj
            labels = [int(base[i][1]) for i in range(len(base))]
            num_classes = len(set(labels))

            model = get_model(model_name, num_classes=num_classes, in_channels=in_channels).to(args.device)

            _, curve, _ = train_and_record_curve(
                model,
                train_loader,
                val_loader,
                num_epochs=10,
                device=args.device
            )

            print(f"📈 Final accuracy after 10 epochs: {curve[-1]:.4f}")

    elif args.mode == "full_automl":
        print(f"\n🤖 Running full AutoML pipeline on {args.dataset}...\n")
        print(f"📎 Using regressor: {args.regressor_path}")
        print(f"📁 Using data_dir: {args.data_dir}")
        best_model, best_score = run_full_automl(
            dataset_name=args.dataset,
            regressor_path=args.regressor_path,
            device=args.device,
            data_dir=args.data_dir
        )
        print(f"\n🏆 Best model: {best_model} with predicted accuracy: {best_score:.4f}")

if __name__ == "__main__":
    main()
