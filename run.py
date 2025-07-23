import argparse
import torch
from src.automl.generate_curve_dataset import generate_curve_dataset
from src.automl.curve_predictor import train_regressor, evaluate_regressor
from src.automl.automl import AutoML

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, required=True,
                        choices=['curve', 'train_regressor', 'eval_regressor', 'full_automl'],
                        help='Select mode: curve / train_regressor / eval_regressor / full_automl')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['fashion', 'flowers', 'emotions'],
                        help='Select dataset')
    parser.add_argument('--models', nargs='+',
                        default=['resnet18', 'mobilenet_v2', 'efficientnet_b0'],
                        help='Model names for curve generation or automl')
    parser.add_argument('--output', type=str, default='curve_dataset.pt',
                        help='Output file for curve dataset')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use: cuda or cpu')
    parser.add_argument('--curve_epochs', type=int, default=20,
                        help='Number of epochs for learning curves')
    parser.add_argument("--data-dir", type=str, default="data", help="Path to dataset directory")
    parser.add_argument("--dataset_dir", type=str, default="data", help="Path to dataset root directory")



    args = parser.parse_args()

    if args.mode == "curve":
        generate_curve_dataset(
        dataset_name=args.dataset,
        model_names=args.models,
        dataset_dir=args.dataset_dir,
        output_path=args.output,
        device=args.device,
        n_epochs=args.curve_epochs
)

    elif args.mode == "train_regressor":
        train_regressor(args.output)

    elif args.mode == "eval_regressor":
        evaluate_regressor()

    elif args.mode == "full_automl":
        automl = AutoML(device=args.device)
        best_model, best_score = automl.run(dataset_name=args.dataset)
        print(f"Best model: {best_model} with predicted final accuracy: {best_score:.4f}")

if __name__ == "__main__":
    main()
