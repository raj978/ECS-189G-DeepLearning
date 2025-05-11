import os
import sys
import argparse
import numpy as np
import torch

# Add the project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# Import the run_cnn_model function from the main script
from script.stage_3_script.script_cnn import run_cnn_model

# Import stage 3 components
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Evaluate_CNN import Evaluate_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_CNN import Setting_CNN

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run a single CNN model on a specific dataset')
    
    # Required arguments
    parser.add_argument('--dataset', type=str, required=True, choices=['MNIST', 'ORL', 'CIFAR'],
                        help='Dataset to use (MNIST, ORL, or CIFAR)')
    
    # Optional arguments with default values
    parser.add_argument('--model_name', type=str, default=None,
                       help='Name for the model (default: cnn_<dataset>_custom)')
    parser.add_argument('--max_epoch', type=int, default=20,
                       help='Maximum number of epochs for training (default: 20)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate (default: 0.001)')
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd', 'rmsprop'],
                       help='Optimizer to use (default: adam)')
    parser.add_argument('--loss', type=str, default='cross_entropy', choices=['cross_entropy', 'mse'],
                       help='Loss function to use (default: cross_entropy)')
    parser.add_argument('--conv_layers', type=int, default=2,
                       help='Number of convolutional layers (default: 2)')
    parser.add_argument('--fc_layers', type=int, default=2,
                       help='Number of fully connected layers (default: 2)')
    parser.add_argument('--dropout', type=float, default=0.25,
                       help='Dropout rate (default: 0.25)')
    parser.add_argument('--kernel_size', type=int, default=3,
                       help='Kernel size for convolutional layers (default: 3)')
    parser.add_argument('--padding', type=int, default=1,
                       help='Padding for convolutional layers (default: 1)')
    parser.add_argument('--stride', type=int, default=1,
                       help='Stride for convolutional layers (default: 1)')
    parser.add_argument('--pool_size', type=int, default=2,
                       help='Pooling size (default: 2)')
    
    return parser.parse_args()

def main():
    """Main function to run a single CNN model"""
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Check for CUDA
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model configuration
    model_config = {
        "model_name": args.model_name or f"cnn_{args.dataset.lower()}_custom",
        "max_epoch": args.max_epoch,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "optimizer_name": args.optimizer,
        "loss_fn_name": args.loss,
        "conv_layers": args.conv_layers,
        "fc_layers": args.fc_layers,
        "dropout_rate": args.dropout,
        "kernel_size": args.kernel_size,
        "padding": args.padding,
        "stride": args.stride,
        "pool_size": args.pool_size,
    }
    
    # Set dataset-specific FC units based on dataset
    if args.dataset == "CIFAR":
        model_config["fc_units"] = [512, 256][:args.fc_layers]
    else:
        model_config["fc_units"] = [256, 128][:args.fc_layers]
    
    # Create directories for saving results
    model_save_dir = "models/stage_3"
    result_dir = "result/stage_3"
    plot_dir = "figures/stage_3"
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Model name for saving files
    model_name = model_config["model_name"]
    
    # Initialize dataset loader
    data_obj = Dataset_Loader("CNN", "")
    data_obj.dataset_source_folder_path = "data/stage_3_data"
    data_obj.dataset_name = args.dataset
    
    # Initialize CNN model with configuration parameters
    method_obj = Method_CNN("CNN", "")
    method_obj.model_path = os.path.join(model_save_dir, f"{model_name}_model.pt")
    method_obj.hist_path = os.path.join(model_save_dir, f"{model_name}_history.json")
    
    # Export model to ONNX format for visualization
    onnx_path = os.path.join(model_save_dir, f"{model_name}_model.onnx")
    method_obj.export_to_onnx(onnx_path)
    
    # Set dataset-specific parameters based on the dataset
    if args.dataset == "MNIST":
        method_obj.in_channels = 1
        method_obj.num_classes = 10
        method_obj.input_height = 28
        method_obj.input_width = 28
    elif args.dataset == "ORL":
        method_obj.in_channels = 1
        method_obj.num_classes = 40
        method_obj.input_height = 64
        method_obj.input_width = 64
    elif args.dataset == "CIFAR":
        method_obj.in_channels = 3
        method_obj.num_classes = 10
        method_obj.input_height = 32
        method_obj.input_width = 32
    
    # Set model configuration
    method_obj.max_epoch = model_config["max_epoch"]
    method_obj.batch_size = model_config["batch_size"]
    method_obj.learning_rate = model_config["learning_rate"]
    method_obj.optimizer_name = model_config["optimizer_name"]
    method_obj.loss_fn_name = model_config["loss_fn_name"]
    method_obj.conv_layers = model_config["conv_layers"]
    method_obj.fc_layers = model_config["fc_layers"]
    method_obj.fc_units = model_config["fc_units"]
    method_obj.dropout_rate = model_config["dropout_rate"]
    method_obj.kernel_size = model_config["kernel_size"]
    method_obj.stride = model_config["stride"]
    method_obj.padding = model_config["padding"]
    method_obj.pool_size = model_config["pool_size"]
    
    # Initialize result saver
    result_obj = Result_Saver("saver", "")
    result_dest_path = os.path.join(result_dir, f"{model_name}_prediction_result")
    result_obj.result_destination_file_path = result_dest_path
    result_obj.metrics_path = os.path.join(result_dir, f"{model_name}_metrics.json")
    
    # Initialize evaluator
    evaluate_obj = Evaluate_CNN("CNN-evaluation", "")
    evaluate_obj.plot_path = os.path.join(plot_dir, f"{model_name}_learning_curve.png")
    evaluate_obj.metrics_path = result_obj.metrics_path
    
    # Initialize setting
    setting_obj = Setting_CNN("CNN-setting", "")
    
    # Prepare the setting
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    
    # Run the setting
    metrics = setting_obj.load_run_save_evaluate()
    
    # Print summary of results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    print(f"Model: {model_config['model_name']}")
    print(f"Dataset: {args.dataset}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro-F1: {metrics['f1']:.4f}")
    print(f"Weighted-F1: {metrics['f1_weighted']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")

if __name__ == "__main__":
    main() 