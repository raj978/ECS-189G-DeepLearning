import os
import sys
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
from datetime import datetime
import argparse

# Add the project root directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(__file__, "..", "..", "..")))

# Import stage 3 components
from local_code.stage_3_code.Dataset_Loader import Dataset_Loader
from local_code.stage_3_code.Method_CNN import Method_CNN
from local_code.stage_3_code.Evaluate_CNN import Evaluate_CNN
from local_code.stage_3_code.Result_Saver import Result_Saver
from local_code.stage_3_code.Setting_CNN import Setting_CNN

# Function to configure and run a CNN model for a specific dataset
def run_cnn_model(dataset_name, model_config=None):
    """
    Configure and run a CNN model for a specific dataset
    
    Args:
        dataset_name: Name of the dataset (MNIST, ORL, or CIFAR)
        model_config: Dictionary with model configuration parameters
    
    Returns:
        metrics: Dictionary with evaluation metrics
    """
    print(f"\n{'='*50}")
    print(f"Running CNN model on {dataset_name} dataset")
    print(f"{'='*50}")
    
    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Default model configuration
    default_config = {
        "model_name": f"cnn_{dataset_name.lower()}_default",
        "max_epoch": 20,
        "batch_size": 64,
        "learning_rate": 1e-3,
        "optimizer_name": "adam",
        "loss_fn_name": "cross_entropy",
        "conv_layers": 2,
        "fc_layers": 2,
        "fc_units": [256, 128],
        "dropout_rate": 0.25,
        "kernel_size": 3,
        "stride": 1,
        "padding": 1,
        "pool_size": 2
    }
    
    # Update with custom configuration if provided
    config = default_config.copy()
    if model_config:
        config.update(model_config)
    
    # Model name for saving files
    model_name = config["model_name"]
    
    # Create directories for saving results
    model_save_dir = "models/stage_3"
    result_dir = "result/stage_3_result"
    plot_dir = "figures/stage_3"
    
    # Ensure all directories exist
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Create dataset-specific subdirectories
    dataset_model_dir = os.path.join(model_save_dir, dataset_name.lower())
    dataset_result_dir = os.path.join(result_dir, dataset_name.lower())
    dataset_plot_dir = os.path.join(plot_dir, dataset_name.lower())
    
    os.makedirs(dataset_model_dir, exist_ok=True)
    os.makedirs(dataset_result_dir, exist_ok=True)
    os.makedirs(dataset_plot_dir, exist_ok=True)
    
    # Initialize dataset loader
    data_obj = Dataset_Loader("CNN", "")
    data_obj.dataset_source_folder_path = "data/stage_3_data"
    data_obj.dataset_name = dataset_name
    
    # Enable data augmentation for CIFAR
    if dataset_name == "CIFAR":
        data_obj.use_augmentation = True
        print("Enabled data augmentation for CIFAR")
    
    # Initialize CNN model with configuration parameters
    method_obj = Method_CNN("CNN", "")
    method_obj.model_path = os.path.join(dataset_model_dir, f"{model_name}_model.pt")
    method_obj.hist_path = os.path.join(dataset_model_dir, f"{model_name}_history.json")
    
    # Set dataset-specific parameters based on the dataset
    if dataset_name == "MNIST":
        # MNIST is 28x28 grayscale (1 channel) with 10 classes (0-9)
        method_obj.in_channels = 1
        method_obj.num_classes = 10
        method_obj.input_height = 28
        method_obj.input_width = 28
    elif dataset_name == "ORL":
        # ORL is 112x92 grayscale (1 channel) with 40 classes (1-40)
        method_obj.in_channels = 1
        method_obj.num_classes = 40
        method_obj.input_height = 112
        method_obj.input_width = 92
    elif dataset_name == "CIFAR":
        # CIFAR is 32x32 RGB (3 channels) with 10 classes (0-9)
        method_obj.in_channels = 3
        method_obj.num_classes = 10
        method_obj.input_height = 32
        method_obj.input_width = 32
        # Enhanced CIFAR configuration
        method_obj.conv_layers = 3  # More conv layers
        method_obj.fc_layers = 3    # More FC layers
        method_obj.fc_units = [512, 256, 128]  # Larger FC layers
        method_obj.dropout_rate = 0.3  # Slightly higher dropout
        method_obj.kernel_size = 3
        method_obj.stride = 1
        method_obj.padding = 1
        method_obj.pool_size = 2
        method_obj.learning_rate = 5e-4  # Lower learning rate
        method_obj.max_epoch = 50  # More epochs
        method_obj.batch_size = 128  # Larger batch size
    
    # Set model configuration
    method_obj.max_epoch = config["max_epoch"]
    method_obj.batch_size = config["batch_size"]
    method_obj.learning_rate = config["learning_rate"]
    method_obj.optimizer_name = config["optimizer_name"]
    method_obj.loss_fn_name = config["loss_fn_name"]
    method_obj.conv_layers = config["conv_layers"]
    method_obj.fc_layers = config["fc_layers"]
    if "fc_units" in config:
        method_obj.fc_units = config["fc_units"]
    method_obj.dropout_rate = config["dropout_rate"]
    
    # Initialize result saver
    result_obj = Result_Saver("saver", "")
    result_dest_path = os.path.join(dataset_result_dir, f"{model_name}_prediction_result")
    result_obj.result_destination_file_path = result_dest_path
    result_obj.metrics_path = os.path.join(dataset_result_dir, f"{model_name}_metrics.json")
    
    # Initialize evaluator
    evaluate_obj = Evaluate_CNN("CNN-evaluation", "")
    evaluate_obj.plot_path = os.path.join(dataset_plot_dir, f"{model_name}_learning_curve.png")
    evaluate_obj.metrics_path = os.path.join(dataset_result_dir, f"{model_name}_metrics.json")
    
    # Initialize setting
    setting_obj = Setting_CNN("CNN-setting", "")
    
    # Prepare the setting
    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()
    
    # Run the setting
    metrics = setting_obj.load_run_save_evaluate()
    
    # Print final results summary
    print(f"\n{'='*50}")
    print(f"RESULTS SUMMARY for {dataset_name} - {model_name}")
    print(f"{'='*50}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"F1 Score: {metrics['f1']:.4f}")
    print(f"Model saved to: {method_obj.model_path}")
    print(f"Results saved to: {result_obj.result_destination_file_path}")
    print(f"Metrics saved to: {result_obj.metrics_path}")
    print(f"Learning curve saved to: {evaluate_obj.plot_path}")
    print(f"{'='*50}")
    
    return metrics

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run CNN models on image datasets')
    parser.add_argument('--datasets', nargs='+', choices=['MNIST', 'ORL', 'CIFAR'],
                      default=['MNIST', 'ORL', 'CIFAR'],
                      help='Datasets to run (default: all)')
    parser.add_argument('--epochs', type=int, default=20,
                      help='Number of epochs for training (default: 20)')
    parser.add_argument('--batch-size', type=int, default=64,
                      help='Batch size for training (default: 64)')
    parser.add_argument('--quick-test', action='store_true',
                      help='Run a quick test with reduced epochs')
    parser.add_argument('--use_cuda', action='store_true',
                      help='Use CUDA if available')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    # Set random seeds for reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
    
    # Check for CUDA
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Enable cuDNN benchmark for faster training if using CUDA
    if args.use_cuda and torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        print("CUDA acceleration enabled")
    
    # Adjust epochs for quick test
    if args.quick_test:
        args.epochs = 2
        print("Running in quick test mode with 2 epochs")
    
    # Store results
    results = {}
    
    # Run models for selected datasets
    for dataset in args.datasets:
        print(f"\n{'='*50}")
        print(f"Starting {dataset} model training")
        print(f"{'='*50}")
        
        # Configure model
        if dataset == "CIFAR":
            # Set CIFAR-specific epochs (50 default unless quick test)
            cifar_epochs = 50 if not args.quick_test else 2
            print(f"Using {cifar_epochs} epochs for CIFAR model training")
            
            # Simplified but enhanced CIFAR configuration
            model_config = {
                "model_name": f"cnn_{dataset.lower()}_enhanced",
                "max_epoch": cifar_epochs,  # Using CIFAR-specific epochs instead of args.epochs
                "batch_size": 128,
                "learning_rate": 1e-3,
                "optimizer_name": "adam",  # Back to adam for reliability
                "loss_fn_name": "cross_entropy",
                "conv_layers": 3,  # Deeper network but not too complex
                "fc_layers": 3,
                "fc_units": [512, 256, 128],  # Increased capacity
                "dropout_rate": 0.3,  # Moderate dropout
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool_size": 2,
                "use_resnet": False,  # Disable ResNet for now
                "use_adv_optim": False,  # Disable advanced optimizer
                "weight_decay": 1e-4,  # Still use weight decay but milder
                "momentum": 0.9,
                "label_smoothing": 0.0  # Disable label smoothing
            }
        else:
            # Default configuration for other datasets
            model_config = {
                "model_name": f"cnn_{dataset.lower()}_default",
                "max_epoch": args.epochs if not args.quick_test else 2,
                "batch_size": args.batch_size,
                "learning_rate": 1e-3,
                "optimizer_name": "adam",
                "loss_fn_name": "cross_entropy",
                "conv_layers": 2,
                "fc_layers": 2,
                "fc_units": [256, 128],
                "dropout_rate": 0.25,
                "kernel_size": 3,
                "stride": 1,
                "padding": 1,
                "pool_size": 2
            }
        
        # Run model
        try:
            metrics = run_cnn_model(dataset_name=dataset, model_config=model_config)
            results[dataset] = metrics
            print(f"\n{dataset} model completed successfully!")
        except Exception as e:
            print(f"\nError running {dataset} model: {str(e)}")
            continue
    
    # Print summary of results
    if results:
        print("\n\n" + "="*50)
        print("SUMMARY OF RESULTS")
        print("="*50)
        for dataset, metrics in results.items():
            print(f"{dataset}: Accuracy = {metrics['accuracy']:.4f}, F1 = {metrics['f1']:.4f}")
    else:
        print("\nNo results to display - all models failed")

if __name__ == "__main__":
    main() 