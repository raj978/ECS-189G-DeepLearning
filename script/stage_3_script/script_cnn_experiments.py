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

# Function from the main script
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
    print(f"Running CNN model on {dataset_name} dataset with config: {model_config['model_name'] if model_config else 'default'}")
    print(f"{'='*50}")
    
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
    
    os.makedirs(model_save_dir, exist_ok=True)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)
    
    # Initialize dataset loader
    data_obj = Dataset_Loader("CNN", "")
    data_obj.dataset_source_folder_path = "data/stage_3_data"
    data_obj.dataset_name = dataset_name
    
    # Initialize CNN model with configuration parameters
    method_obj = Method_CNN("CNN", "")
    method_obj.model_path = os.path.join(model_save_dir, f"{model_name}_model.pt")
    method_obj.hist_path = os.path.join(model_save_dir, f"{model_name}_history.json")
    
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
    method_obj.kernel_size = config["kernel_size"]
    method_obj.stride = config["stride"]
    method_obj.padding = config["padding"]
    method_obj.pool_size = config["pool_size"]
    
    # Initialize result saver
    result_obj = Result_Saver("saver", "")
    result_dest_path = os.path.join(result_dir, f"{model_name}_prediction_result")
    result_obj.result_destination_file_path = result_dest_path
    result_obj.metrics_path = os.path.join(result_dir, f"{model_name}_metrics.json")
    
    # Initialize evaluator
    evaluate_obj = Evaluate_CNN("CNN-evaluation", "")
    evaluate_obj.plot_path = os.path.join(plot_dir, f"{model_name}_learning_curve.png")
    evaluate_obj.metrics_path = os.path.join(result_dir, f"{model_name}_metrics.json")
    
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


def run_mnist_experiments():
    """Run experiments on MNIST dataset"""
    print(f"\n{'='*50}")
    print(f"MNIST EXPERIMENTS")
    print(f"{'='*50}")
    
    results = {}
    
    # Base configuration
    base_config = {
        "model_name": "mnist_base",
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
    
    # Run base model
    print("\nRunning base model...")
    metrics = run_cnn_model("MNIST", base_config)
    results["base"] = metrics
    
    # Experiment 1: Deeper network
    deeper_config = base_config.copy()
    deeper_config.update({
        "model_name": "mnist_deeper",
        "conv_layers": 3,
        "fc_layers": 3,
        "fc_units": [512, 256, 128]
    })
    
    print("\nRunning deeper network...")
    metrics = run_cnn_model("MNIST", deeper_config)
    results["deeper"] = metrics
    
    # Experiment 2: Higher dropout
    dropout_config = base_config.copy()
    dropout_config.update({
        "model_name": "mnist_dropout",
        "dropout_rate": 0.5
    })
    
    print("\nRunning higher dropout...")
    metrics = run_cnn_model("MNIST", dropout_config)
    results["dropout"] = metrics
    
    # Experiment 3: Different learning rate
    lr_config = base_config.copy()
    lr_config.update({
        "model_name": "mnist_lr",
        "learning_rate": 5e-4
    })
    
    print("\nRunning different learning rate...")
    metrics = run_cnn_model("MNIST", lr_config)
    results["learning_rate"] = metrics
    
    # Experiment 4: Larger kernel size
    kernel_config = base_config.copy()
    kernel_config.update({
        "model_name": "mnist_kernel5",
        "kernel_size": 5,
        "padding": 2
    })
    
    print("\nRunning larger kernel size...")
    metrics = run_cnn_model("MNIST", kernel_config)
    results["kernel_size"] = metrics
    
    # Experiment 5: Different pooling
    pool_config = base_config.copy()
    pool_config.update({
        "model_name": "mnist_pool3",
        "pool_size": 3
    })
    
    print("\nRunning different pooling...")
    metrics = run_cnn_model("MNIST", pool_config)
    results["pool_size"] = metrics
    
    return results


def run_orl_experiments():
    """Run experiments on ORL dataset"""
    print(f"\n{'='*50}")
    print(f"ORL EXPERIMENTS")
    print(f"{'='*50}")
    
    results = {}
    
    # Base configuration
    base_config = {
        "model_name": "orl_base",
        "max_epoch": 20,
        "batch_size": 32,  # Smaller batch size due to smaller dataset
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
    
    # Run base model
    print("\nRunning base model...")
    metrics = run_cnn_model("ORL", base_config)
    results["base"] = metrics
    
    # Experiment 1: Deeper network
    deeper_config = base_config.copy()
    deeper_config.update({
        "model_name": "orl_deeper",
        "conv_layers": 3,
        "fc_layers": 3,
        "fc_units": [512, 256, 128]
    })
    
    print("\nRunning deeper network...")
    metrics = run_cnn_model("ORL", deeper_config)
    results["deeper"] = metrics
    
    # Experiment 2: Higher dropout
    dropout_config = base_config.copy()
    dropout_config.update({
        "model_name": "orl_dropout",
        "dropout_rate": 0.5
    })
    
    print("\nRunning higher dropout...")
    metrics = run_cnn_model("ORL", dropout_config)
    results["dropout"] = metrics
    
    # Experiment 3: Different learning rate
    lr_config = base_config.copy()
    lr_config.update({
        "model_name": "orl_lr",
        "learning_rate": 5e-4
    })
    
    print("\nRunning different learning rate...")
    metrics = run_cnn_model("ORL", lr_config)
    results["learning_rate"] = metrics
    
    # Experiment 4: Larger kernel size
    kernel_config = base_config.copy()
    kernel_config.update({
        "model_name": "orl_kernel5",
        "kernel_size": 5,
        "padding": 2
    })
    
    print("\nRunning larger kernel size...")
    metrics = run_cnn_model("ORL", kernel_config)
    results["kernel_size"] = metrics
    
    # Experiment 5: Different pooling
    pool_config = base_config.copy()
    pool_config.update({
        "model_name": "orl_pool3",
        "pool_size": 3
    })
    
    print("\nRunning different pooling...")
    metrics = run_cnn_model("ORL", pool_config)
    results["pool_size"] = metrics
    
    return results


def run_cifar_experiments():
    """Run experiments on CIFAR dataset"""
    print(f"\n{'='*50}")
    print(f"CIFAR EXPERIMENTS")
    print(f"{'='*50}")
    
    results = {}
    
    # Base configuration
    base_config = {
        "model_name": "cifar_base",
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
    
    # Run base model
    print("\nRunning base model...")
    metrics = run_cnn_model("CIFAR", base_config)
    results["base"] = metrics
    
    # Experiment 1: Deeper network
    deeper_config = base_config.copy()
    deeper_config.update({
        "model_name": "cifar_deeper",
        "conv_layers": 3,
        "fc_layers": 3,
        "fc_units": [512, 256, 128]
    })
    
    print("\nRunning deeper network...")
    metrics = run_cnn_model("CIFAR", deeper_config)
    results["deeper"] = metrics
    
    # Experiment 2: Higher dropout
    dropout_config = base_config.copy()
    dropout_config.update({
        "model_name": "cifar_dropout",
        "dropout_rate": 0.5
    })
    
    print("\nRunning higher dropout...")
    metrics = run_cnn_model("CIFAR", dropout_config)
    results["dropout"] = metrics
    
    # Experiment 3: Different learning rate
    lr_config = base_config.copy()
    lr_config.update({
        "model_name": "cifar_lr",
        "learning_rate": 5e-4
    })
    
    print("\nRunning different learning rate...")
    metrics = run_cnn_model("CIFAR", lr_config)
    results["learning_rate"] = metrics
    
    # Experiment 4: Larger kernel size
    kernel_config = base_config.copy()
    kernel_config.update({
        "model_name": "cifar_kernel5",
        "kernel_size": 5,
        "padding": 2
    })
    
    print("\nRunning larger kernel size...")
    metrics = run_cnn_model("CIFAR", kernel_config)
    results["kernel_size"] = metrics
    
    # Experiment 5: Different pooling
    pool_config = base_config.copy()
    pool_config.update({
        "model_name": "cifar_pool3",
        "pool_size": 3
    })
    
    print("\nRunning different pooling...")
    metrics = run_cnn_model("CIFAR", pool_config)
    results["pool_size"] = metrics
    
    return results


def save_summary_results(all_results):
    """Save a summary of all experiment results to a JSON file"""
    summary_path = "result/stage_3/summary.json"
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=4)
        
    print(f"Summary results saved to {summary_path}")
    
    # Create a bar chart of accuracy for all models
    accuracies = []
    model_names = []
    
    for dataset, models in all_results.items():
        for model_name, metrics in models.items():
            model_names.append(model_name)
            accuracies.append(metrics['accuracy'])
    
    plt.figure(figsize=(12, 8))
    plt.bar(model_names, accuracies)
    plt.xticks(rotation=45, ha='right')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison Across All Models')
    plt.tight_layout()
    
    summary_plot_path = "figures/stage_3/summary_comparison.png"
    plt.savefig(summary_plot_path)
    print(f"Summary plot saved to {summary_plot_path}")
    plt.close()


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run CNN experiments on different datasets')
    parser.add_argument('--datasets', nargs='+', default=['MNIST', 'ORL', 'CIFAR'],
                      help='List of datasets to process (MNIST, ORL, CIFAR)')
    parser.add_argument('--use_cuda', action='store_true',
                      help='Use CUDA if available')
    args = parser.parse_args()

    # Set device
    device = torch.device('cuda' if args.use_cuda and torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Run experiments for each dataset
    all_results = {}
    
    if 'MNIST' in args.datasets:
        mnist_results = run_mnist_experiments()
        all_results['MNIST'] = mnist_results
    
    if 'ORL' in args.datasets:
        orl_results = run_orl_experiments()
        all_results['ORL'] = orl_results
    
    if 'CIFAR' in args.datasets:
        cifar_results = run_cifar_experiments()
        all_results['CIFAR'] = cifar_results
    
    # Save all results
    results_dir = "result/stage_3_result"
    os.makedirs(results_dir, exist_ok=True)
    
    with open(os.path.join(results_dir, "all_experiment_results.json"), 'w') as f:
        json.dump(all_results, f, indent=4)
    
    print(f"\nAll experiment results saved to {os.path.join(results_dir, 'all_experiment_results.json')}")


if __name__ == '__main__':
    main() 