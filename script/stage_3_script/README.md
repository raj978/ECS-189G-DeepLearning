# Stage 3: Convolutional Neural Networks

This stage involves implementing CNN models to classify images from three different datasets: MNIST (handwritten digits), ORL (faces), and CIFAR (colored objects).

## Scripts Overview

Three scripts are provided:

1. `script_cnn.py` - Basic script that runs default CNN models on all three datasets
2. `script_cnn_experiments.py` - Advanced script that runs multiple model configurations for each dataset
3. `run_single_model.py` - Utility script to run a single CNN model with customizable parameters

## Requirements

- Python 3.6+
- PyTorch
- NumPy
- Matplotlib
- scikit-learn

## Dataset Information

The scripts expect three datasets in the `data/stage_3_data/` directory:
- `MNIST`: Handwritten digit dataset (28x28 grayscale images of digits 0-9)
- `ORL`: Face dataset (64x64 grayscale images of 40 different subjects)
- `CIFAR`: Colored object dataset (32x32 RGB images of 10 different classes)

## Running the Scripts

### Basic CNN Evaluation

To run the standard CNN configurations on all three datasets:

```bash
python script_cnn.py
```

This will:
- Train and evaluate a basic CNN model on MNIST dataset
- Train and evaluate a basic CNN model on ORL dataset
- Train and evaluate a basic CNN model on CIFAR dataset
- Train and evaluate a deeper CNN model on MNIST dataset
- Print a summary of results

### Advanced Experiments

To run multiple configurations and compare different CNN architectures:

```bash
python script_cnn_experiments.py
```

This will run multiple experiments with variations in:
- Number of convolutional layers
- Network depth
- Kernel size
- Padding
- Stride
- Pooling layers
- Hidden layer dimensions
- Dropout rates
- Optimization algorithms
- Loss functions

The experiments are organized by dataset and results are saved as both JSON metrics and learning curve plots.

### Running a Single Model

To run a single CNN model with customizable parameters:

```bash
python run_single_model.py --dataset MNIST --conv_layers 3 --fc_layers 2 --dropout 0.3
```

Available arguments:
- `--dataset`: Required, choose from MNIST, ORL, or CIFAR
- `--model_name`: Optional, custom name for the model
- `--max_epoch`: Number of training epochs (default: 20)
- `--batch_size`: Batch size for training (default: 64)
- `--learning_rate`: Learning rate (default: 0.001)
- `--optimizer`: Optimizer to use (adam, sgd, rmsprop) (default: adam)
- `--loss`: Loss function (cross_entropy, mse) (default: cross_entropy)
- `--conv_layers`: Number of convolutional layers (default: 2)
- `--fc_layers`: Number of fully connected layers (default: 2)
- `--dropout`: Dropout rate (default: 0.25)
- `--kernel_size`: Kernel size for conv layers (default: 3)
- `--padding`: Padding for conv layers (default: 1)
- `--stride`: Stride for conv layers (default: 1)
- `--pool_size`: Pooling size (default: 2)

## Output Files

The scripts generate several output files:

1. Model files: `models/stage_3/[model_name]_model.pt`
2. Training history: `models/stage_3/[model_name]_history.json`
3. Evaluation metrics: `result/stage_3_result/[model_name]_metrics.json`
4. Learning curves: `figures/stage_3/[model_name]_learning_curve.png`
5. Summary results: `result/stage_3_result/summary.json`
6. Summary comparison plot: `figures/stage_3/summary_comparison.png`

## CUDA Support

The scripts automatically detect if CUDA is available and use GPU acceleration when possible. The implementation supports optional CUDA acceleration (requirement 3-7) if you have compatible hardware.

## Customization

You can modify the hyperparameters and model configurations in the scripts to experiment with different architectures and training settings. 