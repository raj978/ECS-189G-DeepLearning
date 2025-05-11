"""
Concrete SettingModule class for the CNN classification task
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
import numpy as np
import json
from torch.utils.data import DataLoader, TensorDataset
from local_code.base_class.setting import setting


class Setting_CNN(setting):
    """
    Setting class for CNN image classification
    
    Attributes:
        data_obj: Dataset_Loader object
        method_obj: Method_CNN object
        result_obj: Result_Saver object
        evaluate_obj: Evaluate_CNN object
    """
    
    dataset_name = None
    
    def __init__(self, sName=None, sDescription=None):
        super().__init__(sName, sDescription)
        self.data_obj = None
        self.method_obj = None
        self.result_obj = None
        self.evaluate_obj = None
    
    def prepare(self, data_obj, method_obj, result_obj, evaluate_obj):
        """
        Prepare the setting with required objects
        
        Args:
            data_obj: Dataset_Loader object for loading data
            method_obj: Method_CNN object implementing the CNN model
            result_obj: Result_Saver object for saving results
            evaluate_obj: Evaluate_CNN object for evaluating results
        """
        self.data_obj = data_obj
        self.method_obj = method_obj
        self.result_obj = result_obj
        self.evaluate_obj = evaluate_obj
        
        # Set dataset name
        self.dataset_name = data_obj.dataset_name
    
    def print_setup_summary(self):
        """
        Print a summary of the experimental setup
        """
        print("=" * 50)
        print("EXPERIMENTAL SETUP SUMMARY")
        print("=" * 50)
        print(f"Dataset: {self.dataset_name}")
        print(f"Model: {self.method_obj.name}")
        print(f"Optimizer: {self.method_obj.optimizer_name}")
        print(f"Loss Function: {self.method_obj.loss_fn_name}")
        print(f"Convolutional Layers: {self.method_obj.conv_layers}")
        print(f"Fully Connected Layers: {self.method_obj.fc_layers}")
        print(f"Learning Rate: {self.method_obj.learning_rate}")
        print(f"Batch Size: {self.method_obj.batch_size}")
        print(f"Max Epochs: {self.method_obj.max_epoch}")
        print(f"Dropout Rate: {self.method_obj.dropout_rate}")
        print("=" * 50)
    
    def load_run_save_evaluate(self):
        """
        Load data, run the CNN model, save and evaluate results
        
        Returns:
            A dictionary containing evaluation metrics
        """
        # Load data
        print("Loading data...")
        self.data_obj.dataset_source_folder_path = self.data_obj.dataset_source_folder_path
        data = self.data_obj.load()
        
        # Check device
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # Setup method
        self.method_obj.data = data
        self.method_obj.to(device)
        
        # Train model
        print("Training model...")
        history = self.method_obj.learning()
        
        # Test model
        print("Testing model...")
        pred_y = self.method_obj.testing()
        
        # Save results
        print("Saving results...")
        result_data = {
            'pred_y': pred_y,
            'true_y': data['test']['y']
        }
        self.result_obj.data = result_data
        self.result_obj.save()
        
        # Evaluate results
        print("Evaluating results...")
        eval_data = {
            'pred_y': pred_y,
            'true_y': data['test']['y'],
            'history': history
        }
        self.evaluate_obj.data = eval_data
        metrics = self.evaluate_obj.evaluate()
        
        # Save metrics to the result_obj for future reference
        self.result_obj.metrics = metrics
        
        # Return metrics
        return metrics
    
    def run(self):
        """
        Run the full experimental pipeline
        
        Returns:
            A dictionary containing evaluation metrics
        """
        print("Setting up experiment...")
        
        # Print setup
        self.print_setup_summary()
        
        # Run the pipeline
        metrics = self.load_run_save_evaluate()
        
        # Print summary of results
        print("\n" + "=" * 50)
        print("RESULTS SUMMARY")
        print("=" * 50)
        print(f"Dataset: {self.dataset_name}")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro-Precision: {metrics['precision']:.4f}")
        print(f"Macro-Recall: {metrics['recall']:.4f}")
        print(f"Macro-F1: {metrics['f1']:.4f}")
        print(f"Weighted-F1: {metrics['f1_weighted']:.4f}")
        print("=" * 50)
        
        return metrics 