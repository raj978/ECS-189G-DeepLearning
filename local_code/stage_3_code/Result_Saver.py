"""
Concrete ResultSaver class for saving classification results
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import pickle
import numpy as np
import torch
import os
import json
from local_code.base_class.result import result


class Result_Saver(result):
    """
    Result Saver for CNN model predictions and metrics
    
    Attributes:
        data: Dictionary containing prediction data to save
        result_destination_file_path: Path to save the prediction results
        metrics_path: Path to save the evaluation metrics
        metrics: Dictionary of evaluation metrics
    """
    
    data = None
    result_destination_file_path = None
    metrics_path = None
    metrics = None
    
    def __init__(self, sName=None, sDescription=None):
        """Initialize the Result_Saver"""
        super().__init__(sName, sDescription)
    
    def save(self):
        """Save results to file"""
        print(f"Saving results...")
        
        # Create directory if it doesn't exist
        if self.result_destination_file_path:
            directory = os.path.dirname(self.result_destination_file_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
        
        # Extract data
        pred_y = self.data['pred_y']
        true_y = self.data['true_y']
        
        # Convert tensors to numpy arrays if needed
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.cpu().numpy()
        if isinstance(true_y, torch.Tensor):
            true_y = true_y.cpu().numpy()
        
        # Format results
        results = {
            'pred_y': pred_y.tolist() if isinstance(pred_y, np.ndarray) else pred_y,
            'true_y': true_y.tolist() if isinstance(true_y, np.ndarray) else true_y
        }
        
        # Save results as a pickle file
        try:
            with open(self.result_destination_file_path, 'wb') as f:
                pickle.dump(results, f)
            print(f"Prediction results saved to {self.result_destination_file_path}")
            
            # Also save as JSON for easier reading
            json_path = self.result_destination_file_path + '.json'
            with open(json_path, 'w') as f:
                json.dump(results, f, indent=4)
            print(f"Prediction results also saved to {json_path}")
        except Exception as e:
            print(f"Error saving results: {str(e)}")
        
        # Save metrics if available
        if self.metrics is not None and self.metrics_path is not None:
            try:
                # Create directory if it doesn't exist
                os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
                
                # Convert numpy values to Python native types for JSON serialization
                clean_metrics = {}
                for key, value in self.metrics.items():
                    if isinstance(value, np.ndarray):
                        clean_metrics[key] = value.tolist()
                    elif isinstance(value, np.floating):
                        clean_metrics[key] = float(value)
                    elif isinstance(value, np.integer):
                        clean_metrics[key] = int(value)
                    else:
                        clean_metrics[key] = value
                
                # Save metrics as JSON
                with open(self.metrics_path, 'w') as f:
                    json.dump(clean_metrics, f, indent=4)
                print(f"Metrics saved to {self.metrics_path}")
            except Exception as e:
                print(f"Error saving metrics: {str(e)}")
        
        return None
    
    def load(self):
        """
        Load the saved prediction results from file
        
        Returns:
            A dictionary containing the prediction results
        """
        if os.path.exists(self.result_destination_file_path):
            with open(self.result_destination_file_path, 'rb') as f:
                data = pickle.load(f)
            
            print(f"Prediction results loaded from {self.result_destination_file_path}")
            return data
        else:
            print(f"No saved results found at {self.result_destination_file_path}")
            return None
    
    def load_metrics(self):
        """
        Load the saved evaluation metrics from file
        
        Returns:
            A dictionary containing the evaluation metrics
        """
        if self.metrics_path and os.path.exists(self.metrics_path):
            with open(self.metrics_path, 'r') as f:
                metrics = json.load(f)
            
            print(f"Metrics loaded from {self.metrics_path}")
            return metrics
        else:
            print(f"No saved metrics found at {self.metrics_path}")
            return None 