"""
Concrete Evaluate class for classification results
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import numpy as np
import matplotlib.pyplot as plt
import os
import json
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from local_code.base_class.evaluate import evaluate


class Evaluate_CNN(evaluate):
    data = None
    plot_path = None
    metrics_path = None
    
    def __init__(self, eName=None, eDescription=None):
        super().__init__(eName, eDescription)
    
    def plot_learning_curve(self, history):
        """Plot the learning curves for training and validation"""
        plt.figure(figsize=(12, 5))
        
        # Plot training & validation loss
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        if 'val_loss' in history and history['val_loss']:
            plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Model Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot training & validation accuracy
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Training Accuracy')
        if 'val_acc' in history and history['val_acc']:
            plt.plot(history['val_acc'], label='Validation Accuracy')
        plt.title('Model Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        
        if self.plot_path:
            os.makedirs(os.path.dirname(self.plot_path), exist_ok=True)
            plt.savefig(self.plot_path)
            print(f"Learning curve saved to {self.plot_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, true_y, pred_y, class_names=None):
        """Plot the confusion matrix for classification results"""
        conf_matrix = confusion_matrix(true_y, pred_y)
        num_classes = conf_matrix.shape[0]
        
        if class_names is None:
            class_names = [str(i) for i in range(num_classes)]
        
        plt.figure(figsize=(10, 8))
        plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(num_classes)
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        
        fmt = 'd'
        thresh = conf_matrix.max() / 2.
        for i in range(num_classes):
            for j in range(num_classes):
                plt.text(j, i, format(conf_matrix[i, j], fmt),
                        ha="center", va="center",
                        color="white" if conf_matrix[i, j] > thresh else "black")
        
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        
        # Save if path is provided
        if self.plot_path:
            conf_matrix_path = self.plot_path.replace('.png', '_confusion_matrix.png')
            plt.savefig(conf_matrix_path)
            print(f"Confusion matrix saved to {conf_matrix_path}")
        
        plt.show()
    
    def evaluate_metrics(self, true_y, pred_y):
        """Calculate and return evaluation metrics"""
        # Convert torch tensors to numpy arrays if needed
        if isinstance(true_y, torch.Tensor):
            true_y = true_y.cpu().numpy()
        if isinstance(pred_y, torch.Tensor):
            pred_y = pred_y.cpu().numpy()
            
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(true_y, pred_y)
        
        # Precision, Recall, F1 (macro averaged)
        metrics['precision'] = precision_score(true_y, pred_y, average='macro', zero_division=0)
        metrics['recall'] = recall_score(true_y, pred_y, average='macro', zero_division=0)
        metrics['f1'] = f1_score(true_y, pred_y, average='macro', zero_division=0)
        
        # Precision, Recall, F1 (weighted averaged)
        metrics['precision_weighted'] = precision_score(true_y, pred_y, average='weighted', zero_division=0)
        metrics['recall_weighted'] = recall_score(true_y, pred_y, average='weighted', zero_division=0)
        metrics['f1_weighted'] = f1_score(true_y, pred_y, average='weighted', zero_division=0)
        
        # Save metrics to file
        if self.metrics_path:
            try:
                os.makedirs(os.path.dirname(self.metrics_path), exist_ok=True)
                
                # Convert numpy values to Python native types for JSON serialization
                clean_metrics = {}
                for key, value in metrics.items():
                    if isinstance(value, np.ndarray):
                        clean_metrics[key] = value.tolist()
                    elif isinstance(value, np.floating):
                        clean_metrics[key] = float(value)
                    elif isinstance(value, np.integer):
                        clean_metrics[key] = int(value)
                    else:
                        clean_metrics[key] = value
                
                with open(self.metrics_path, 'w') as f:
                    json.dump(clean_metrics, f, indent=4)
                print(f"Metrics saved to {self.metrics_path}")
            except Exception as e:
                print(f"Error saving metrics: {str(e)}")
        
        return metrics
    
    def evaluate(self):
        """Main evaluation method"""
        true_y = self.data['true_y']
        pred_y = self.data['pred_y']
        
        # Plot learning curves if history is available
        if 'history' in self.data:
            try:
                self.plot_learning_curve(self.data['history'])
            except Exception as e:
                print(f"Warning: Could not plot learning curves: {str(e)}")
        
        # Calculate metrics
        try:
            metrics = self.evaluate_metrics(true_y, pred_y)
            
            # Print evaluation results
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro-Precision: {metrics['precision']:.4f}")
            print(f"Macro-Recall: {metrics['recall']:.4f}")
            print(f"Macro-F1: {metrics['f1']:.4f}")
            print(f"Weighted-F1: {metrics['f1_weighted']:.4f}")
            
            # Plot confusion matrix
            try:
                self.plot_confusion_matrix(true_y, pred_y)
            except Exception as e:
                print(f"Warning: Could not plot confusion matrix: {str(e)}")
            
            return metrics
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            return {
                'accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'f1_weighted': 0.0
            }
