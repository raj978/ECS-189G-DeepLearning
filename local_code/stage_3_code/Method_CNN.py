"""
Concrete CNN Model for the specific classification task
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import time
import json
import os
from local_code.base_class.method import method


class Method_CNN(method, nn.Module):
    data = None
    max_epoch = 100
    learning_rate = 1e-3
    batch_size = 128
    model_path = None
    hist_path = None
    
    # Model configuration parameters
    conv_layers = 2
    kernel_size = 3
    stride = 1
    padding = 1
    pool_size = 2
    fc_layers = 2
    fc_units = [256, 128]
    dropout_rate = 0.25
    
    # Define in_channels based on dataset
    in_channels = 1  # Default for MNIST, ORL (grayscale)
    # Set for CIFAR-10 (3-channel RGB)
    
    # Number of classes based on dataset
    num_classes = 10  # Default for MNIST, CIFAR
    # Set for ORL (40 subjects)
    
    # Input dimensions
    input_height = 28  # Default for MNIST
    input_width = 28   # Default for MNIST
    # Set for ORL (64x64), CIFAR (32x32)
    
    def __init__(self, mName, mDescription):
        method.__init__(self, mName, mDescription)
        nn.Module.__init__(self)
        
        self.name = mName  # Initialize the name attribute
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Will be defined in build_model
        self.conv_net = None
        self.fc_net = None
        
        # Optimizer configuration (default: Adam)
        self.optimizer_name = "adam"
        
        # Loss function configuration (default: CrossEntropy)
        self.loss_fn_name = "cross_entropy"
    
    def build_model(self):
        """Build the CNN model architecture based on configuration parameters"""
        # Calculate output dimensions after convolution and pooling layers
        feature_width = self.input_width
        feature_height = self.input_height
        
        # Build convolutional layers
        conv_layers = []
        
        # First conv layer
        conv_layers.append(nn.Conv2d(
            in_channels=self.in_channels, 
            out_channels=16, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            padding=self.padding
        ))
        conv_layers.append(nn.BatchNorm2d(16))
        conv_layers.append(nn.ReLU())
        conv_layers.append(nn.MaxPool2d(kernel_size=self.pool_size))
        conv_layers.append(nn.Dropout(self.dropout_rate))
        
        # Update feature dimensions after first conv layer
        feature_width = (feature_width + 2*self.padding - self.kernel_size) // self.stride + 1
        feature_width = feature_width // self.pool_size
        feature_height = (feature_height + 2*self.padding - self.kernel_size) // self.stride + 1
        feature_height = feature_height // self.pool_size
        
        # Additional conv layers
        in_channels = 16
        for i in range(1, self.conv_layers):
            out_channels = in_channels * 2
            
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride=self.stride, 
                padding=self.padding
            ))
            conv_layers.append(nn.BatchNorm2d(out_channels))
            conv_layers.append(nn.ReLU())
            conv_layers.append(nn.MaxPool2d(kernel_size=self.pool_size))
            conv_layers.append(nn.Dropout(self.dropout_rate))
            
            # Update feature dimensions after each conv layer
            feature_width = (feature_width + 2*self.padding - self.kernel_size) // self.stride + 1
            feature_width = feature_width // self.pool_size
            feature_height = (feature_height + 2*self.padding - self.kernel_size) // self.stride + 1
            feature_height = feature_height // self.pool_size
            
            in_channels = out_channels
        
        self.conv_net = nn.Sequential(*conv_layers)
        
        # Calculate the flattened feature size
        flattened_size = in_channels * feature_width * feature_height
        print(f"Flattened feature size: {flattened_size} = {in_channels} × {feature_width} × {feature_height}")
        
        # Build fully connected layers
        fc_layers = []
        
        # First FC layer
        fc_layers.append(nn.Linear(flattened_size, self.fc_units[0]))
        fc_layers.append(nn.ReLU())
        fc_layers.append(nn.Dropout(self.dropout_rate))
        
        # Additional FC layers
        for i in range(1, min(self.fc_layers, len(self.fc_units))):
            fc_layers.append(nn.Linear(self.fc_units[i-1], self.fc_units[i]))
            fc_layers.append(nn.ReLU())
            fc_layers.append(nn.Dropout(self.dropout_rate))
        
        # Output layer
        fc_layers.append(nn.Linear(self.fc_units[min(self.fc_layers, len(self.fc_units))-1], self.num_classes))
        
        self.fc_net = nn.Sequential(*fc_layers)
        
    def forward(self, x):
        """Forward pass through the network"""
        # Convolutional layers
        x = self.conv_net(x)
        
        # Flatten
        x = torch.flatten(x, 1)
        
        # Fully connected layers
        x = self.fc_net(x)
        
        return x
    
    def train_model(self, train_loader, val_loader=None):
        """Train the CNN model using the provided data loader"""
        # Initialize loss function
        if self.loss_fn_name == "cross_entropy":
            criterion = nn.CrossEntropyLoss()
        elif self.loss_fn_name == "mse":
            criterion = nn.MSELoss()
        else:
            criterion = nn.CrossEntropyLoss()  # default
        
        # Initialize optimizer
        if self.optimizer_name == "adam":
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        elif self.optimizer_name == "sgd":
            optimizer = optim.SGD(self.parameters(), lr=self.learning_rate, momentum=0.9)
        elif self.optimizer_name == "rmsprop":
            optimizer = optim.RMSprop(self.parameters(), lr=self.learning_rate)
        else:
            optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)  # default
        
        # Learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Training history
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        best_val_acc = 0.0
        
        # Training loop
        for epoch in range(self.max_epoch):
            start_time = time.time()
            
            # Train mode
            self.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                train_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            # Calculate average training loss and accuracy
            train_loss = train_loss / train_total
            train_acc = train_correct / train_total
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                self.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for inputs, labels in val_loader:
                        inputs, labels = inputs.to(self.device), labels.to(self.device)
                        
                        outputs = self(inputs)
                        loss = criterion(outputs, labels)
                        
                        val_loss += loss.item() * inputs.size(0)
                        _, predicted = torch.max(outputs, 1)
                        val_total += labels.size(0)
                        val_correct += (predicted == labels).sum().item()
                
                val_loss = val_loss / val_total
                val_acc = val_correct / val_total
                
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                
                # Update learning rate
                scheduler.step(val_loss)
                
                # Save best model
                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    if self.model_path:
                        torch.save(self.state_dict(), self.model_path)
                
                print(f'Epoch {epoch+1}/{self.max_epoch}, '
                      f'Time: {time.time() - start_time:.2f}s, '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                      f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            else:
                print(f'Epoch {epoch+1}/{self.max_epoch}, '
                      f'Time: {time.time() - start_time:.2f}s, '
                      f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}')
        
        # Save training history
        if self.hist_path:
            os.makedirs(os.path.dirname(self.hist_path), exist_ok=True)
            with open(self.hist_path, 'w') as f:
                json.dump(history, f)
        
        return history
    
    def test_model(self, test_loader):
        """Test the CNN model on the test dataset"""
        self.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self(inputs)
                _, predicted = torch.max(outputs, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        test_acc = correct / total
        print(f'Test Accuracy: {test_acc:.4f}')
        
        return test_acc, all_preds, all_labels
    
    def learning(self):
        """Train the model"""
        # Move model to device
        self.to(self.device)
        
        # Get data loaders
        train_X = self.data['train']['X'].to(self.device)
        train_y = self.data['train']['y'].to(self.device)
        
        # Create dataset and data loader
        train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        
        # Build model architecture
        self.build_model()
        
        # Train the model
        history = self.train_model(train_loader)
        
        return history
    
    def testing(self):
        """Test the model and return predictions"""
        # Move model to device
        self.to(self.device)
        
        # Get test data
        test_X = self.data['test']['X'].to(self.device)
        
        # Make predictions
        self.eval()
        with torch.no_grad():
            outputs = self(test_X)
            _, predicted = torch.max(outputs, 1)
        
        return predicted.cpu()
    
    def run(self):
        """Run the model training and testing pipeline"""
        print('Method running...')
        print('--start training...')
        self.learning()
        print('--start testing...')
        pred_y = self.testing()
        
        return {'pred_y': pred_y, 'true_y': self.data['test']['y']}
