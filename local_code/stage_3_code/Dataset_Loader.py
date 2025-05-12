"""
Concrete IO class for a specific dataset
"""

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
from local_code.base_class.dataset import dataset


class CustomImageDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label


class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_name = None
    
    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)
        
    def load_mnist(self):
        """Load the MNIST dataset from pickle file"""
        print("Loading MNIST dataset...")
        mnist_file = os.path.join(self.dataset_source_folder_path, 'MNIST')
        
        # Check if file exists
        if not os.path.isfile(mnist_file):
            raise FileNotFoundError(f"MNIST file not found at {mnist_file}")
            
        try:
            with open(mnist_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading MNIST dataset: {str(e)}")
        
        # Extract images and labels from the loaded data
        train_X = []
        train_y = []
        for instance in data['train']:
            train_X.append(instance['image'])
            train_y.append(instance['label'])
        
        test_X = []
        test_y = []
        for instance in data['test']:
            test_X.append(instance['image'])
            test_y.append(instance['label'])
        
        # Convert to numpy arrays
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        
        # Convert to torch tensors and reshape if needed
        # MNIST is grayscale with shape (N, 28, 28)
        train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(1) / 255.0  # Add channel dimension
        test_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(1) / 255.0
        train_y = torch.tensor(train_y, dtype=torch.long)
        test_y = torch.tensor(test_y, dtype=torch.long)
        
        print(f"MNIST dataset loaded: {train_X.shape[0]} training samples, {test_X.shape[0]} test samples")
        
        return {'train': {'X': train_X, 'y': train_y}, 'test': {'X': test_X, 'y': test_y}}
    
    def load_orl(self):
        """Load the ORL dataset from pickle file"""
        print("Loading ORL dataset...")
        orl_file = os.path.join(self.dataset_source_folder_path, 'ORL')
        
        # Check if file exists
        if not os.path.isfile(orl_file):
            raise FileNotFoundError(f"ORL file not found at {orl_file}")
            
        try:
            with open(orl_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading ORL dataset: {str(e)}")
        
        # Extract images and labels from the loaded data
        train_X = []
        train_y = []
        for instance in data['train']:
            # ORL is grayscale but stored as RGB with identical channels
            # According to ReadMe, we'll use the first channel (R) as they're identical
            img = instance['image'][:,:,0]  # Take only the R channel as per ReadMe
            train_X.append(img)
            # ORL labels are 1-indexed, convert to 0-indexed for PyTorch
            train_y.append(instance['label'] - 1) 
        
        test_X = []
        test_y = []
        for instance in data['test']:
            img = instance['image'][:,:,0]  # Take only the R channel
            test_X.append(img)
            test_y.append(instance['label'] - 1)
        
        # Convert to numpy arrays
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        
        # Print shapes to verify dimensions
        print(f"ORL train_X shape: {train_X.shape}")  # Should be (360, 112, 92)
        print(f"ORL test_X shape: {test_X.shape}")    # Should be (40, 112, 92)
        
        # Convert to torch tensors - add channel dimension
        train_X = torch.tensor(train_X, dtype=torch.float32).unsqueeze(1) / 255.0
        test_X = torch.tensor(test_X, dtype=torch.float32).unsqueeze(1) / 255.0
        train_y = torch.tensor(train_y, dtype=torch.long)
        test_y = torch.tensor(test_y, dtype=torch.long)
        
        print(f"ORL dataset loaded: {train_X.shape[0]} training samples, {test_X.shape[0]} test samples")
        print(f"ORL data tensor shapes - Train: {train_X.shape}, Test: {test_X.shape}")
        
        return {'train': {'X': train_X, 'y': train_y}, 'test': {'X': test_X, 'y': test_y}}
    
    def load_cifar(self):
        """Load the CIFAR dataset"""
        print("Loading CIFAR dataset...")
        cifar_file = os.path.join(self.dataset_source_folder_path, 'CIFAR')
        
        # Check if file exists
        if not os.path.isfile(cifar_file):
            raise FileNotFoundError(f"CIFAR file not found at {cifar_file}")
            
        try:
            with open(cifar_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise Exception(f"Error loading CIFAR dataset: {str(e)}")
        
        # Extract images and labels from the loaded data
        train_X = []
        train_y = []
        for instance in data['train']:
            train_X.append(instance['image'])
            train_y.append(instance['label'])
        
        test_X = []
        test_y = []
        for instance in data['test']:
            test_X.append(instance['image'])
            test_y.append(instance['label'])
        
        # Convert to numpy arrays
        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)
        
        # Print shapes to verify dimensions
        print(f"CIFAR train_X shape: {train_X.shape}")  # Should be (50000, 32, 32, 3)
        print(f"CIFAR test_X shape: {test_X.shape}")    # Should be (10000, 32, 32, 3)
        
        # Convert to torch tensors - rearrange channels for PyTorch format (N,C,H,W)
        train_X = np.transpose(train_X, (0, 3, 1, 2))
        test_X = np.transpose(test_X, (0, 3, 1, 2))
        
        # Convert to float and normalize to [0, 1]
        train_X = train_X.astype(np.float32) / 255.0
        test_X = test_X.astype(np.float32) / 255.0
        
        # Apply normalization for CIFAR
        if hasattr(self, 'use_augmentation') and self.use_augmentation:
            print("Applying normalization to CIFAR data...")
            
            # CIFAR-10 mean and std for each channel
            mean = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32).reshape(1, 3, 1, 1)
            std = np.array([0.2023, 0.1994, 0.2010], dtype=np.float32).reshape(1, 3, 1, 1)
            
            # Normalize training data
            train_X = (train_X - mean) / std
            test_X = (test_X - mean) / std
            
            print("Data normalization complete.")
            
        # Convert to PyTorch tensors
        train_X = torch.from_numpy(train_X)
        test_X = torch.from_numpy(test_X)
        train_y = torch.tensor(train_y, dtype=torch.long)
        test_y = torch.tensor(test_y, dtype=torch.long)
        
        print(f"CIFAR dataset loaded: {train_X.shape[0]} training samples, {test_X.shape[0]} test samples")
        
        return {'train': {'X': train_X, 'y': train_y}, 'test': {'X': test_X, 'y': test_y}}
    
    def load(self):
        if self.dataset_name == "MNIST":
            return self.load_mnist()
        elif self.dataset_name == "ORL":
            return self.load_orl()
        elif self.dataset_name == "CIFAR":
            return self.load_cifar()
        else:
            raise ValueError(f"Dataset {self.dataset_name} not supported")
