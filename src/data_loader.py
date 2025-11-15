"""
Data Loading and Preprocessing
Handles MNIST, Fashion-MNIST, and CIFAR-10 datasets
"""

import torch
from torchvision import datasets, transforms
import os


def get_data_loader(dataset_name, batch_size, train=True, data_dir='./datasets'):
    """
    Get data loader for specified dataset
    
    Args:
        dataset_name (str): Name of dataset ('mnist', 'fashion_mnist', 'cifar10_subset')
        batch_size (int): Batch size for the loader
        train (bool): Whether to load training or test data
        data_dir (str): Directory to save/load datasets
    
    Returns:
        torch.utils.data.DataLoader: Data loader for the dataset
    """
    
    # Create data directory if it doesn't exist
    os.makedirs(data_dir, exist_ok=True)
    
    if dataset_name.lower() == 'mnist':
        # Normalize to [-1, 1] for GAN training
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))  # Normalize to [-1, 1]
        ])
        
        dataset = datasets.MNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=True
        )
    
    elif dataset_name.lower() == 'fashion_mnist':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = datasets.FashionMNIST(
            root=data_dir,
            train=train,
            transform=transform,
            download=True
        )
    
    elif dataset_name.lower() == 'cifar10_subset':
        # CIFAR-10 is already 32x32, but we need to handle 3 channels
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale for consistency
            transforms.Resize(28),  # Resize to 28x28 for consistency
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        
        dataset = datasets.CIFAR10(
            root=data_dir,
            train=train,
            transform=transform,
            download=True
        )
        
        # Use only subset for faster training
        if train:
            indices = torch.randperm(len(dataset))[:10000]
            dataset = torch.utils.data.Subset(dataset, indices)
        else:
            indices = torch.randperm(len(dataset))[:2000]
            dataset = torch.utils.data.Subset(dataset, indices)
    
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=0,  # Set to 0 for compatibility
        drop_last=train
    )
    
    return data_loader


def get_client_data_loaders(config, data_dir='./datasets'):
    """
    Get data loaders for all 3 clients with different datasets and batch sizes
    
    Args:
        config (dict): Configuration dictionary containing:
            - datasets: List of dataset names
            - batch_sizes: List of batch sizes for each client
        data_dir (str): Directory for datasets
    
    Returns:
        list: List of (train_loader, test_loader) tuples for each client
    """
    client_data_loaders = []
    
    for client_id in range(config['num_clients']):
        dataset_name = config['datasets'][client_id]
        batch_size = config['batch_sizes'][client_id]
        
        train_loader = get_data_loader(
            dataset_name,
            batch_size=batch_size,
            train=True,
            data_dir=data_dir
        )
        
        test_loader = get_data_loader(
            dataset_name,
            batch_size=batch_size,
            train=False,
            data_dir=data_dir
        )
        
        client_data_loaders.append((train_loader, test_loader))
        
        print(f"Client {client_id + 1}: {dataset_name} - Batch Size: {batch_size}")
        print(f"  Training samples: {len(train_loader.dataset)}")
        print(f"  Testing samples: {len(test_loader.dataset)}")
    
    return client_data_loaders
