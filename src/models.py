"""
Federated Learning Models: Generator and Discriminator for GANs
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    Generator model for GAN
    Takes random noise as input and generates fake images
    """
    
    def __init__(self, noise_dim=100, channels=1, image_size=28):
        """
        Args:
            noise_dim (int): Dimension of input noise
            channels (int): Number of image channels (1 for grayscale, 3 for RGB)
            image_size (int): Size of generated images (28 or 32)
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.channels = channels
        self.image_size = image_size
        
        # Calculate the size needed before reshape
        self.fc_size = 256 * 7 * 7  # For 28x28 images: 256 * 7 * 7
        
        self.model = nn.Sequential(
            # Input: noise_dim -> output: 256 * 7 * 7
            nn.Linear(noise_dim, 256 * 7 * 7),
            nn.BatchNorm1d(256 * 7 * 7),
            nn.ReLU(inplace=True),
            
            # Reshape in forward pass (256, 7, 7)
            
            # Transposed Conv layers for upsampling
            # (256, 7, 7) -> (128, 14, 14)
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            # (128, 14, 14) -> (64, 28, 28)
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            # (64, 28, 28) -> (channels, 28, 28)
            nn.Conv2d(64, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.Tanh()  # Output range [-1, 1]
        )
    
    def forward(self, z):
        """
        Args:
            z (torch.Tensor): Noise tensor of shape (batch_size, noise_dim)
        
        Returns:
            torch.Tensor: Generated images of shape (batch_size, channels, image_size, image_size)
        """
        # Linear layer + reshape
        x = self.model[0](z)  # Linear layer
        x = self.model[1](x)  # BatchNorm1d
        x = self.model[2](x)  # ReLU
        
        # Reshape to (batch_size, 256, 7, 7)
        x = x.view(z.size(0), 256, 7, 7)
        
        # Pass through convolutional layers
        x = self.model[3](x)  # ConvTranspose2d
        x = self.model[4](x)  # BatchNorm2d
        x = self.model[5](x)  # ReLU
        
        x = self.model[6](x)  # ConvTranspose2d
        x = self.model[7](x)  # BatchNorm2d
        x = self.model[8](x)  # ReLU
        
        x = self.model[9](x)  # Conv2d
        x = self.model[10](x)  # Tanh
        
        return x


class Discriminator(nn.Module):
    """
    Discriminator model for GAN
    Classifies whether images are real or generated (fake)
    """
    
    def __init__(self, channels=1, image_size=28):
        """
        Args:
            channels (int): Number of image channels (1 for grayscale, 3 for RGB)
            image_size (int): Size of input images (28 or 32)
        """
        super(Discriminator, self).__init__()
        self.channels = channels
        self.image_size = image_size
        
        self.model = nn.Sequential(
            # Input: (channels, 28, 28)
            # (channels, 28, 28) -> (64, 14, 14)
            nn.Conv2d(channels, 64, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (64, 14, 14) -> (128, 7, 7)
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            # (128, 7, 7) -> (256, 3, 3) or similar
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # Adaptive pooling to handle different input sizes
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Fully connected layer for classification
        self.fc = nn.Linear(256, 1)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input image of shape (batch_size, channels, image_size, image_size)
        
        Returns:
            torch.Tensor: Probability that image is real, shape (batch_size, 1)
        """
        x = self.model(x)
        
        # Adaptive pooling to handle variable sizes
        x = self.adaptive_pool(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Classification
        x = self.fc(x)
        x = self.sigmoid(x)
        
        return x


def create_models(config, device):
    """
    Create Generator and Discriminator models
    
    Args:
        config (dict): Configuration dictionary
        device (torch.device): Device to place models on
    
    Returns:
        tuple: (Generator, Discriminator) models
    """
    generator = Generator(
        noise_dim=config['noise_dim'],
        channels=config['channels'],
        image_size=config['image_size']
    ).to(device)
    
    discriminator = Discriminator(
        channels=config['channels'],
        image_size=config['image_size']
    ).to(device)
    
    return generator, discriminator


def count_parameters(model):
    """Count total trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
