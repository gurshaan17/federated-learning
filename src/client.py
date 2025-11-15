"""
Federated Client Implementation
Each client trains generator and discriminator locally
"""

import torch
import torch.nn as nn
import torch.optim as optim
from .utils import create_noise, calculate_discriminator_accuracy


class FederatedClient:
    """
    Federated Learning Client
    Trains GAN locally on private data without sharing raw data
    """
    
    def __init__(self, client_id, generator, discriminator, train_loader, 
                 batch_size, config, device):
        """
        Args:
            client_id (int): Unique identifier for this client
            generator (nn.Module): Generator model
            discriminator (nn.Module): Discriminator model
            train_loader (DataLoader): Training data loader
            batch_size (int): Batch size for this client
            config (dict): Configuration dictionary
            device (torch.device): Device to use (cuda/cpu)
        """
        self.client_id = client_id
        self.generator = generator
        self.discriminator = discriminator
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.config = config
        self.device = device
        
        # Optimizers
        self.optimizer_g = optim.Adam(
            self.generator.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], 0.999)
        )
        
        self.optimizer_d = optim.Adam(
            self.discriminator.parameters(),
            lr=config['learning_rate'],
            betas=(config['beta1'], 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        # Metrics
        self.local_metrics = {
            'generator_loss': 0.0,
            'discriminator_loss': 0.0,
            'discriminator_acc': 0.0
        }
    
    def train_epoch(self):
        """
        Train for one epoch on local data
        Returns accumulated metrics
        """
        total_gen_loss = 0.0
        total_disc_loss = 0.0
        total_acc = 0.0
        num_batches = 0
        
        for batch_idx, (real_data, _) in enumerate(self.train_loader):
            real_data = real_data.to(self.device)
            current_batch_size = real_data.size(0)
            
            # Labels for real and fake data
            real_labels = torch.ones(current_batch_size, 1, device=self.device)
            fake_labels = torch.zeros(current_batch_size, 1, device=self.device)
            
            # Train Discriminator
            self.optimizer_d.zero_grad()
            
            # Real data
            real_output = self.discriminator(real_data)
            d_loss_real = self.criterion(real_output, real_labels)
            
            # Fake data
            noise = create_noise(current_batch_size, self.config['noise_dim'], self.device)
            fake_data = self.generator(noise).detach()
            fake_output = self.discriminator(fake_data)
            d_loss_fake = self.criterion(fake_output, fake_labels)
            
            # Total discriminator loss
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()
            self.optimizer_d.step()
            
            # Train Generator
            self.optimizer_g.zero_grad()
            
            noise = create_noise(current_batch_size, self.config['noise_dim'], self.device)
            fake_data = self.generator(noise)
            fake_output = self.discriminator(fake_data)
            
            # Generator wants discriminator to think fake is real
            g_loss = self.criterion(fake_output, real_labels)
            g_loss.backward()
            self.optimizer_g.step()
            
            # Calculate accuracy
            with torch.no_grad():
                real_output = self.discriminator(real_data)
                fake_output = self.discriminator(fake_data)
                acc = calculate_discriminator_accuracy(real_output, fake_output)
            
            # Accumulate metrics
            total_gen_loss += g_loss.item()
            total_disc_loss += d_loss.item()
            total_acc += acc
            num_batches += 1
        
        # Average metrics
        self.local_metrics['generator_loss'] = total_gen_loss / num_batches
        self.local_metrics['discriminator_loss'] = total_disc_loss / num_batches
        self.local_metrics['discriminator_acc'] = total_acc / num_batches
        
        return self.local_metrics
    
    def train(self, epochs):
        """
        Train for multiple epochs
        Args:
            epochs (int): Number of epochs to train
        """
        for epoch in range(epochs):
            self.train_epoch()
    
    def get_model_weights(self):
        """Get current model weights (flattened)"""
        gen_weights = torch.cat([p.data.cpu().flatten() 
                                for p in self.generator.parameters()])
        disc_weights = torch.cat([p.data.cpu().flatten() 
                                 for p in self.discriminator.parameters()])
        return gen_weights, disc_weights
    
    def set_model_weights(self, gen_weights, disc_weights):
        """
        Set model weights from federated aggregation
        Args:
            gen_weights: Generator weights
            disc_weights: Discriminator weights
        """
        offset = 0
        
        # Set generator weights
        for param in self.generator.parameters():
            size = param.data.numel()
            param.data = gen_weights[offset:offset + size].view_as(param.data).to(self.device)
            offset += size
        
        offset = 0
        
        # Set discriminator weights
        for param in self.discriminator.parameters():
            size = param.data.numel()
            param.data = disc_weights[offset:offset + size].view_as(param.data).to(self.device)
            offset += size
    
    def get_metrics(self):
        """Get latest local metrics"""
        return self.local_metrics
