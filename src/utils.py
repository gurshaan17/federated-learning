"""
Utility Functions for Federated Learning with GANs
"""

import os
import json
import csv
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path


class MetricsTracker:
    """Track training metrics across rounds and clients"""
    
    def __init__(self, config):
        self.config = config
        self.metrics = {
            'round': [],
            'client_id': [],
            'generator_loss': [],
            'discriminator_loss': [],
            'discriminator_acc': [],
            'communication_round': []
        }
        
        self.communication_rounds = 0
    
    def add_metrics(self, round_num, client_id, gen_loss, disc_loss, disc_acc):
        """Add metrics for a client in a round"""
        self.metrics['round'].append(round_num)
        self.metrics['client_id'].append(client_id)
        self.metrics['generator_loss'].append(gen_loss)
        self.metrics['discriminator_loss'].append(disc_loss)
        self.metrics['discriminator_acc'].append(disc_acc)
        self.metrics['communication_round'].append(self.communication_rounds)
    
    def record_communication_round(self):
        """Record a communication round between server and clients"""
        self.communication_rounds += 1
    
    def save_metrics(self, save_dir='./results'):
        """Save metrics to CSV file"""
        os.makedirs(save_dir, exist_ok=True)
        
        csv_path = os.path.join(save_dir, 'metrics.csv')
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.metrics.keys())
            
            for i in range(len(self.metrics['round'])):
                row = [self.metrics[key][i] for key in self.metrics.keys()]
                writer.writerow(row)
        
        print(f"Metrics saved to {csv_path}")
        return csv_path
    
    def get_round_summary(self, round_num):
        """Get summary statistics for a specific round"""
        round_indices = [i for i, r in enumerate(self.metrics['round']) if r == round_num]
        
        if not round_indices:
            return None
        
        avg_gen_loss = np.mean([self.metrics['generator_loss'][i] for i in round_indices])
        avg_disc_loss = np.mean([self.metrics['discriminator_loss'][i] for i in round_indices])
        avg_acc = np.mean([self.metrics['discriminator_acc'][i] for i in round_indices])
        
        return {
            'avg_gen_loss': avg_gen_loss,
            'avg_disc_loss': avg_disc_loss,
            'avg_acc': avg_acc
        }


class Logger:
    """Log training information to file and console"""
    
    def __init__(self, log_dir='./logs'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')
        
        with open(self.log_file, 'w') as f:
            f.write(f"Training started at {datetime.now()}\n")
            f.write("=" * 80 + "\n\n")
    
    def log(self, message):
        """Log message to both file and console"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(message + '\n')
    
    def log_config(self, config):
        """Log configuration"""
        msg = "Configuration:\n" + json.dumps(config, indent=2)
        self.log(msg)


def create_noise(batch_size, noise_dim, device):
    """Create random noise tensor"""
    return torch.randn(batch_size, noise_dim, device=device)


def calculate_discriminator_accuracy(real_output, fake_output):
    """
    Calculate discriminator accuracy
    Real images should have output close to 1
    Fake images should have output close to 0
    """
    real_correct = (real_output > 0.5).float().mean().item()
    fake_correct = (fake_output <= 0.5).float().mean().item()
    
    accuracy = (real_correct + fake_correct) / 2
    return accuracy


def get_device(config):
    """Get appropriate device (cuda or cpu)"""
    if config.get('device', 'cuda') == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")
    
    return device


def save_checkpoint(generator, discriminator, optimizer_g, optimizer_d, round_num, save_dir='./results'):
    """Save model checkpoint"""
    os.makedirs(save_dir, exist_ok=True)
    
    checkpoint = {
        'round': round_num,
        'generator_state': generator.state_dict(),
        'discriminator_state': discriminator.state_dict(),
        'optimizer_g_state': optimizer_g.state_dict(),
        'optimizer_d_state': optimizer_d.state_dict(),
    }
    
    checkpoint_path = os.path.join(save_dir, f'checkpoint_round_{round_num}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    return checkpoint_path


def load_checkpoint(checkpoint_path, generator, discriminator, optimizer_g, optimizer_d):
    """Load model checkpoint"""
    checkpoint = torch.load(checkpoint_path)
    
    generator.load_state_dict(checkpoint['generator_state'])
    discriminator.load_state_dict(checkpoint['discriminator_state'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g_state'])
    optimizer_d.load_state_dict(checkpoint['optimizer_d_state'])
    
    return checkpoint['round']


def plot_training_metrics(metrics_tracker, save_dir='./results'):
    """Plot training metrics"""
    os.makedirs(os.path.join(save_dir, 'plots'), exist_ok=True)
    
    # Convert to numpy arrays for plotting
    rounds = np.array(metrics_tracker.metrics['round'])
    gen_losses = np.array(metrics_tracker.metrics['generator_loss'])
    disc_losses = np.array(metrics_tracker.metrics['discriminator_loss'])
    disc_accs = np.array(metrics_tracker.metrics['discriminator_acc'])
    
    # Calculate per-round averages
    unique_rounds = np.unique(rounds)
    avg_gen_loss = [np.mean(gen_losses[rounds == r]) for r in unique_rounds]
    avg_disc_loss = [np.mean(disc_losses[rounds == r]) for r in unique_rounds]
    avg_acc = [np.mean(disc_accs[rounds == r]) for r in unique_rounds]
    
    # Set style
    sns.set_style("whitegrid")
    plt.figure(figsize=(15, 12))
    
    # Plot 1: Generator Loss
    plt.subplot(2, 2, 1)
    plt.plot(unique_rounds, avg_gen_loss, marker='o', linewidth=2, markersize=6)
    plt.xlabel('Round')
    plt.ylabel('Generator Loss')
    plt.title('Average Generator Loss across Rounds')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Discriminator Loss
    plt.subplot(2, 2, 2)
    plt.plot(unique_rounds, avg_disc_loss, marker='s', linewidth=2, markersize=6, color='orange')
    plt.xlabel('Round')
    plt.ylabel('Discriminator Loss')
    plt.title('Average Discriminator Loss across Rounds')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Discriminator Accuracy
    plt.subplot(2, 2, 3)
    plt.plot(unique_rounds, avg_acc, marker='^', linewidth=2, markersize=6, color='green')
    plt.xlabel('Round')
    plt.ylabel('Accuracy')
    plt.title('Average Discriminator Accuracy across Rounds')
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1])
    
    # Plot 4: All metrics together (normalized)
    plt.subplot(2, 2, 4)
    plt.plot(unique_rounds, avg_gen_loss / max(avg_gen_loss) if max(avg_gen_loss) > 0 else avg_gen_loss, 
             label='Gen Loss (norm)', marker='o', linewidth=2)
    plt.plot(unique_rounds, avg_disc_loss / max(avg_disc_loss) if max(avg_disc_loss) > 0 else avg_disc_loss, 
             label='Disc Loss (norm)', marker='s', linewidth=2)
    plt.plot(unique_rounds, avg_acc, label='Disc Accuracy', marker='^', linewidth=2)
    plt.xlabel('Round')
    plt.ylabel('Normalized Value / Accuracy')
    plt.title('All Metrics Overview')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, 'plots', 'training_metrics.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plot saved to {plot_path}")
    return plot_path


def generate_sample_images(generator, num_samples=16, noise_dim=100, device='cuda'):
    """Generate sample images from generator"""
    generator.eval()
    
    with torch.no_grad():
        noise = torch.randn(num_samples, noise_dim, device=device)
        images = generator(noise)
    
    return images


def visualize_generated_images(images, save_path, num_samples=16):
    """Visualize and save generated images"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Normalize images from [-1, 1] to [0, 1]
    images = (images + 1) / 2
    images = images.clamp(0, 1)
    
    # Create grid
    grid_size = int(np.sqrt(num_samples))
    fig, axes = plt.subplots(grid_size, grid_size, figsize=(8, 8))
    
    for idx, ax in enumerate(axes.flat):
        if idx < len(images):
            img = images[idx].cpu().squeeze().numpy()
            ax.imshow(img, cmap='gray')
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.close()
    
    print(f"Generated images saved to {save_path}")
