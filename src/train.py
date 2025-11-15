"""
Main Training Script for Federated Learning with GANs
"""

import os
import sys
import json
import torch
import argparse
from pathlib import Path

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src import (
    create_models,
    count_parameters,
    FederatedClient,
    FederatedServer,
    get_client_data_loaders,
    MetricsTracker,
    Logger,
    get_device,
    generate_sample_images,
    visualize_generated_images,
    plot_training_metrics
)


def load_config(config_path='./config.json'):
    """Load configuration from JSON file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config


def main(config_path='./config.json'):
    """
    Main training loop for Federated Learning with GANs
    """
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = get_device(config)
    
    # Setup logging
    logger = Logger(log_dir='./logs')
    logger.log("Starting Federated Learning with GANs")
    logger.log_config(config)
    
    # Set random seed for reproducibility
    torch.manual_seed(config['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['seed'])
    
    # Create global models
    logger.log("\n" + "="*80)
    logger.log("Creating Global Models")
    logger.log("="*80)
    
    generator, discriminator = create_models(config, device)
    
    gen_params = count_parameters(generator)
    disc_params = count_parameters(discriminator)
    
    logger.log(f"Generator Parameters: {gen_params:,}")
    logger.log(f"Discriminator Parameters: {disc_params:,}")
    logger.log(f"Total Parameters: {gen_params + disc_params:,}")
    
    # Load datasets for clients
    logger.log("\n" + "="*80)
    logger.log("Loading Client Datasets (System Heterogeneity via Batch Sizes)")
    logger.log("="*80)
    
    client_data_loaders = get_client_data_loaders(config, data_dir='./datasets')
    
    # Create federated clients
    logger.log("\n" + "="*80)
    logger.log("Initializing Federated Clients")
    logger.log("="*80)
    
    clients = []
    for client_id in range(config['num_clients']):
        train_loader, test_loader = client_data_loaders[client_id]
        batch_size = config['batch_sizes'][client_id]
        
        # Create local copies of models for each client
        gen_local = create_models(config, device)[0]
        disc_local = create_models(config, device)[1]
        
        client = FederatedClient(
            client_id=client_id,
            generator=gen_local,
            discriminator=disc_local,
            train_loader=train_loader,
            batch_size=batch_size,
            config=config,
            device=device
        )
        
        clients.append(client)
        logger.log(f"Client {client_id + 1}: Batch Size = {batch_size}, Dataset = {config['datasets'][client_id]}")
    
    # Create federated server
    logger.log("\n" + "="*80)
    logger.log("Initializing Federated Server (FedAvg Aggregation)")
    logger.log("="*80)
    
    server = FederatedServer(generator, discriminator, config)
    
    # Get aggregation statistics
    agg_stats = server.get_aggregation_stats(clients)
    logger.log(f"Total Training Samples: {agg_stats['total_samples']:,}")
    logger.log(f"Average Samples per Client: {agg_stats['average_client_size']:.0f}")
    logger.log(f"Size Heterogeneity Ratio: {agg_stats['size_heterogeneity']:.2f}x")
    
    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(config)
    
    # Federated Learning Loop
    logger.log("\n" + "="*80)
    logger.log("Starting Federated Learning Training")
    logger.log("="*80 + "\n")
    
    for round_num in range(config['num_rounds']):
        logger.log(f">>> COMMUNICATION ROUND {round_num + 1}/{config['num_rounds']}")
        
        # Record communication round
        metrics_tracker.record_communication_round()
        
        # Client-side training
        logger.log(f"  [CLIENT TRAINING]")
        for client in clients:
            logger.log(f"    Client {client.client_id + 1} training with batch size {client.batch_size}...")
            client.train(epochs=config['epochs_per_client'])
            
            metrics = client.get_metrics()
            logger.log(f"      Gen Loss: {metrics['generator_loss']:.4f}, "
                      f"Disc Loss: {metrics['discriminator_loss']:.4f}, "
                      f"Disc Acc: {metrics['discriminator_acc']:.4f}")
            
            # Track metrics
            metrics_tracker.add_metrics(
                round_num=round_num,
                client_id=client.client_id,
                gen_loss=metrics['generator_loss'],
                disc_loss=metrics['discriminator_loss'],
                disc_acc=metrics['discriminator_acc']
            )
        
        # Server-side aggregation
        logger.log(f"  [SERVER AGGREGATION]")
        agg_info = server.aggregate(clients, device)
        logger.log(f"    Aggregated {agg_info['num_clients']} clients")
        logger.log(f"    Total samples aggregated: {agg_info['total_samples']:,}")
        
        # Broadcast aggregated weights to clients
        logger.log(f"  [WEIGHT BROADCAST]")
        server.broadcast_weights(clients, device)
        logger.log(f"    Global model broadcasted to all clients")
        
        # Get round summary
        round_summary = metrics_tracker.get_round_summary(round_num)
        if round_summary:
            logger.log(f"  [ROUND {round_num + 1} SUMMARY]")
            logger.log(f"    Avg Gen Loss: {round_summary['avg_gen_loss']:.4f}")
            logger.log(f"    Avg Disc Loss: {round_summary['avg_disc_loss']:.4f}")
            logger.log(f"    Avg Disc Accuracy: {round_summary['avg_acc']:.4f}")
        
        # Generate and save sample images every few rounds
        if (round_num + 1) % 5 == 0 and config.get('save_generated_images', False):
            logger.log(f"  [GENERATING SAMPLES]")
            samples = generate_sample_images(
                generator,
                num_samples=config.get('num_generated_samples', 16),
                noise_dim=config['noise_dim'],
                device=device
            )
            
            os.makedirs('./results/samples', exist_ok=True)
            sample_path = f"./results/samples/round_{round_num + 1}.png"
            visualize_generated_images(samples, sample_path, config.get('num_generated_samples', 16))
        
        logger.log("")
    
    # Training Complete
    logger.log("="*80)
    logger.log("Federated Learning Training Complete!")
    logger.log("="*80)
    
    # Save metrics
    logger.log("\n[SAVING RESULTS]")
    metrics_path = metrics_tracker.save_metrics('./results')
    logger.log(f"Metrics saved to: {metrics_path}")
    
    # Plot training metrics
    plot_path = plot_training_metrics(metrics_tracker, './results')
    logger.log(f"Training plots saved to: {plot_path}")
    
    # Save final models
    logger.log("\n[SAVING MODELS]")
    os.makedirs('./results', exist_ok=True)
    
    torch.save(generator.state_dict(), './results/final_generator.pth')
    torch.save(discriminator.state_dict(), './results/final_discriminator.pth')
    
    logger.log("Final generator saved to: ./results/final_generator.pth")
    logger.log("Final discriminator saved to: ./results/final_discriminator.pth")
    
    # Print summary statistics
    logger.log("\n" + "="*80)
    logger.log("TRAINING SUMMARY")
    logger.log("="*80)
    logger.log(f"Total Communication Rounds: {metrics_tracker.communication_rounds}")
    logger.log(f"Number of Clients: {config['num_clients']}")
    logger.log(f"Local Epochs per Round: {config['epochs_per_client']}")
    logger.log(f"Batch Sizes (Heterogeneous): {config['batch_sizes']}")
    logger.log(f"Datasets: {config['datasets']}")
    logger.log(f"Results saved in: ./results/")
    logger.log(f"Logs saved in: ./logs/")
    logger.log("="*80)
    
    return metrics_tracker


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Federated Learning with GANs')
    parser.add_argument('--config', type=str, default='./config.json',
                       help='Path to config file (default: ./config.json)')
    
    args = parser.parse_args()
    
    metrics_tracker = main(config_path=args.config)
