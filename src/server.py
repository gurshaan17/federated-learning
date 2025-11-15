"""
Federated Server Implementation
Aggregates model weights from all clients using FedAvg algorithm
"""

import torch
import copy


class FederatedServer:
    """
    Federated Learning Server
    Coordinates training and aggregates model weights from clients
    """
    
    def __init__(self, generator, discriminator, config):
        """
        Args:
            generator (nn.Module): Global generator model
            discriminator (nn.Module): Global discriminator model
            config (dict): Configuration dictionary
        """
        self.generator = generator
        self.discriminator = discriminator
        self.config = config
        
        # Store initial global weights
        self.global_gen_weights = self._get_weights(generator)
        self.global_disc_weights = self._get_weights(discriminator)
    
    def _get_weights(self, model):
        """Extract flattened weights from a model"""
        return torch.cat([p.data.cpu().flatten() for p in model.parameters()])
    
    def _set_weights(self, model, weights, device):
        """Set model weights from flattened tensor"""
        offset = 0
        for param in model.parameters():
            size = param.data.numel()
            param.data = weights[offset:offset + size].view_as(param.data).to(device)
            offset += size
    
    def aggregate(self, clients, device):
        """
        Aggregate model weights using FedAvg algorithm
        
        FedAvg: w_t+1 = Σ (n_k / N) * w_t,k
        where n_k is the number of samples for client k
        and N is the total number of samples across all clients
        
        Args:
            clients (list): List of FederatedClient objects
            device (torch.device): Device to use
        """
        
        # Get weights from all clients
        client_gen_weights = []
        client_disc_weights = []
        total_samples = 0
        
        for client in clients:
            gen_w, disc_w = client.get_model_weights()
            client_gen_weights.append(gen_w)
            client_disc_weights.append(disc_w)
            
            # Count samples in client's dataset
            total_samples += len(client.train_loader.dataset)
        
        # Calculate weighted average (FedAvg)
        aggregated_gen_weights = torch.zeros_like(client_gen_weights[0])
        aggregated_disc_weights = torch.zeros_like(client_disc_weights[0])
        
        for i, client in enumerate(clients):
            # Weight for this client = number of samples / total samples
            client_samples = len(client.train_loader.dataset)
            weight = client_samples / total_samples
            
            aggregated_gen_weights += weight * client_gen_weights[i]
            aggregated_disc_weights += weight * client_disc_weights[i]
        
        # Update global model weights
        self.global_gen_weights = aggregated_gen_weights
        self.global_disc_weights = aggregated_disc_weights
        
        self._set_weights(self.generator, aggregated_gen_weights, device)
        self._set_weights(self.discriminator, aggregated_disc_weights, device)
        
        return {
            'total_samples': total_samples,
            'num_clients': len(clients),
            'aggregation_method': 'FedAvg'
        }
    
    def broadcast_weights(self, clients, device):
        """
        Broadcast aggregated weights to all clients
        
        Args:
            clients (list): List of FederatedClient objects
            device (torch.device): Device to use
        """
        for client in clients:
            client.set_model_weights(self.global_gen_weights, self.global_disc_weights)
    
    def get_global_weights(self):
        """Get current global model weights"""
        return self.global_gen_weights, self.global_disc_weights
    
    def get_aggregation_stats(self, clients):
        """Get statistics about the aggregation"""
        client_sizes = [len(client.train_loader.dataset) for client in clients]
        total_samples = sum(client_sizes)
        
        stats = {
            'client_sizes': client_sizes,
            'total_samples': total_samples,
            'average_client_size': total_samples / len(clients),
            'size_heterogeneity': max(client_sizes) / min(client_sizes)
        }
        
        return stats
