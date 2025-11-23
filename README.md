# Federated Learning with GANs - Complete Implementation

A comprehensive implementation of **Federated Learning with Generative Adversarial Networks (GANs)** that addresses system heterogeneity through adaptive batch sizing and distributed training across multiple clients.

## Features

- **Federated GAN Training**: Train GAN models across 3 distributed clients without sharing raw data
- **System Heterogeneity Handling**: Adaptive batch sizes for different computational capabilities
- **Three Lightweight Datasets**: MNIST, Fashion-MNIST, and CIFAR-10 (subsampled)
- **FedAvg Aggregation**: Federated Averaging algorithm for secure model aggregation
- **Comprehensive Tracking**: Monitor accuracy, loss, and communication rounds
- **Visualization**: Generate plots for training metrics and GAN performance

## Project Structure

```
federated-learning-gan/
├── src/
│   ├── __init__.py
│   ├── models.py              # Generator and Discriminator architectures
│   ├── client.py              # Client-side federated learning logic
│   ├── server.py              # Server-side aggregation logic
│   ├── data_loader.py         # Dataset loading and preprocessing
│   ├── utils.py               # Utility functions
│   └── train.py               # Main training script
├── config.json                # Configuration file
├── README.md                  # This file
├── requirements.txt           # Dependencies
├── .gitignore
├── datasets/                  # Downloaded datasets
├── results/                   # Training results and plots
└── logs/                      # Training logs
```

## Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd federated-learning-gan
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Configuration

Edit `config.json` to customize training parameters:

```json
{
  "num_clients": 3,
  "num_rounds": 10,
  "epochs_per_client": 5,
  "learning_rate": 0.0002,
  "beta1": 0.5,
  "noise_dim": 100,
  "device": "cuda",
  "datasets": ["mnist", "fashion_mnist", "cifar10"],
  "batch_sizes": [32, 64, 128],
  "seed": 42
}
```

## Usage

### Quick Start

```bash
python src/train.py
```

### Training Parameters

The main training script will:
1. Load 3 different lightweight datasets
2. Initialize 3 federated clients with different batch sizes
3. Create Generator and Discriminator models
4. Run federated learning for specified rounds
5. Aggregate models using FedAvg
6. Track accuracy and communication metrics
7. Generate visualization plots

### Output

After training completes, you'll find:
- `results/metrics.csv` - Training metrics (accuracy, loss, rounds)
- `results/plots/` - Visualization plots
- `logs/training.log` - Detailed training logs

## Datasets

The implementation uses three lightweight datasets for clients:

| Client | Dataset | Classes | Size |
|--------|---------|---------|------|
| Client 1 | MNIST | 10 | ~70MB |
| Client 2 | Fashion-MNIST | 10 | ~32MB |
| Client 3 | CIFAR-10 (subsampled) | 10 | ~30MB |

## System Heterogeneity Handling

The project simulates different computational capabilities through:

- **Batch Sizes**: Client 1 uses 32, Client 2 uses 64, Client 3 uses 128
- **Adaptive Training**: Each client trains based on its batch size capacity
- **Graceful Handling**: Server aggregates updates without requiring synchronized clients

## Federated Learning Algorithm

### FedAvg Aggregation

```
for each round t:
    1. Server sends global model to all clients
    2. Each client trains on local data with its batch size
    3. Clients upload trained weights
    4. Server aggregates: w_global = Σ(client_weight * local_weights)
    5. Server broadcasts updated global model
```

## GAN Architecture

### Generator
- Input: Random noise (100-dim)
- Hidden layers: 256 → 512 → 1024
- Output: Image (28×28 or 32×32)

### Discriminator
- Input: Image (28×28 or 32×32)
- Hidden layers: 1024 → 512 → 256
- Output: Binary classification (real/fake)

## Monitoring Training

The training process logs:
- **Communication Rounds**: Number of federated learning rounds completed
- **Generator Loss**: Loss from each client's generator
- **Discriminator Loss**: Loss from each client's discriminator
- **Average Accuracy**: Discriminator accuracy across clients
- **Client Batch Sizes**: Tracks heterogeneous batch sizes

## Key Results to Expect

After training for 10 rounds on 3 clients:
- Discriminator Accuracy: ~70-80%
- Stable GAN training without mode collapse
- Communication-efficient model aggregation
- Successful handling of heterogeneous batch sizes

## Troubleshooting

### Out of Memory (OOM)
- Reduce `noise_dim` in config.json
- Decrease batch sizes in the config
- Use CPU instead of CUDA

### Slow Training
- Reduce `num_rounds` for quicker testing
- Decrease `epochs_per_client` for faster iterations
- Use a subset of data for prototyping

## Future Enhancements

- [ ] Non-IID data distribution across clients
- [ ] Differential privacy integration
- [ ] Communication compression
- [ ] Multi-GPU support
- [ ] Visualization of generated images during training

## License

MIT License - Feel free to use for educational purposes

## Support

For issues or questions, please create an issue in the GitHub repository.
