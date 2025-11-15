# Federated Learning with GANs - Complete Setup Guide

## Project Overview

This is a **production-ready federated learning system** that trains **Generative Adversarial Networks (GANs)** across 3 distributed clients while addressing:
- **System Heterogeneity**: Different batch sizes (32, 64, 128) simulating different computational capabilities
- **Adversarial Learning**: GANs for generating images
- **Communication Efficiency**: FedAvg aggregation algorithm
- **Privacy Preservation**: No raw data sharing, only model parameters

## Quick Start (3 steps)

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Training
```bash
python quickstart.py
```

Or directly:
```bash
python src/train.py
```

### Step 3: View Results
```bash
python evaluate.py
```

Results are saved in `./results/` directory.

---

## Detailed Setup Instructions

### Prerequisites
- Python 3.7+
- pip or conda
- 4GB+ RAM (or GPU with CUDA 11.0+)

### Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd federated-learning-gan
   ```

2. **Create virtual environment (recommended)**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   python -c "import torch; print(f'PyTorch {torch.__version__}')"
   ```

---

## Project Structure Explained

```
federated-learning-gan/
├── src/
│   ├── __init__.py              # Package initialization
│   ├── models.py                # Generator & Discriminator architectures
│   ├── client.py                # FederatedClient class
│   ├── server.py                # FederatedServer class (FedAvg aggregation)
│   ├── data_loader.py           # Dataset loading for 3 clients
│   ├── utils.py                 # Utilities & metrics tracking
│   └── train.py                 # Main training script
├── config.json                  # Training configuration
├── requirements.txt             # Python dependencies
├── README.md                    # Project README
├── quickstart.py               # Quick start helper
├── evaluate.py                 # Results evaluation
├── datasets/                    # Downloaded datasets (auto-created)
├── results/                     # Training results (auto-created)
│   ├── metrics.csv             # Training metrics
│   ├── plots/                  # Training plots
│   ├── samples/                # Generated images per round
│   ├── final_generator.pth     # Final generator weights
│   └── final_discriminator.pth # Final discriminator weights
└── logs/                        # Training logs (auto-created)
```

---

## Configuration Guide

Edit `config.json` to customize training:

```json
{
  "num_clients": 3,                          // Number of federated clients
  "num_rounds": 10,                          // Communication rounds
  "epochs_per_client": 5,                    // Local epochs per round
  "learning_rate": 0.0002,                   // Adam learning rate
  "beta1": 0.5,                              // Adam beta1
  "noise_dim": 100,                          // Generator input noise dimension
  "device": "cuda",                          // Device: "cuda" or "cpu"
  "image_size": 28,                          // Generated image size
  "channels": 1,                             // Image channels (1=grayscale)
  "datasets": ["mnist", "fashion_mnist", "cifar10_subset"],  // Client datasets
  "batch_sizes": [32, 64, 128],              // System heterogeneity
  "seed": 42,                                // Random seed
  "log_interval": 100,                       // Logging interval
  "save_generated_images": true,             // Save sample images
  "num_generated_samples": 16                // Number of samples to generate
}
```

### Key Configuration Parameters

**num_rounds**: Higher values = better convergence but longer training
- Default: 10 (for quick testing)
- Recommended: 20-50 (for good results)

**epochs_per_client**: Local training epochs per communication round
- Default: 5
- Higher values = more local computation, fewer communication rounds

**batch_sizes**: System heterogeneity simulation
- Client 1: batch_size=32 (slower device)
- Client 2: batch_size=64 (medium device)
- Client 3: batch_size=128 (faster device)

---

## Running Training

### Basic Training
```bash
python src/train.py
```

### Custom Configuration
```bash
python src/train.py --config custom_config.json
```

### With GPU
```bash
# Already configured in config.json as "device": "cuda"
# Just ensure CUDA is installed
```

### With CPU Only
Change `config.json`:
```json
{
  "device": "cpu"
}
```

---

## Understanding the Training Process

### What Happens During Training

1. **Initialization**
   - Global Generator and Discriminator created
   - 3 local copies created for each client
   - Datasets loaded (MNIST, Fashion-MNIST, CIFAR-10 subsampled)

2. **Communication Round** (repeated num_rounds times):
   ```
   a) Server sends global model to clients
   b) Each client trains locally:
      - Trains discriminator on real and generated data
      - Trains generator to fool discriminator
      - Different batch sizes per client (system heterogeneity)
   c) Clients upload trained weights
   d) Server aggregates using FedAvg:
      w_global = Σ(n_k / N) * w_local_k
      where n_k = client's samples, N = total samples
   e) Server broadcasts updated weights
   ```

3. **Metrics Tracked**
   - Generator Loss: How well generator trains
   - Discriminator Loss: Classification loss
   - Discriminator Accuracy: Real/Fake classification accuracy
   - Communication Rounds: Number of server-client interactions

4. **Output Generated**
   - CSV with all metrics
   - Plots showing loss and accuracy trends
   - Sample generated images every 5 rounds
   - Final trained models

---

## Datasets Used

| Client | Dataset | Size | Classes | Notes |
|--------|---------|------|---------|-------|
| Client 1 | MNIST | ~60K train | 10 | Handwritten digits |
| Client 2 | Fashion-MNIST | ~60K train | 10 | Clothing items |
| Client 3 | CIFAR-10 | ~10K train (subsampled) | 10 | Natural images (grayscale) |

**Why different datasets?**
- Simulates real federated scenarios with heterogeneous data
- Tests model's ability to learn generalizable representations
- Datasets auto-download on first run

---

## System Heterogeneity Handling

The project addresses **system heterogeneity** through:

### 1. Different Batch Sizes
- Client 1: batch_size=32 (processes 32 images per gradient step)
- Client 2: batch_size=64 (processes 64 images per gradient step)
- Client 3: batch_size=128 (processes 128 images per gradient step)

**Why this matters:**
- Simulates devices with different computational power
- Smaller batch sizes = slower processing, more memory efficient
- Larger batch sizes = faster processing, needs more memory
- GANs train with different effective learning rates per batch size

### 2. Graceful Aggregation
- Server doesn't wait for all clients
- Weights are aggregated using FedAvg with importance weighting
- Each client's contribution is weighted by sample count: `w_k = n_k / N`

### 3. Adaptive Training
- Each client trains for same epochs but different batch sizes
- Effective steps per client: `steps = dataset_size / batch_size`
- Client 3 completes more gradient steps than Client 1 in same epoch

---

## Results Interpretation

### Metrics CSV (`results/metrics.csv`)
```
round,client_id,generator_loss,discriminator_loss,discriminator_acc,communication_round
0,0,1.234,0.567,0.65,0
0,1,1.123,0.612,0.61,0
0,2,1.456,0.534,0.68,0
1,0,1.001,0.456,0.72,1
...
```

**What to look for:**
- Generator loss: Should decrease over rounds (smaller = better)
- Discriminator loss: Should stabilize around 0.5 (balanced training)
- Discriminator accuracy: Target ~50-70% (too high means generator is weak)

### Plots (`results/plots/training_metrics.png`)
- **Plot 1**: Generator loss trend (should decrease)
- **Plot 2**: Discriminator loss trend (should stabilize)
- **Plot 3**: Discriminator accuracy (should be stable)
- **Plot 4**: Normalized overview of all metrics

### Generated Samples (`results/samples/round_X.png`)
- 4x4 grid of generated images
- Quality improves with more rounds
- Shows generator learning progress

---

## Evaluation

### Automatic Evaluation
```bash
python evaluate.py
```

This generates:
- Console statistics
- Detailed report in `results/evaluation_report.txt`

### Manual Inspection
1. Check `results/metrics.csv` for loss values
2. Look at `results/plots/training_metrics.png` for trends
3. View generated images in `results/samples/`
4. Review logs in `logs/training_*.log`

---

## Troubleshooting

### Issue: CUDA Out of Memory
**Solution:**
```json
{
  "device": "cpu",
  "noise_dim": 50,
  "batch_sizes": [16, 32, 64]
}
```

### Issue: Training is Very Slow
**Solution:**
```json
{
  "num_rounds": 5,
  "epochs_per_client": 2,
  "batch_sizes": [64, 128, 256]
}
```

### Issue: Generated Images Look Bad
**Solution:**
- Increase `num_rounds` to 20+
- Increase `epochs_per_client` to 10+
- Wait for more communication rounds

### Issue: Discriminator Accuracy = 50% (Can't Distinguish Real/Fake)
- This is actually good! Generator is fooling discriminator
- Continue training

### Issue: Generator Loss is Very High and Not Decreasing
- Try different learning rates (e.g., 0.0001 or 0.0003)
- Increase number of rounds
- Check that discriminator is training

---

## Advanced Usage

### Running with Different Datasets
Modify `config.json`:
```json
{
  "datasets": ["mnist", "mnist", "mnist"]  // All same dataset
}
```

Available datasets:
- "mnist"
- "fashion_mnist"
- "cifar10_subset"

### Adjusting Learning Dynamics
```json
{
  "learning_rate": 0.0001,    // Lower = slower but more stable
  "beta1": 0.5,                // Adam momentum
  "epochs_per_client": 10      // More local computation
}
```

### Monitoring Training in Real-Time
```bash
tail -f logs/training_*.log
```

### Saving Checkpoints (Manual)
Add to `src/train.py`:
```python
if (round_num + 1) % 5 == 0:
    torch.save(generator.state_dict(), f'./results/gen_round_{round_num}.pth')
    torch.save(discriminator.state_dict(), f'./results/disc_round_{round_num}.pth')
```

---

## Performance Expectations

### On CPU (Modern CPU)
- Time per round: 30-60 seconds
- Total time (10 rounds): 5-10 minutes

### On GPU (NVIDIA RTX 3060)
- Time per round: 5-10 seconds
- Total time (10 rounds): 1-2 minutes

### Memory Usage
- GPU: ~2-4 GB (depending on batch size)
- CPU: ~1-2 GB

---

## For Your Teacher/Presentation

### Key Points to Highlight

1. **Federated Learning Implementation**
   - ✅ Distributed training across 3 clients
   - ✅ No raw data sharing
   - ✅ Server-side aggregation (FedAvg)
   - ✅ Communication round tracking

2. **GANs in Federated Setting**
   - ✅ Generator and Discriminator trained collaboratively
   - ✅ Each client maintains local copies
   - ✅ Global model aggregation every round
   - ✅ Sample image generation

3. **System Heterogeneity Handling**
   - ✅ Different batch sizes per client
   - ✅ Weighted aggregation by sample count
   - ✅ Graceful handling of slow clients
   - ✅ Adaptive local training

4. **Comprehensive Tracking**
   - ✅ Metrics CSV export
   - ✅ Loss and accuracy plots
   - ✅ Training logs
   - ✅ Per-client statistics
   - ✅ Communication round counting

---

## GitHub Setup

To upload to GitHub:

```bash
git init
git add .
git commit -m "Initial commit: Federated Learning with GANs"
git branch -M main
git remote add origin https://github.com/your-username/federated-learning-gan.git
git push -u origin main
```

Ensure `.gitignore` excludes large files:
- `datasets/` - Auto-downloaded
- `results/` - Generated during training
- `logs/` - Training logs
- `__pycache__/` - Python cache

---

## Support and Questions

For issues or clarifications:
1. Check the logs in `./logs/`
2. Review the console output during training
3. Check `results/evaluation_report.txt`
4. Refer to docstrings in source code

---

## Citation for Papers

If using papers provided in class, cite:
- Zhang, C., et al. (2021). "A survey on federated learning"
- Mammen, P. M. (2021). "Federated Learning: Opportunities and Challenges"
- Li, L., et al. (2020). "A review of applications in federated learning"

Good luck with your assignment!
