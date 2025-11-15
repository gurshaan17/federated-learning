# Federated Learning with GANs - File Summary

## 📁 Complete Project File Structure

This document provides a summary of all files in your federated learning project.

---

## 🔧 Configuration & Setup Files

### 1. `config.json`
- **Purpose**: Main configuration file for training
- **What it contains**: 
  - Number of clients (3)
  - Training rounds (10)
  - Model hyperparameters
  - Dataset selection and batch sizes
  - Device settings (cuda/cpu)
- **Edit this to**: Customize training parameters

### 2. `requirements.txt`
- **Purpose**: Python package dependencies
- **Install with**: `pip install -r requirements.txt`
- **Contains**: PyTorch, NumPy, Pandas, Matplotlib, etc.

### 3. `.gitignore`
- **Purpose**: Specifies files to exclude from Git
- **Excludes**: Large datasets, results, logs, Python cache

---

## 📝 Main Documentation

### 4. `README.md`
- **Purpose**: Project overview and quick reference
- **Contents**:
  - Features overview
  - Project structure
  - Installation steps
  - Usage instructions
  - Configuration guide
  - Troubleshooting
  - Future enhancements

### 5. `SETUP_GUIDE.md` (THIS FILE)
- **Purpose**: Comprehensive setup and usage guide
- **Contents**:
  - Detailed installation
  - Configuration explanations
  - Training process overview
  - Results interpretation
  - Troubleshooting
  - Advanced usage
  - Performance expectations

---

## 🚀 Executable Scripts

### 6. `quickstart.py` (Root Level)
- **Purpose**: Quick start helper script
- **Usage**: `python quickstart.py`
- **What it does**:
  - Checks Python version
  - Verifies dependencies
  - Creates necessary directories
  - Launches training

### 7. `evaluate.py` (Root Level)
- **Purpose**: Evaluate training results
- **Usage**: `python evaluate.py`
- **Generates**:
  - Statistics printout
  - Evaluation report in `results/evaluation_report.txt`
  - Per-client analysis

---

## 📦 Source Code (`src/` directory)

### 8. `src/__init__.py`
- **Purpose**: Package initialization
- **Imports**: All main classes and functions
- **Allows**: `from src import ...`

### 9. `src/models.py`
- **Purpose**: Neural network model definitions
- **Classes**:
  - `Generator`: Takes noise, generates images
  - `Discriminator`: Classifies real vs fake images
- **Functions**:
  - `create_models()`: Initialize both models
  - `count_parameters()`: Count model parameters

### 10. `src/client.py`
- **Purpose**: Federated client implementation
- **Class**: `FederatedClient`
- **Capabilities**:
  - Local GAN training
  - Weight extraction/setting
  - Metrics tracking
  - Train epochs with adaptive batch sizes

### 11. `src/server.py`
- **Purpose**: Federated server implementation
- **Class**: `FederatedServer`
- **Features**:
  - FedAvg aggregation algorithm
  - Weight broadcasting
  - Aggregation statistics

### 12. `src/data_loader.py`
- **Purpose**: Dataset loading and preprocessing
- **Functions**:
  - `get_data_loader()`: Load single dataset
  - `get_client_data_loaders()`: Load all client datasets
- **Datasets**:
  - MNIST
  - Fashion-MNIST
  - CIFAR-10 (subsampled)

### 13. `src/utils.py`
- **Purpose**: Utility functions and classes
- **Key Classes**:
  - `MetricsTracker`: Track training metrics
  - `Logger`: File and console logging
- **Key Functions**:
  - `create_noise()`: Generate random noise
  - `calculate_discriminator_accuracy()`: Compute accuracy
  - `get_device()`: Get cuda/cpu device
  - `save_checkpoint()`: Save model weights
  - `plot_training_metrics()`: Visualize training
  - `visualize_generated_images()`: Show generated samples

### 14. `src/train.py`
- **Purpose**: Main training script
- **Usage**: `python src/train.py [--config config.json]`
- **What it does**:
  - Loads configuration
  - Initializes models and clients
  - Runs federated learning loop
  - Tracks metrics
  - Saves results

---

## 📊 Output Directories (Auto-Created)

### `datasets/` Directory
- **Auto-created**: First run
- **Contains**: Downloaded datasets
  - MNIST (60K+ images)
  - Fashion-MNIST (60K+ images)
  - CIFAR-10 subset (10K images)
- **Size**: ~150 MB total

### `results/` Directory
- **Auto-created**: After first training run
- **Contains**:
  - `metrics.csv`: All training metrics
  - `plots/training_metrics.png`: Training plots
  - `samples/round_X.png`: Generated images
  - `final_generator.pth`: Final generator weights
  - `final_discriminator.pth`: Final discriminator weights
  - `evaluation_report.txt`: Detailed analysis

### `logs/` Directory
- **Auto-created**: During first training
- **Contains**:
  - `training_YYYYMMDD_HHMMSS.log`: Detailed training logs

---

## 🔄 Data Flow

```
config.json
    ↓
src/train.py (main script)
    ├── Loads config
    ├── src/models.py (create Generator & Discriminator)
    ├── src/data_loader.py (load 3 client datasets)
    ├── src/client.py (create 3 FederatedClient objects)
    ├── src/server.py (create FederatedServer)
    │
    └── Training Loop (10 rounds):
        ├── Client 1 trains locally (batch_size=32)
        ├── Client 2 trains locally (batch_size=64)
        ├── Client 3 trains locally (batch_size=128)
        ├── src/server.py aggregates weights (FedAvg)
        ├── src/utils.py tracks metrics
        └── Results saved to results/
            ├── metrics.csv
            ├── plots/
            ├── samples/
            └── .pth files

evaluate.py
    └── Reads results/ and generates report
```

---

## 🎯 Getting Started

1. **Install**: `pip install -r requirements.txt`
2. **Train**: `python quickstart.py` or `python src/train.py`
3. **Evaluate**: `python evaluate.py`
4. **View**: Check `results/` directory

---

## 💡 Key Features by File

| Feature | File(s) |
|---------|---------|
| 3 Clients | `src/client.py` |
| GAN Training | `src/models.py` |
| Federated Aggregation | `src/server.py` |
| System Heterogeneity | `src/data_loader.py` + config.json |
| Metrics Tracking | `src/utils.py` |
| Visualization | `src/utils.py` |
| Configuration | `config.json` |
| Training Loop | `src/train.py` |
| Quick Start | `quickstart.py` |
| Results Analysis | `evaluate.py` |

---

## 📋 File Modifications for Custom Use

### To Change Number of Clients
Modify `config.json`:
```json
{
  "num_clients": 5,
  "datasets": ["mnist", "fashion_mnist", "cifar10_subset", "mnist", "fashion_mnist"],
  "batch_sizes": [32, 48, 64, 80, 128]
}
```

### To Use Different Datasets
Modify `config.json`:
```json
{
  "datasets": ["cifar10_subset", "cifar10_subset", "cifar10_subset"]
}
```

### To Train Longer
Modify `config.json`:
```json
{
  "num_rounds": 50,
  "epochs_per_client": 10
}
```

### To Use CPU Only
Modify `config.json`:
```json
{
  "device": "cpu"
}
```

---

## ✅ Checklist Before Submission

- [ ] All files in correct directories
- [ ] `requirements.txt` has all dependencies
- [ ] `config.json` properly configured
- [ ] `src/` directory has all 6 Python files
- [ ] README.md and documentation complete
- [ ] `.gitignore` set up correctly
- [ ] Code follows PEP8 style
- [ ] Comments explain key sections
- [ ] Training completes without errors
- [ ] Results are generated and saved

---

## 🐛 Common File Issues

**Issue**: `ModuleNotFoundError: No module named 'src'`
- **Fix**: Run from project root: `python src/train.py`

**Issue**: `config.json not found`
- **Fix**: Ensure config.json in project root, not in subdirectory

**Issue**: `Datasets already exist` warnings
- **Fix**: Normal, safe to ignore - datasets are cached

**Issue**: Permission denied on logs/
- **Fix**: Run with appropriate permissions or delete logs/ directory

---

## 📚 File Relationships

```
src/train.py (orchestrator)
├── imports config.json
├── uses src/models.py (Generator, Discriminator)
├── uses src/data_loader.py (load datasets)
├── creates src/client.py objects (local training)
├── creates src/server.py object (aggregation)
└── uses src/utils.py (logging, metrics, visualization)

src/client.py (client object)
├── contains Generator & Discriminator models
├── trains locally
├── uses src/utils.py for metrics

src/server.py (server object)
├── aggregates from clients using FedAvg
└── broadcasts back to clients

src/utils.py (helpers)
├── MetricsTracker (csv writing)
├── Logger (file & console logging)
└── Visualization functions (plots, images)

quickstart.py (entry point)
└── calls src/train.py

evaluate.py (post-training)
└── reads results/ outputs
```

---

## 🎓 For Your Assignment

**What to show your teacher:**
1. This file structure in GitHub repository
2. Run `python quickstart.py` to show training
3. Show `results/metrics.csv` for metrics
4. Show `results/plots/training_metrics.png` for visualization
5. Show `results/samples/` for generated images
6. Explain system heterogeneity via `config.json` batch sizes
7. Show FedAvg aggregation in `src/server.py`
8. Explain GAN architecture in `src/models.py`

**Key files to highlight:**
- `README.md` - Project overview
- `src/train.py` - Main training logic
- `src/server.py` - Aggregation (FedAvg)
- `src/client.py` - Client training logic
- `config.json` - System heterogeneity setup

---

This file structure is **production-ready** and **GitHub-ready**!
