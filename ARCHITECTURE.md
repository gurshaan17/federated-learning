# Federated Learning with GANs - Architecture & Implementation Guide

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     FEDERATED LEARNING SERVER                       │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  Global Generator & Discriminator Models                     │  │
│  │  - Maintains current global weights                          │  │
│  │  - No data storage                                           │  │
│  └──────────────────────────────────────────────────────────────┘  │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │  FedAvg Aggregation (Weighted Average)                       │  │
│  │  w_t+1 = Σ (n_k / N) * w_t,k                                 │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
     ↑                    ↑                    ↑
     │                    │                    │
  Download model       Download model       Download model
  Update weights       Update weights       Update weights
     │                    │                    │
     ↓                    ↓                    ↓
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   CLIENT 1   │  │   CLIENT 2   │  │   CLIENT 3   │
├──────────────┤  ├──────────────┤  ├──────────────┤
│ Gen + Disc   │  │ Gen + Disc   │  │ Gen + Disc   │
│              │  │              │  │              │
│ MNIST        │  │ Fashion      │  │ CIFAR-10     │
│ 60K samples  │  │ MNIST        │  │ 10K samples  │
│ BatchSize=32 │  │ 60K samples  │  │ BatchSize=   │
│              │  │ BatchSize=64 │  │ 128          │
│              │  │              │  │              │
│ Train:       │  │ Train:       │  │ Train:       │
│ 5 epochs     │  │ 5 epochs     │  │ 5 epochs     │
│ local        │  │ local        │  │ local        │
│ (System      │  │ (System      │  │ (System      │
│ Hetero.)     │  │ Hetero.)     │  │ Hetero.)     │
└──────────────┘  └──────────────┘  └──────────────┘
```

## 🔄 Training Loop (Each Communication Round)

```
START OF ROUND t
│
├─ [SERVER] Broadcast global weights to all clients
│
├─ [CLIENT 1] Local Training
│   ├─ Load MNIST (60K samples)
│   ├─ Create batches of 32
│   ├─ For 5 epochs:
│   │   ├─ Train Discriminator on real + fake data
│   │   └─ Train Generator to fool Discriminator
│   └─ Calculate: Gen Loss, Disc Loss, Accuracy
│
├─ [CLIENT 2] Local Training (parallel)
│   ├─ Load Fashion-MNIST (60K samples)
│   ├─ Create batches of 64
│   ├─ For 5 epochs:
│   │   ├─ Train Discriminator on real + fake data
│   │   └─ Train Generator to fool Discriminator
│   └─ Calculate: Gen Loss, Disc Loss, Accuracy
│
├─ [CLIENT 3] Local Training (parallel)
│   ├─ Load CIFAR-10 (10K samples)
│   ├─ Create batches of 128
│   ├─ For 5 epochs:
│   │   ├─ Train Discriminator on real + fake data
│   │   └─ Train Generator to fool Discriminator
│   └─ Calculate: Gen Loss, Disc Loss, Accuracy
│
├─ [CLIENTS] Upload trained weights to server
│   ├─ Client 1: weights (importance: 60K/130K)
│   ├─ Client 2: weights (importance: 60K/130K)
│   └─ Client 3: weights (importance: 10K/130K)
│
├─ [SERVER] FedAvg Aggregation
│   ├─ Aggregate Generator: w_g = 0.46*w1_g + 0.46*w2_g + 0.08*w3_g
│   └─ Aggregate Discriminator: w_d = 0.46*w1_d + 0.46*w2_d + 0.08*w3_d
│
├─ [SERVER] Track Metrics
│   ├─ Average Generator Loss
│   ├─ Average Discriminator Loss
│   ├─ Average Discriminator Accuracy
│   └─ Communication Round Counter
│
└─ END OF ROUND - Repeat for next round
```

## 🧠 Model Architecture

### Generator
```
Input: Random Noise (batch_size, 100)
  ↓
Linear Layer: (batch_size, 100) → (batch_size, 12544)
  ↓
Reshape: (batch_size, 12544) → (batch_size, 256, 7, 7)
  ↓
ConvTranspose2d: (batch_size, 256, 7, 7) → (batch_size, 128, 14, 14)
  ↓ [BatchNorm + ReLU]
ConvTranspose2d: (batch_size, 128, 14, 14) → (batch_size, 64, 28, 28)
  ↓ [BatchNorm + ReLU]
Conv2d: (batch_size, 64, 28, 28) → (batch_size, 1, 28, 28)
  ↓
Tanh Activation: Output in [-1, 1]
  ↓
Output: Generated Images (batch_size, 1, 28, 28)
```

### Discriminator
```
Input: Real or Generated Images (batch_size, 1, 28, 28)
  ↓
Conv2d: (batch_size, 1, 28, 28) → (batch_size, 64, 14, 14)
  ↓ [LeakyReLU(0.2)]
Conv2d: (batch_size, 64, 14, 14) → (batch_size, 128, 7, 7)
  ↓ [BatchNorm + LeakyReLU(0.2)]
Conv2d: (batch_size, 128, 7, 7) → (batch_size, 256, 3, 3)
  ↓ [BatchNorm + LeakyReLU(0.2)]
Adaptive Avg Pool: → (batch_size, 256, 1, 1)
  ↓
Flatten: (batch_size, 256)
  ↓
Linear: (batch_size, 256) → (batch_size, 1)
  ↓
Sigmoid: Output in [0, 1]
  ↓
Output: Probability is Real (batch_size, 1)
```

## 🔐 System Heterogeneity Handling

### Problem: Different Device Capabilities

```
Real-World Scenario:
  Device 1 (Phone): Limited compute, batch_size = 32
  Device 2 (Laptop): Medium compute, batch_size = 64
  Device 3 (Desktop): High compute, batch_size = 128
```

### Solution: Adaptive Batch Sizes

```
Per-Device Configuration:
┌────────────┬───────────┬─────────────────┬──────────────────┐
│   Device   │ Batch     │  Effective      │  Computation     │
│            │  Size     │  Steps/Epoch    │  Time            │
├────────────┼───────────┼─────────────────┼──────────────────┤
│ Client 1   │    32     │  60000/32=1875  │ 1875 steps       │
│ (MNIST)    │           │                 │ (slower device)  │
├────────────┼───────────┼─────────────────┼──────────────────┤
│ Client 2   │    64     │  60000/64=938   │ 938 steps        │
│ (Fashion)  │           │                 │ (medium device)  │
├────────────┼───────────┼─────────────────┼──────────────────┤
│ Client 3   │    128    │  10000/128=78   │ 78 steps         │
│ (CIFAR-10) │           │                 │ (fast device)    │
└────────────┴───────────┴─────────────────┴──────────────────┘
```

### Graceful Aggregation

```
Client Data Contributions:
  Client 1: 60,000 samples (46.2% weight)
  Client 2: 60,000 samples (46.2% weight)
  Client 3: 10,000 samples (7.7% weight)
            ─────────────────
            130,000 total

FedAvg Weights:
  w_global = 0.462 * w_client1 + 0.462 * w_client2 + 0.077 * w_client3

Benefits:
  ✓ Larger clients have more influence
  ✓ Smaller clients still participate
  ✓ Weighted by data quantity
  ✓ Handles stragglers naturally
```

## 📊 Metrics Tracking

```
Per Round, Per Client:
┌──────────────────────┬───────────┬──────────────────────────┐
│      Metric          │ Short     │     Interpretation       │
├──────────────────────┼───────────┼──────────────────────────┤
│ Generator Loss       │ G_loss    │ Lower = better generator │
│                      │           │ (improving image quality)│
├──────────────────────┼───────────┼──────────────────────────┤
│ Discriminator Loss   │ D_loss    │ Should stabilize ~0.5-0.7│
│                      │           │ (balanced adversarial)   │
├──────────────────────┼───────────┼──────────────────────────┤
│ Discriminator Acc    │ D_acc     │ 50-70% optimal           │
│                      │           │ (50% = can't tell        │
│                      │           │  70% = slightly biased)  │
├──────────────────────┼───────────┼──────────────────────────┤
│ Communication Rounds │ Comm_round│ Number of server-client  │
│                      │           │ synchronizations         │
└──────────────────────┴───────────┴──────────────────────────┘
```

## 🎯 Expected Training Dynamics

### Round 1 (Initial)
```
Generator Loss: HIGH (random noise → random images)
Discriminator Loss: HIGH (untrained)
Discriminator Acc: ~50% (random guessing)
```

### Rounds 2-5 (Improving)
```
Generator Loss: DECREASING (learning to generate)
Discriminator Loss: STABILIZING (finding balance)
Discriminator Acc: INCREASING → STABILIZING (improving differentiation)
```

### Rounds 6-10 (Convergence)
```
Generator Loss: LOW & STABLE (good image generation)
Discriminator Loss: ~0.5-0.7 (balanced)
Discriminator Acc: ~60-70% (discriminator skilled)
```

## 💾 Data Flow & State Management

```
INITIALIZATION:
  Global Models (1 Generator + 1 Discriminator)
          ↓
  Create Local Copies for Each Client (3x Generator + 3x Discriminator)
          ↓
  Initialize with Same Weights

ROUND t:
  [Server] broadcast_weights(clients, global_weights)
          ↓
  [Clients] set_model_weights(global_weights)
          ↓
  [Clients] train() → update local_weights
          ↓
  [Clients] get_model_weights() → upload to server
          ↓
  [Server] aggregate(clients) → compute global_weights
          ↓
  [Metrics] track(round, client, losses, accuracy)
          ↓
  Save to CSV, Generate Plots
```

## 🔄 FedAvg Algorithm (Pseudocode)

```python
# Server-side aggregation
def fedavg_aggregate(clients):
    total_samples = sum(len(c.data) for c in clients)
    
    # Initialize aggregated weights to zero
    agg_gen_weights = zeros_like(global_generator.weights)
    agg_disc_weights = zeros_like(global_discriminator.weights)
    
    # Weighted average
    for client in clients:
        client_weight = len(client.data) / total_samples
        
        gen_w, disc_w = client.get_model_weights()
        
        agg_gen_weights += client_weight * gen_w
        agg_disc_weights += client_weight * disc_w
    
    # Update global models
    global_generator.load_weights(agg_gen_weights)
    global_discriminator.load_weights(agg_disc_weights)
    
    return agg_gen_weights, agg_disc_weights

# Client-side training
def client_train_epoch():
    for batch_real_data in data_loader:
        # ===== Train Discriminator =====
        real_output = discriminator(batch_real_data)
        real_loss = binary_cross_entropy(real_output, ones)
        
        noise = random_normal(batch_size, noise_dim)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data.detach())
        fake_loss = binary_cross_entropy(fake_output, zeros)
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        optimizer_d.step()
        
        # ===== Train Generator =====
        noise = random_normal(batch_size, noise_dim)
        fake_data = generator(noise)
        fake_output = discriminator(fake_data)
        
        g_loss = binary_cross_entropy(fake_output, ones)
        g_loss.backward()
        optimizer_g.step()
        
        # Calculate accuracy
        accuracy = (real_correct + fake_correct) / 2
```

## 📈 Training Curves (Expected)

```
Generator Loss:           Discriminator Loss:      Discriminator Accuracy:
1.0 ├─ ╱╲                1.0 ├─ ╱╲                1.0 ├─ ─────────
    │  ╱  ╲                   │  ╱  ╲                   │
0.8 │ ╱    ╲              0.8 │ ╱    ╲              0.8 │    ╱╲___
    │╱      ╲_            0.6 │      ╱  ╲__            │   ╱
0.6 │        ╲_          0.4 │     ╱      ╲__       0.6 │  ╱
    │         ╲_            │____╱           ╲      │ ╱
0.4 └──────────╲_          0.2                   │0.4└──────────
    0 2 4 6 8 10           0 2 4 6 8 10          0 2 4 6 8 10
    Rounds                 Rounds                Rounds

↓
Expected Behavior:
- Gen Loss ↘ (generator improves)
- Disc Loss → ~0.5 (equilibrium)
- Disc Acc → stable 60-70% (balanced)
```

## 🔗 Privacy Aspects

```
Without Federated Learning:
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Raw     │  │ Raw     │  │ Raw     │
│ Data 1  │  │ Data 2  │  │ Data 3  │
└────┬────┘  └────┬────┘  └────┬────┘
     │           │           │
     └───────────┴───────────┘
           ↓
       [RISK: DATA BREACH]
       Central Database with
       all sensitive data

With Federated Learning:
┌─────────┐  ┌─────────┐  ┌─────────┐
│ Local   │  │ Local   │  │ Local   │
│ Data 1  │  │ Data 2  │  │ Data 3  │
│ STAYS   │  │ STAYS   │  │ STAYS   │
└────┬────┘  └────┬────┘  └────┬────┘
     │ weights   │ weights   │ weights
     │ only      │ only      │ only
     └───────────┴───────────┘
           ↓
       Central Server
       Only sees MODEL WEIGHTS
       (numbers, not data)
       ✓ SAFE: No raw data shared
```

## 📝 Key Formulas

### FedAvg Aggregation
```
w_{t+1} = Σ_{k=1}^{K} (n_k / N) * w_{t,k}

where:
  K = number of clients
  n_k = number of samples for client k
  N = Σ n_k (total samples)
  w_{t,k} = weights of client k at round t
  w_{t+1} = aggregated weights for round t+1
```

### Generator Loss
```
L_G = -E[log(D(G(z)))]

where:
  G = generator
  D = discriminator
  z = random noise
  G(z) = generated image
  D(G(z)) = discriminator's probability that generated image is real
  Goal: Maximize D(G(z)) → minimize L_G
```

### Discriminator Loss
```
L_D = -E[log(D(x))] - E[log(1 - D(G(z)))]

where:
  x = real image
  D(x) = probability real image is real
  D(G(z)) = probability fake image is real
  Goal: Minimize L_D (maximize both terms)
```

## ✅ Verification Checklist

This implementation includes:

- [x] 3 Distributed Clients
- [x] Different Datasets per Client
- [x] System Heterogeneity (batch sizes)
- [x] GAN Architecture (Generator + Discriminator)
- [x] Federated Training Loop
- [x] FedAvg Aggregation
- [x] Metrics Tracking
- [x] Communication Round Counting
- [x] Privacy Preservation (no raw data sharing)
- [x] Visualization & Reporting
- [x] Production-Ready Code
- [x] Comprehensive Documentation

Ready for submission to GitHub! 🎉
