"""
Quick Start Script - Run this to start training
"""

import subprocess
import sys
import os

def check_python_version():
    """Check if Python version is 3.7+"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 7):
        print("Error: Python 3.7+ required")
        sys.exit(1)
    print(f"✓ Python {version.major}.{version.minor} detected")

def check_dependencies():
    """Check if all dependencies are installed"""
    try:
        import torch
        import torchvision
        import numpy
        import pandas
        import matplotlib
        print("✓ All required packages installed")
        return True
    except ImportError as e:
        print(f"✗ Missing package: {e}")
        print("\nInstall dependencies with:")
        print("  pip install -r requirements.txt")
        return False

def create_directories():
    """Create necessary directories"""
    dirs = ['./datasets', './results', './results/samples', './results/plots', './logs']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
    print("✓ Directories created")

def main():
    print("="*80)
    print("Federated Learning with GANs - Quick Start")
    print("="*80)
    
    # Check Python version
    print("\n[1/4] Checking Python version...")
    check_python_version()
    
    # Check dependencies
    print("\n[2/4] Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    # Create directories
    print("\n[3/4] Creating directories...")
    create_directories()
    
    # Start training
    print("\n[4/4] Starting training...")
    print("\n" + "="*80)
    
    # Run training
    subprocess.run([sys.executable, 'src/train.py', '--config', 'config.json'])

if __name__ == "__main__":
    main()
