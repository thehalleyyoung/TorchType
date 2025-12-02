#!/usr/bin/env python3
"""
Download MNIST dataset for C++ quantization demo using PyTorch
"""

import os
import torch
import torchvision

def download_mnist():
    """Download MNIST dataset files using PyTorch"""
    
    data_dir = 'data'
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading MNIST training set...")
    try:
        train_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=True,
            download=True
        )
        print(f"✓ Downloaded {len(train_dataset)} training samples")
    except Exception as e:
        print(f"✗ Error downloading training set: {e}")
        return False
    
    print("\nDownloading MNIST test set...")
    try:
        test_dataset = torchvision.datasets.MNIST(
            root=data_dir,
            train=False,
            download=True
        )
        print(f"✓ Downloaded {len(test_dataset)} test samples")
    except Exception as e:
        print(f"✗ Error downloading test set: {e}")
        return False
    
    # Check that raw files exist
    raw_dir = os.path.join(data_dir, 'MNIST', 'raw')
    if os.path.exists(raw_dir):
        files = os.listdir(raw_dir)
        print(f"\n✓ Raw files created in {raw_dir}:")
        for f in files:
            print(f"  - {f}")
    
    return True

if __name__ == '__main__':
    print("╔═══════════════════════════════════════════════╗")
    print("║  Downloading MNIST Dataset for C++ Demo      ║")
    print("╚═══════════════════════════════════════════════╝\n")
    
    success = download_mnist()
    
    if success:
        print("\n✓ MNIST dataset downloaded successfully!")
        print("\nYou can now run:")
        print("  cd build")
        print("  ./mnist_real_quantization")
    else:
        print("\nDownload failed. The demo will use synthetic data.")

