#!/usr/bin/env python3
"""
Download MNIST dataset and save in format LibTorch can read.

This creates .pt files containing tensors that can be loaded directly in C++.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import os
import sys

def download_and_save_mnist(data_dir="./data"):
    """Download MNIST and save as PyTorch tensors."""
    
    os.makedirs(data_dir, exist_ok=True)
    
    print("Downloading MNIST dataset...")
    
    # Download training data
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transforms.ToTensor()
    )
    
    # Download test data
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transforms.ToTensor()
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")
    
    # Convert to tensors
    print("\nConverting to tensors...")
    
    train_images = []
    train_labels = []
    for img, label in train_dataset:
        train_images.append(img)
        train_labels.append(label)
    
    test_images = []
    test_labels = []
    for img, label in test_dataset:
        test_images.append(img)
        test_labels.append(label)
    
    train_images = torch.stack(train_images)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_images = torch.stack(test_images)
    test_labels = torch.tensor(test_labels, dtype=torch.long)
    
    print(f"Train images shape: {train_images.shape}")
    print(f"Train labels shape: {train_labels.shape}")
    print(f"Test images shape: {test_images.shape}")
    print(f"Test labels shape: {test_labels.shape}")
    
    # Save as .pt files
    train_data_path = os.path.join(data_dir, "mnist_train_images.pt")
    train_labels_path = os.path.join(data_dir, "mnist_train_labels.pt")
    test_data_path = os.path.join(data_dir, "mnist_test_images.pt")
    test_labels_path = os.path.join(data_dir, "mnist_test_labels.pt")
    
    print("\nSaving tensors...")
    torch.save(train_images, train_data_path)
    torch.save(train_labels, train_labels_path)
    torch.save(test_images, test_data_path)
    torch.save(test_labels, test_labels_path)
    
    print(f"\n✅ MNIST dataset ready!")
    print(f"   Training data: {train_data_path}")
    print(f"   Training labels: {train_labels_path}")
    print(f"   Test data: {test_data_path}")
    print(f"   Test labels: {test_labels_path}")
    
    return True

if __name__ == "__main__":
    data_dir = sys.argv[1] if len(sys.argv) > 1 else "./data"
    download_and_save_mnist(data_dir)
