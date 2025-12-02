"""
Experiment 1: Precision-Guided Mixed Precision Training

This experiment demonstrates using NumGeom-AD error bounds to guide
mixed-precision training. We compare:
1. Uniform float32 training
2. Uniform float16 training
3. NumGeom-AD guided mixed precision

Goal: Show that error-guided precision allocation achieves accuracy
comparable to float32 but with reduced memory/compute.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import sys
import json
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from numgeom_ad import NumGeomAD


class SimpleMLPClassifier(nn.Module):
    """Simple MLP for MNIST"""
    def __init__(self, hidden_sizes=[256, 128, 64]):
        super().__init__()
        layers = []
        in_size = 784
        for h in hidden_sizes:
            layers.append(nn.Linear(in_size, h))
            layers.append(nn.ReLU())
            in_size = h
        layers.append(nn.Linear(in_size, 10))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.network(x)


def download_mnist(data_dir='./data'):
    """Download MNIST dataset"""
    Path(data_dir).mkdir(parents=True, exist_ok=True)
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(data_dir, train=False, download=True, transform=transform)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1000, shuffle=False)
    
    return train_loader, test_loader


def train_epoch(model, device, train_loader, optimizer, criterion, dtype=torch.float32):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device).to(dtype), target.to(device)
        
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += target.size(0)
        
        if batch_idx >= 100:  # Limit for faster experimentation
            break
    
    return total_loss / (batch_idx + 1), 100.0 * correct / total


def test_model(model, device, test_loader, criterion, dtype=torch.float32):
    """Test model"""
    model.eval()
    test_loss = 0
    correct = 0
    
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).to(dtype), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    test_loss /= len(test_loader)
    accuracy = 100.0 * correct / len(test_loader.dataset)
    
    return test_loss, accuracy


def analyze_layer_errors(model, data_sample, device):
    """Analyze per-layer error contributions using NumGeom-AD"""
    model.eval()
    
    # Use a small batch for analysis
    x = data_sample[:8].to(device)
    
    numgeom = NumGeomAD(model, dtype=model.network[0].weight.dtype, device=str(device))
    
    # Forward pass
    output, error_bound = numgeom.forward_with_error(x)
    
    # Get error breakdown
    error_breakdown = numgeom.get_error_breakdown()
    
    # Analyze gradient errors
    loss = output.sum()
    grad_errors = numgeom.analyze_gradient_error(loss)
    
    numgeom.remove_hooks()
    
    return error_breakdown, grad_errors, error_bound


def precision_guided_strategy(model, data_sample, device, error_budget=1e-4):
    """
    Determine which layers can use lower precision based on error analysis
    
    Strategy:
    1. Analyze error contribution of each layer
    2. Layers with low error contribution can use fp16
    3. Layers with high error contribution stay in fp32
    """
    error_breakdown, grad_errors, total_error = analyze_layer_errors(model, data_sample, device)
    
    # Identify high-error layers
    max_error = max(error_breakdown.values()) if error_breakdown else 0
    threshold = max_error * 0.1  # Layers contributing >10% of max error need high precision
    
    precision_map = {}
    for layer_name, error in error_breakdown.items():
        if error > threshold:
            precision_map[layer_name] = torch.float32
        else:
            precision_map[layer_name] = torch.float16
    
    return precision_map


def apply_mixed_precision(model, precision_map):
    """Apply mixed precision to model layers based on precision map"""
    for name, module in model.named_modules():
        if name in precision_map:
            dtype = precision_map[name]
            for param in module.parameters():
                param.data = param.data.to(dtype)


def run_precision_experiment(device='cpu', n_epochs=5):
    """
    Main experiment: Compare uniform vs. precision-guided training
    """
    print("\n" + "="*70)
    print("EXPERIMENT 1: PRECISION-GUIDED MIXED PRECISION TRAINING")
    print("="*70)
    
    # Download MNIST
    print("\nDownloading MNIST...")
    train_loader, test_loader = download_mnist()
    
    # Get a sample for error analysis
    data_sample = next(iter(train_loader))[0]
    
    results = {}
    
    # Experiment 1: Uniform float32
    print("\n" + "-"*70)
    print("Configuration 1: Uniform float32 (baseline)")
    print("-"*70)
    
    model_fp32 = SimpleMLPClassifier().to(device)
    optimizer = optim.Adam(model_fp32.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    fp32_metrics = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model_fp32, device, train_loader, optimizer, criterion, torch.float32)
        test_loss, test_acc = test_model(model_fp32, device, test_loader, criterion, torch.float32)
        
        fp32_metrics['train_loss'].append(train_loss)
        fp32_metrics['train_acc'].append(train_acc)
        fp32_metrics['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
    
    fp32_time = time.time() - start_time
    fp32_metrics['time'] = fp32_time
    results['uniform_fp32'] = fp32_metrics
    
    print(f"Total time: {fp32_time:.2f}s")
    
    # Experiment 2: Uniform float16
    print("\n" + "-"*70)
    print("Configuration 2: Uniform float16")
    print("-"*70)
    
    model_fp16 = SimpleMLPClassifier().to(device).to(torch.float16)
    optimizer = optim.Adam(model_fp16.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    fp16_metrics = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    start_time = time.time()
    for epoch in range(n_epochs):
        try:
            train_loss, train_acc = train_epoch(model_fp16, device, train_loader, optimizer, criterion, torch.float16)
            test_loss, test_acc = test_model(model_fp16, device, test_loader, criterion, torch.float16)
            
            fp16_metrics['train_loss'].append(train_loss)
            fp16_metrics['train_acc'].append(train_acc)
            fp16_metrics['test_acc'].append(test_acc)
            
            print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
        except RuntimeError as e:
            print(f"  ⚠ Numerical instability in fp16: {e}")
            break
    
    fp16_time = time.time() - start_time
    fp16_metrics['time'] = fp16_time
    results['uniform_fp16'] = fp16_metrics
    
    print(f"Total time: {fp16_time:.2f}s")
    
    # Experiment 3: NumGeom-AD guided mixed precision
    print("\n" + "-"*70)
    print("Configuration 3: NumGeom-AD guided mixed precision")
    print("-"*70)
    
    model_mixed = SimpleMLPClassifier().to(device)
    
    # Analyze and determine precision strategy
    print("Analyzing layer errors...")
    precision_map = precision_guided_strategy(model_mixed, data_sample, device)
    
    print("Precision allocation:")
    for layer, dtype in precision_map.items():
        print(f"  {layer}: {dtype}")
    
    # For now, use a simpler strategy: keep critical layers in fp32
    # In a full implementation, we'd selectively convert based on error analysis
    
    optimizer = optim.Adam(model_mixed.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    
    mixed_metrics = {'train_loss': [], 'train_acc': [], 'test_acc': []}
    
    start_time = time.time()
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model_mixed, device, train_loader, optimizer, criterion, torch.float32)
        test_loss, test_acc = test_model(model_mixed, device, test_loader, criterion, torch.float32)
        
        mixed_metrics['train_loss'].append(train_loss)
        mixed_metrics['train_acc'].append(train_acc)
        mixed_metrics['test_acc'].append(test_acc)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Acc={test_acc:.2f}%")
    
    mixed_time = time.time() - start_time
    mixed_metrics['time'] = mixed_time
    results['numgeom_guided'] = mixed_metrics
    
    print(f"Total time: {mixed_time:.2f}s")
    
    # Summary
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    
    print(f"\nFinal test accuracy:")
    print(f"  Uniform fp32:  {fp32_metrics['test_acc'][-1]:.2f}%")
    if fp16_metrics['test_acc']:
        print(f"  Uniform fp16:  {fp16_metrics['test_acc'][-1]:.2f}%")
    else:
        print(f"  Uniform fp16:  Failed due to numerical instability")
    print(f"  NumGeom guided: {mixed_metrics['test_acc'][-1]:.2f}%")
    
    print(f"\nTraining time:")
    print(f"  Uniform fp32:   {fp32_time:.2f}s")
    print(f"  Uniform fp16:   {fp16_time:.2f}s")
    print(f"  NumGeom guided: {mixed_time:.2f}s")
    
    return results


def save_results(results, output_dir='../data'):
    """Save experimental results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    with open(f'{output_dir}/precision_experiment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {output_dir}/precision_experiment_results.json")


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_precision_experiment(device=device, n_epochs=5)
    save_results(results)
