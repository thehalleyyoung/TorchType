"""
Experiment 2: Gradient Instability Detection

This experiment demonstrates using NumGeom-AD to detect numerical
instabilities during training that would otherwise be silent failures.

We create pathological training scenarios and show that NumGeom-AD
correctly identifies the problematic layers/operations.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from numgeom_ad import NumGeomAD


class PathologicalModel1(nn.Module):
    """Model with saturating softmax"""
    def __init__(self, saturation_factor=1.0):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 10)
        self.saturation_factor = saturation_factor
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x) * self.saturation_factor  # Multiply logits
        return self.softmax(x)


class PathologicalModel2(nn.Module):
    """Model with vanishing gradients (very deep + saturating activations)"""
    def __init__(self, depth=10):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.Linear(20, 20))
            layers.append(nn.Tanh())  # Saturating activation
        layers.append(nn.Linear(20, 10))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PathologicalModel3(nn.Module):
    """Model with exploding gradients (large weights)"""
    def __init__(self, weight_scale=1.0):
        super().__init__()
        self.fc1 = nn.Linear(10, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, 10)
        
        # Initialize with large weights
        with torch.no_grad():
            self.fc1.weight.mul_(weight_scale)
            self.fc2.weight.mul_(weight_scale)
            self.fc3.weight.mul_(weight_scale)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def monitor_training_with_numgeom(model, train_data, n_steps=50, device='cpu'):
    """
    Train model while monitoring with NumGeom-AD
    
    Returns error bounds and warnings over training
    """
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    numgeom = NumGeomAD(model, dtype=torch.float32, device=str(device))
    
    history = {
        'step': [],
        'loss': [],
        'error_bound': [],
        'n_warnings': [],
        'max_grad_error': [],
        'grad_norm': []
    }
    
    for step in range(n_steps):
        # Random batch
        x, y = train_data
        x, y = x.to(device), y.to(device)
        
        optimizer.zero_grad()
        
        # Forward with error tracking
        output, error_bound = numgeom.forward_with_error(x)
        loss = criterion(output, y)
        
        # Backward
        loss.backward()
        
        # Analyze gradient errors
        grad_errors = numgeom.analyze_gradient_error(loss)
        max_grad_error = max(grad_errors.values()) if grad_errors else 0
        
        # Check stability
        warnings = numgeom.check_stability(threshold=1e-5)
        
        # Compute gradient norm
        grad_norm = 0
        for p in model.parameters():
            if p.grad is not None:
                grad_norm += p.grad.norm().item() ** 2
        grad_norm = np.sqrt(grad_norm)
        
        # Record
        history['step'].append(step)
        history['loss'].append(loss.item())
        history['error_bound'].append(error_bound)
        history['n_warnings'].append(len(warnings))
        history['max_grad_error'].append(max_grad_error)
        history['grad_norm'].append(grad_norm)
        
        # Take optimizer step
        optimizer.step()
        
        if step % 10 == 0:
            print(f"Step {step}: Loss={loss.item():.4f}, ErrorBound={error_bound:.2e}, Warnings={len(warnings)}, GradNorm={grad_norm:.2f}")
    
    numgeom.remove_hooks()
    
    return history


def run_pathological_experiments(device='cpu'):
    """
    Run experiments on pathological models
    """
    print("\n" + "="*70)
    print("EXPERIMENT 2: GRADIENT INSTABILITY DETECTION")
    print("="*70)
    
    # Generate synthetic data
    torch.manual_seed(42)
    train_x = torch.randn(64, 10)
    train_y = torch.randn(64, 10)
    
    all_results = {}
    
    # Experiment 2.1: Saturating softmax
    print("\n" + "-"*70)
    print("Experiment 2.1: Saturating Softmax Detection")
    print("-"*70)
    
    saturation_factors = [1.0, 10.0, 100.0]
    
    for factor in saturation_factors:
        print(f"\nSaturation factor: {factor}")
        model = PathologicalModel1(saturation_factor=factor)
        history = monitor_training_with_numgeom(model, (train_x, train_y), n_steps=30, device=device)
        
        # Check if warnings increase with saturation
        avg_warnings = np.mean(history['n_warnings'])
        max_error = np.max(history['error_bound'])
        
        print(f"  Average warnings per step: {avg_warnings:.1f}")
        print(f"  Max error bound: {max_error:.2e}")
        
        all_results[f'saturating_softmax_factor_{factor}'] = history
    
    # Experiment 2.2: Vanishing gradients
    print("\n" + "-"*70)
    print("Experiment 2.2: Vanishing Gradient Detection")
    print("-"*70)
    
    depths = [3, 5, 10]
    
    for depth in depths:
        print(f"\nNetwork depth: {depth}")
        model = PathologicalModel2(depth=depth)
        train_x_deep = torch.randn(64, 20)
        train_y_deep = torch.randn(64, 10)
        
        history = monitor_training_with_numgeom(model, (train_x_deep, train_y_deep), n_steps=30, device=device)
        
        # Check gradient norm decay
        final_grad_norm = history['grad_norm'][-1]
        initial_grad_norm = history['grad_norm'][0]
        
        print(f"  Initial gradient norm: {initial_grad_norm:.2e}")
        print(f"  Final gradient norm: {final_grad_norm:.2e}")
        print(f"  Ratio: {final_grad_norm / (initial_grad_norm + 1e-10):.2e}")
        
        all_results[f'vanishing_gradients_depth_{depth}'] = history
    
    # Experiment 2.3: Exploding gradients
    print("\n" + "-"*70)
    print("Experiment 2.3: Exploding Gradient Detection")
    print("-"*70)
    
    weight_scales = [1.0, 5.0, 10.0]
    
    for scale in weight_scales:
        print(f"\nWeight scale: {scale}")
        model = PathologicalModel3(weight_scale=scale)
        history = monitor_training_with_numgeom(model, (train_x, train_y), n_steps=30, device=device)
        
        # Check if error bounds and warnings increase
        max_error = np.max(history['error_bound'])
        max_grad_error = np.max(history['max_grad_error'])
        
        print(f"  Max error bound: {max_error:.2e}")
        print(f"  Max gradient error: {max_grad_error:.2e}")
        
        all_results[f'exploding_gradients_scale_{scale}'] = history
    
    return all_results


def visualize_results(results, output_dir='../data'):
    """
    Create visualizations of instability detection
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Saturating softmax
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    for i, factor in enumerate([1.0, 10.0, 100.0]):
        key = f'saturating_softmax_factor_{factor}'
        if key not in results:
            continue
        
        history = results[key]
        
        ax1 = axes[0, 0]
        ax1.semilogy(history['step'], history['error_bound'], label=f'Factor {factor}')
        ax1.set_xlabel('Training Step')
        ax1.set_ylabel('Error Bound')
        ax1.set_title('Error Bounds: Saturating Softmax')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2 = axes[0, 1]
        ax2.plot(history['step'], history['n_warnings'], label=f'Factor {factor}')
        ax2.set_xlabel('Training Step')
        ax2.set_ylabel('Number of Warnings')
        ax2.set_title('Warnings: Saturating Softmax')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    # Plot 2: Vanishing gradients
    for depth in [3, 5, 10]:
        key = f'vanishing_gradients_depth_{depth}'
        if key not in results:
            continue
        
        history = results[key]
        
        ax3 = axes[1, 0]
        ax3.semilogy(history['step'], history['grad_norm'], label=f'Depth {depth}')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norms: Vanishing Gradients')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 3: Exploding gradients
    for scale in [1.0, 5.0, 10.0]:
        key = f'exploding_gradients_scale_{scale}'
        if key not in results:
            continue
        
        history = results[key]
        
        ax4 = axes[1, 1]
        ax4.semilogy(history['step'], history['max_grad_error'], label=f'Scale {scale}')
        ax4.set_xlabel('Training Step')
        ax4.set_ylabel('Max Gradient Error')
        ax4.set_title('Gradient Errors: Exploding Gradients')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/instability_detection.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/instability_detection.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir}/instability_detection.pdf")
    
    plt.close()


def save_results(results, output_dir='../data'):
    """Save experimental results"""
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Convert to JSON-serializable format
    json_results = {}
    for key, history in results.items():
        json_results[key] = {
            'step': history['step'],
            'loss': history['loss'],
            'error_bound': history['error_bound'],
            'n_warnings': history['n_warnings'],
            'max_grad_error': history['max_grad_error'],
            'grad_norm': history['grad_norm']
        }
    
    with open(f'{output_dir}/instability_detection_results.json', 'w') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"Results saved to {output_dir}/instability_detection_results.json")


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_pathological_experiments(device=device)
    save_results(results)
    visualize_results(results)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
