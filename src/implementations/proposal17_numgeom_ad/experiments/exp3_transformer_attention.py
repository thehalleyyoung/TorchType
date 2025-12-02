"""
Experiment 3: Transformer Attention Precision Analysis

This experiment analyzes attention mechanism precision requirements
using NumGeom-AD on a tiny transformer. We show:
1. Which attention operations need high precision
2. How error bounds vary with sequence length
3. Practical guidance for mixed-precision attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from numgeom_ad import NumGeomAD
from error_functional import get_primitive_error_functional


class TinyAttention(nn.Module):
    """Minimal attention mechanism for analysis"""
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        B, T, C = x.shape
        
        # Compute Q, K, V
        Q = self.W_q(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = (Q @ K.transpose(-2, -1)) / np.sqrt(self.d_k)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = attn_weights @ V
        
        # Concatenate and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(B, T, C)
        output = self.W_o(attn_output)
        
        return output, attn_weights


class TinyTransformer(nn.Module):
    """Minimal transformer for analysis"""
    def __init__(self, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.d_model = d_model
        
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attn': TinyAttention(d_model, n_heads),
                'norm1': nn.LayerNorm(d_model),
                'ff': nn.Sequential(
                    nn.Linear(d_model, d_model * 4),
                    nn.ReLU(),
                    nn.Linear(d_model * 4, d_model)
                ),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(n_layers)
        ])
    
    def forward(self, x):
        for layer in self.layers:
            # Attention block
            attn_out, _ = layer['attn'](x)
            x = layer['norm1'](x + attn_out)
            
            # Feed-forward block
            ff_out = layer['ff'](x)
            x = layer['norm2'](x + ff_out)
        
        return x


def analyze_attention_precision(seq_lengths, d_model=64, device='cpu'):
    """
    Analyze how attention error bounds vary with sequence length
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3.1: Attention Error vs Sequence Length")
    print("="*70)
    
    torch.manual_seed(42)
    
    results = {
        'seq_lengths': seq_lengths,
        'error_bounds': [],
        'qk_lipschitz': [],
        'softmax_intrinsic': [],
        'total_warnings': []
    }
    
    for seq_len in seq_lengths:
        print(f"\nSequence length: {seq_len}")
        
        # Create attention module
        attn = TinyAttention(d_model=d_model).to(device)
        
        # Random input
        x = torch.randn(2, seq_len, d_model).to(device)
        
        # Wrap with NumGeom-AD
        numgeom = NumGeomAD(attn, dtype=torch.float32, device=str(device))
        
        # Forward pass
        output, error_bound = numgeom.forward_with_error(x)
        
        # Get breakdown
        breakdown = numgeom.get_error_breakdown()
        warnings = numgeom.check_stability(threshold=1e-5)
        
        print(f"  Error bound: {error_bound:.2e}")
        print(f"  Warnings: {len(warnings)}")
        
        # Extract specific components
        qk_error = sum(v for k, v in breakdown.items() if 'W_q' in k or 'W_k' in k)
        softmax_error = sum(v for k, v in breakdown.items() if 'softmax' in k.lower())
        
        results['error_bounds'].append(error_bound)
        results['qk_lipschitz'].append(qk_error)
        results['softmax_intrinsic'].append(softmax_error)
        results['total_warnings'].append(len(warnings))
        
        numgeom.remove_hooks()
    
    return results


def analyze_attention_components(device='cpu'):
    """
    Analyze individual attention components
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3.2: Per-Component Error Analysis")
    print("="*70)
    
    torch.manual_seed(42)
    d_model = 64
    seq_len = 32
    
    # Create inputs
    x = torch.randn(4, seq_len, d_model).to(device)
    
    # Analyze each operation separately
    eps_machine = 5.96e-8
    
    components = {}
    
    # 1. QK^T computation
    print("\n1. Query-Key dot product:")
    Q = torch.randn(4, 4, seq_len, 16).to(device)  # (B, H, T, d_k)
    K = torch.randn(4, 4, seq_len, 16).to(device)
    
    scores = Q @ K.transpose(-2, -1)
    qk_error = get_primitive_error_functional('matmul', [Q, K.transpose(-2, -1)], eps_machine)
    
    print(f"  Lipschitz: {qk_error.lipschitz:.2f}")
    print(f"  Intrinsic error: {qk_error.intrinsic:.2e}")
    
    components['qk_matmul'] = {
        'lipschitz': qk_error.lipschitz,
        'intrinsic': qk_error.intrinsic
    }
    
    # 2. Scaling
    print("\n2. Scaling by 1/sqrt(d_k):")
    d_k = 16
    scaled_scores = scores / np.sqrt(d_k)
    
    # Scaling modifies error functional
    scale_factor = 1.0 / np.sqrt(d_k)
    scaled_error = qk_error.scale(scale_factor)
    
    print(f"  Lipschitz after scaling: {scaled_error.lipschitz:.2f}")
    print(f"  Intrinsic error after scaling: {scaled_error.intrinsic:.2e}")
    
    components['scaling'] = {
        'factor': scale_factor,
        'lipschitz': scaled_error.lipschitz,
        'intrinsic': scaled_error.intrinsic
    }
    
    # 3. Softmax
    print("\n3. Softmax:")
    softmax_error = get_primitive_error_functional(
        'softmax', [scaled_scores], eps_machine, dim=-1
    )
    
    print(f"  Lipschitz: {softmax_error.lipschitz:.2f}")
    print(f"  Intrinsic error: {softmax_error.intrinsic:.2e}")
    print(f"  Note: Intrinsic error grows with seq_len and logit differences")
    
    components['softmax'] = {
        'lipschitz': softmax_error.lipschitz,
        'intrinsic': softmax_error.intrinsic
    }
    
    # 4. Attention-Value matmul
    print("\n4. Attention-Value multiplication:")
    attn_weights = F.softmax(scaled_scores, dim=-1)
    V = torch.randn(4, 4, seq_len, 16).to(device)
    
    av_error = get_primitive_error_functional('matmul', [attn_weights, V], eps_machine)
    
    print(f"  Lipschitz: {av_error.lipschitz:.2f}")
    print(f"  Intrinsic error: {av_error.intrinsic:.2e}")
    
    components['av_matmul'] = {
        'lipschitz': av_error.lipschitz,
        'intrinsic': av_error.intrinsic
    }
    
    # Compose all components
    print("\n5. Total attention error (composed):")
    total_error = av_error.compose(softmax_error).compose(scaled_error)
    
    print(f"  Total Lipschitz: {total_error.lipschitz:.2f}")
    print(f"  Total intrinsic error: {total_error.intrinsic:.2e}")
    print(f"  Total error at ε_mach: {total_error(eps_machine):.2e}")
    
    components['total_composed'] = {
        'lipschitz': total_error.lipschitz,
        'intrinsic': total_error.intrinsic,
        'error_at_eps_machine': total_error(eps_machine)
    }
    
    return components


def compare_attention_precisions(device='cpu'):
    """
    Compare attention in different precisions
    """
    print("\n" + "="*70)
    print("EXPERIMENT 3.3: Attention Precision Comparison")
    print("="*70)
    
    torch.manual_seed(42)
    
    d_model = 64
    seq_len = 64
    
    # Create model
    model_fp32 = TinyTransformer(d_model=d_model, n_heads=4, n_layers=2)
    model_fp16 = TinyTransformer(d_model=d_model, n_heads=4, n_layers=2).to(torch.float16)
    
    # Sync weights
    with torch.no_grad():
        for p32, p16 in zip(model_fp32.parameters(), model_fp16.parameters()):
            p16.data = p32.data.to(torch.float16)
    
    # Input
    x_fp32 = torch.randn(4, seq_len, d_model)
    x_fp16 = x_fp32.to(torch.float16)
    
    # Analyze fp32
    print("\nFloat32 analysis:")
    numgeom_fp32 = NumGeomAD(model_fp32, dtype=torch.float32)
    out_fp32, error_fp32 = numgeom_fp32.forward_with_error(x_fp32)
    warnings_fp32 = numgeom_fp32.check_stability(threshold=1e-5)
    
    print(f"  Error bound: {error_fp32:.2e}")
    print(f"  Warnings: {len(warnings_fp32)}")
    
    numgeom_fp32.remove_hooks()
    
    # Analyze fp16
    print("\nFloat16 analysis:")
    numgeom_fp16 = NumGeomAD(model_fp16, dtype=torch.float16)
    out_fp16, error_fp16 = numgeom_fp16.forward_with_error(x_fp16)
    warnings_fp16 = numgeom_fp16.check_stability(threshold=1e-3)  # Higher threshold for fp16
    
    print(f"  Error bound: {error_fp16:.2e}")
    print(f"  Warnings: {len(warnings_fp16)}")
    
    numgeom_fp16.remove_hooks()
    
    # Compare outputs
    output_diff = (out_fp32.to(torch.float16) - out_fp16).abs().max().item()
    print(f"\nOutput difference (fp32 vs fp16): {output_diff:.2e}")
    
    results = {
        'fp32': {
            'error_bound': error_fp32,
            'n_warnings': len(warnings_fp32)
        },
        'fp16': {
            'error_bound': error_fp16,
            'n_warnings': len(warnings_fp16)
        },
        'output_diff': output_diff
    }
    
    return results


def visualize_attention_results(seq_results, component_results, output_dir='../data'):
    """
    Create visualizations for attention analysis
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Error vs sequence length
    ax1 = axes[0, 0]
    ax1.semilogy(seq_results['seq_lengths'], seq_results['error_bounds'], 'o-', linewidth=2)
    ax1.set_xlabel('Sequence Length')
    ax1.set_ylabel('Error Bound')
    ax1.set_title('Attention Error Bound vs Sequence Length')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Warnings vs sequence length
    ax2 = axes[0, 1]
    ax2.plot(seq_results['seq_lengths'], seq_results['total_warnings'], 's-', linewidth=2, color='red')
    ax2.set_xlabel('Sequence Length')
    ax2.set_ylabel('Number of Warnings')
    ax2.set_title('Stability Warnings vs Sequence Length')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Component contributions
    ax3 = axes[1, 0]
    components = ['qk_matmul', 'scaling', 'softmax', 'av_matmul', 'total_composed']
    lipschitz_vals = [component_results[c]['lipschitz'] for c in components if c != 'total_composed']
    lipschitz_vals.append(component_results['total_composed']['lipschitz'])
    
    ax3.bar(range(len(components)), lipschitz_vals, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax3.set_xticks(range(len(components)))
    ax3.set_xticklabels(['QK', 'Scale', 'Softmax', 'AV', 'Total'], rotation=45)
    ax3.set_ylabel('Lipschitz Constant')
    ax3.set_title('Attention Component Lipschitz Constants')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Intrinsic errors
    ax4 = axes[1, 1]
    intrinsic_vals = [component_results[c]['intrinsic'] for c in components if c != 'total_composed']
    intrinsic_vals.append(component_results['total_composed']['intrinsic'])
    
    ax4.bar(range(len(components)), intrinsic_vals, color=['blue', 'green', 'orange', 'red', 'purple'])
    ax4.set_xticks(range(len(components)))
    ax4.set_xticklabels(['QK', 'Scale', 'Softmax', 'AV', 'Total'], rotation=45)
    ax4.set_ylabel('Intrinsic Error')
    ax4.set_yscale('log')
    ax4.set_title('Attention Component Intrinsic Errors')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/attention_analysis.pdf', dpi=150, bbox_inches='tight')
    plt.savefig(f'{output_dir}/attention_analysis.png', dpi=150, bbox_inches='tight')
    print(f"\nVisualization saved to {output_dir}/attention_analysis.pdf")
    
    plt.close()


def run_transformer_experiments(device='cpu'):
    """Run all transformer experiments"""
    print("\n" + "="*70)
    print("EXPERIMENT 3: TRANSFORMER ATTENTION PRECISION ANALYSIS")
    print("="*70)
    
    # Experiment 3.1: Error vs sequence length
    seq_lengths = [8, 16, 32, 64, 128]
    seq_results = analyze_attention_precision(seq_lengths, device=device)
    
    # Experiment 3.2: Component analysis
    component_results = analyze_attention_components(device=device)
    
    # Experiment 3.3: Precision comparison
    precision_results = compare_attention_precisions(device=device)
    
    # Visualize
    visualize_attention_results(seq_results, component_results)
    
    # Save all results
    results = {
        'sequence_length_analysis': seq_results,
        'component_analysis': component_results,
        'precision_comparison': precision_results
    }
    
    Path('../data').mkdir(parents=True, exist_ok=True)
    with open('../data/transformer_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to ../data/transformer_attention_results.json")
    
    return results


if __name__ == '__main__':
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_transformer_experiments(device=device)
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)
