"""
Visualization scripts for Numerical Geometry of RL experiments.

Generates publication-quality plots from experimental data.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import json
from pathlib import Path
from typing import Dict, Any
import seaborn as sns

# Set publication style
plt.style.use('seaborn-v0_8-paper')
mpl.rcParams['figure.dpi'] = 300
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.size'] = 10
mpl.rcParams['axes.labelsize'] = 10
mpl.rcParams['axes.titlesize'] = 11
mpl.rcParams['xtick.labelsize'] = 9
mpl.rcParams['ytick.labelsize'] = 9
mpl.rcParams['legend.fontsize'] = 9


class VisualizationGenerator:
    """Generate all visualizations for the paper."""
    
    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_data(self, filename: str) -> Dict[str, Any]:
        """Load experimental data from JSON."""
        filepath = self.data_dir / filename
        with open(filepath, 'r') as f:
            return json.load(f)
    
    def plot_precision_discount_phase_diagram(self):
        """
        Figure 1: Precision-Discount Phase Diagram
        
        2D plot: x=precision bits, y=gamma
        Color: converged (green) vs diverged (red)
        Theoretical curve overlay
        """
        print("Generating Figure 1: Precision-Discount Phase Diagram...")
        
        data = self.load_data('experiment1_precision_threshold.json')
        
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Collect data points for each environment
        all_points = []
        
        for env_name in ['gridworld_4x4', 'gridworld_8x8', 'frozenlake']:
            env_data = data[env_name]
            
            for gamma_key, result in env_data.items():
                gamma = result['gamma']
                
                for conv_result in result['convergence_results']:
                    p_bits = conv_result['precision_bits']
                    converged = conv_result['converged'] and conv_result['final_error'] < 0.1
                    
                    all_points.append({
                        'gamma': gamma,
                        'precision': p_bits,
                        'converged': converged,
                        'env': env_name
                    })
        
        # Separate converged and diverged points
        converged_points = [p for p in all_points if p['converged']]
        diverged_points = [p for p in all_points if not p['converged']]
        
        # Plot points
        if diverged_points:
            ax.scatter(
                [p['precision'] for p in diverged_points],
                [p['gamma'] for p in diverged_points],
                c='red', marker='x', s=50, alpha=0.6, label='Diverged'
            )
        
        if converged_points:
            ax.scatter(
                [p['precision'] for p in converged_points],
                [p['gamma'] for p in converged_points],
                c='green', marker='o', s=50, alpha=0.6, label='Converged'
            )
        
        # Theoretical curve: p* = log₂(C / (1-γ))
        # Approximate C from data
        gammas_theory = np.linspace(0.7, 0.99, 100)
        C_approx = 100  # Approximate constant
        p_theory = np.log2(C_approx / (1 - gammas_theory))
        p_theory = np.clip(p_theory, 0, 64)
        
        ax.plot(p_theory, gammas_theory, 'b--', linewidth=2, label='Theoretical threshold')
        
        ax.set_xlabel('Precision (bits)')
        ax.set_ylabel('Discount factor γ')
        ax.set_title('Precision-Discount Phase Diagram')
        ax.legend(loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 70)
        ax.set_ylim(0.65, 1.0)
        
        plt.tight_layout()
        output_file = self.output_dir / 'fig1_phase_diagram.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ Saved to {output_file}")
    
    def plot_error_accumulation(self):
        """
        Figure 2: Error Accumulation Over Iterations
        
        Multiple curves for different precisions
        Theoretical bounds as dashed lines
        """
        print("Generating Figure 2: Error Accumulation...")
        
        data = self.load_data('experiment2_error_accumulation.json')
        
        fig, ax = plt.subplots(figsize=(6, 4))
        
        colors = plt.cm.viridis(np.linspace(0, 0.9, len(data)))
        
        for idx, (key, result) in enumerate(sorted(data.items())):
            p_bits = result['precision_bits']
            observed = result['observed_errors']
            theoretical = result['theoretical_errors']
            
            iterations = range(len(observed))
            
            # Plot observed
            ax.plot(
                iterations, observed,
                color=colors[idx], linewidth=1.5,
                label=f'{p_bits}-bit (observed)'
            )
            
            # Plot theoretical as dashed
            ax.plot(
                iterations, theoretical,
                color=colors[idx], linewidth=1.0, linestyle='--',
                alpha=0.7
            )
        
        ax.set_xlabel('Iteration k')
        ax.set_ylabel('Error ||Ṽₖ - V*||')
        ax.set_title('Error Accumulation in Value Iteration')
        ax.set_yscale('log')
        ax.legend(loc='best', ncol=2, fontsize=8)
        ax.grid(True, alpha=0.3, which='both')
        
        plt.tight_layout()
        output_file = self.output_dir / 'fig2_error_accumulation.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ Saved to {output_file}")
    
    def plot_qlearning_stability(self):
        """
        Figure 3: Q-Learning Stability
        
        Learning curves at different precisions and discount factors
        """
        print("Generating Figure 3: Q-Learning Stability...")
        
        data = self.load_data('experiment3_qlearning_stability.json')
        
        fig, axes = plt.subplots(2, 2, figsize=(8, 6))
        axes = axes.flatten()
        
        gamma_keys = sorted([k for k in data.keys() if k.startswith('gamma_')])[:4]
        
        for idx, gamma_key in enumerate(gamma_keys):
            ax = axes[idx]
            gamma_data = data[gamma_key]
            gamma = float(gamma_key.split('_')[1])
            
            for prec_key in ['8bit', '16bit', '32bit']:
                if prec_key in gamma_data:
                    result = gamma_data[prec_key]
                    returns = result['avg_episode_returns']
                    
                    # Smooth with rolling average
                    window = 50
                    if len(returns) >= window:
                        smoothed = np.convolve(returns, np.ones(window)/window, mode='valid')
                        x = range(window-1, len(returns))
                    else:
                        smoothed = returns
                        x = range(len(returns))
                    
                    ax.plot(x, smoothed, label=prec_key, linewidth=1.5)
            
            ax.set_title(f'γ = {gamma}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Q-Learning Stability at Different Precisions', fontsize=12)
        plt.tight_layout()
        
        output_file = self.output_dir / 'fig3_qlearning_stability.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ Saved to {output_file}")
    
    def plot_discount_sensitivity(self):
        """
        Figure 4: Precision Requirements vs Discount Factor
        
        Shows log(1/(1-γ)) scaling
        """
        print("Generating Figure 4: Discount Sensitivity...")
        
        data = self.load_data('experiment4_discount_sensitivity.json')
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 3.5))
        
        gammas = data['gammas']
        theoretical = data['theoretical_min_bits']
        observed = data['observed_min_bits']
        
        # Plot 1: Precision vs Gamma
        ax1.plot(gammas, theoretical, 'b-', linewidth=2, label='Theoretical')
        ax1.plot(gammas, observed, 'ro', markersize=5, label='Observed')
        ax1.set_xlabel('Discount factor γ')
        ax1.set_ylabel('Minimum precision (bits)')
        ax1.set_title('Precision Requirement vs γ')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Verify log scaling
        # p ~ log(1/(1-γ))
        x_theory = np.log(1 / (1 - np.array(gammas)))
        
        ax2.plot(x_theory, theoretical, 'b-', linewidth=2, label='Theoretical')
        ax2.plot(x_theory, observed, 'ro', markersize=5, label='Observed')
        ax2.set_xlabel('log(1/(1-γ))')
        ax2.set_ylabel('Minimum precision (bits)')
        ax2.set_title('Verifying Logarithmic Scaling')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / 'fig4_discount_sensitivity.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ Saved to {output_file}")
    
    def plot_function_approximation(self):
        """
        Figure 5: DQN Function Approximation
        
        Float32 vs Float16 at different discount factors
        """
        print("Generating Figure 5: Function Approximation...")
        
        data = self.load_data('experiment5_function_approximation.json')
        
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        
        gammas = [0.9, 0.95, 0.99]
        
        for idx, gamma in enumerate(gammas):
            ax = axes[idx]
            
            key_f32 = f'gamma_{gamma}_float32'
            key_f16 = f'gamma_{gamma}_float16'
            
            if key_f32 in data:
                returns_f32 = data[key_f32]['avg_episode_returns']
                window = 20
                if len(returns_f32) >= window:
                    smoothed_f32 = np.convolve(returns_f32, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(returns_f32))
                else:
                    smoothed_f32 = returns_f32
                    x = range(len(returns_f32))
                ax.plot(x, smoothed_f32, 'b-', linewidth=1.5, label='float32')
            
            if key_f16 in data:
                returns_f16 = data[key_f16]['avg_episode_returns']
                window = 20
                if len(returns_f16) >= window:
                    smoothed_f16 = np.convolve(returns_f16, np.ones(window)/window, mode='valid')
                    x = range(window-1, len(returns_f16))
                else:
                    smoothed_f16 = returns_f16
                    x = range(len(returns_f16))
                ax.plot(x, smoothed_f16, 'r-', linewidth=1.5, label='float16')
            
            ax.set_title(f'γ = {gamma}')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Return')
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        
        plt.suptitle('Tiny DQN: Float32 vs Float16', fontsize=12)
        plt.tight_layout()
        
        output_file = self.output_dir / 'fig5_function_approximation.pdf'
        plt.savefig(output_file, bbox_inches='tight')
        plt.savefig(output_file.with_suffix('.png'), bbox_inches='tight', dpi=300)
        plt.close()
        
        print(f"  ✓ Saved to {output_file}")
    
    def generate_all(self):
        """Generate all figures."""
        print("\n" + "=" * 60)
        print("GENERATING ALL VISUALIZATIONS")
        print("=" * 60 + "\n")
        
        self.plot_precision_discount_phase_diagram()
        self.plot_error_accumulation()
        self.plot_qlearning_stability()
        self.plot_discount_sensitivity()
        self.plot_function_approximation()
        
        print("\n" + "=" * 60)
        print(f"ALL FIGURES SAVED TO: {self.output_dir}")
        print("=" * 60)


if __name__ == '__main__':
    script_dir = Path(__file__).parent.parent
    data_dir = script_dir / 'data'
    output_dir = script_dir / 'docs' / 'figures'
    
    viz = VisualizationGenerator(data_dir, output_dir)
    viz.generate_all()
