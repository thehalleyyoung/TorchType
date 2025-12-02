#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv('training_curvature.csv')

# Get unique layers
layers = df['layer'].unique()

# Create subplots
fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4*len(layers)), sharex=True)
if len(layers) == 1:
    axes = [axes]

for ax, layer in zip(axes, layers):
    layer_data = df[df['layer'] == layer]
    
    # Plot curvature
    ax.semilogy(layer_data['step'], layer_data['kappa_curv'], 
                label=f'{layer}', linewidth=2)
    
    # Add threshold lines
    ax.axhline(1e6, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax.axhline(1e9, color='red', linestyle='--', alpha=0.5, label='Danger')
    
    ax.set_ylabel('Curvature κ')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Training Step')
plt.suptitle('Curvature Evolution During Training')
plt.tight_layout()
plt.savefig('curvature_timeseries.png', dpi=150)
print('Saved curvature_timeseries.png')

# Create correlation plot if we have gradient norms
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Total curvature
total_curv = df.groupby('step')['kappa_curv'].sum()
ax1.semilogy(total_curv.index, total_curv.values, linewidth=2)
ax1.set_ylabel('Total Curvature')
ax1.grid(True, alpha=0.3)
ax1.set_title('Total Curvature Across All Layers')

# Gradient norm
total_grad = df.groupby('step')['gradient_norm'].sum()
ax2.semilogy(total_grad.index, total_grad.values, linewidth=2, color='green')
ax2.set_ylabel('Total Gradient Norm')
ax2.set_xlabel('Training Step')
ax2.grid(True, alpha=0.3)
ax2.set_title('Total Gradient Norm')

plt.tight_layout()
plt.savefig('curvature_gradient_correlation.png', dpi=150)
print('Saved curvature_gradient_correlation.png')

plt.show()
