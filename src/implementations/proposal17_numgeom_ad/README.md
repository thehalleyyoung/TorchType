# NumGeom-AD: Certified Automatic Differentiation with Error Functionals

Implementation of Proposal 17 from the Numerical Geometry framework.

## Overview

NumGeom-AD augments PyTorch's autograd with certified error bounds based on error functional theory. It tracks both gradients and their numerical reliability, enabling:

- ✅ Certified bounds on gradient error
- ✅ Detection of numerical instabilities (saturated softmax, ill-conditioned layers)
- ✅ Precision guidance for mixed-precision training
- ✅ Debugging numerical issues in deep learning

## Quick Start

```bash
# Run all tests
make test

# Run all experiments
make experiments

# Generate plots for paper
make plots

# Compile paper
make paper

# Run everything
make all
```

## Installation

No special installation needed. Uses standard PyTorch:

```bash
pip install torch torchvision numpy matplotlib
```

## Usage Example

```python
import torch
import torch.nn as nn
import sys
sys.path.append('src')

from numgeom_ad import NumGeomAD

# Create model
model = nn.Sequential(
    nn.Linear(10, 20),
    nn.ReLU(),
    nn.Linear(20, 10),
    nn.Softmax(dim=-1)
)

# Wrap with NumGeom-AD
numgeom = NumGeomAD(model, dtype=torch.float32)

# Forward with error tracking
x = torch.randn(8, 10)
output, error_bound = numgeom.forward_with_error(x)

print(f"Output error bound: {error_bound:.2e}")

# Check for stability issues
warnings = numgeom.check_stability(threshold=1e-5)
for w in warnings:
    print(f"⚠ {w}")

# Analyze gradient errors
loss = output.sum()
loss.backward()
grad_errors = numgeom.analyze_gradient_error(loss)

for layer, error in grad_errors.items():
    print(f"{layer}: gradient error bound = {error:.2e}")

numgeom.remove_hooks()
```

## Directory Structure

```
proposal17_numgeom_ad/
├── src/
│   ├── error_functional.py      # Error functional algebra
│   └── numgeom_ad.py             # PyTorch wrapper
├── tests/
│   └── test_comprehensive.py    # Test suite
├── experiments/
│   ├── exp1_precision_guided.py # Mixed-precision experiments
│   ├── exp2_instability_detection.py
│   └── exp3_transformer_attention.py
├── data/                         # Experimental results
├── scripts/
│   └── generate_plots.py        # Plot generation
├── docs/
│   ├── numgeom_ad_paper_simple.pdf
│   └── figures/
├── Makefile                      # Build automation
└── README.md                     # This file
```

## Experiments

All experiments run on a laptop in <30 minutes:

1. **Bound Tightness**: Compare predicted vs observed errors
2. **Instability Detection**: Detect saturated softmax, vanishing/exploding gradients
3. **Transformer Analysis**: Attention precision requirements

Run individual experiments:

```bash
python3.11 experiments/exp2_instability_detection.py
python3.11 experiments/exp3_transformer_attention.py
```

## Results Summary

- ✓ **Stability Composition Theorem verified** on 100 random compositions
- ✓ **100% detection rate** on injected numerical instabilities
- ✓ **8/8 primitive operations** tested and validated
- ✓ **2.11× average overhead** (acceptable for debugging)
- ✓ **Transformer attention** requires ≥10 bits precision

## Theory

Error functionals Φ(ε) = L·ε + Δ compose via Stability Composition Theorem:

For F = f_n ∘ ... ∘ f_1:

```
Φ_F(ε) = (∏ L_i)·ε + ∑_i Δ_i·(∏_{j>i} L_j)
```

Where:
- L_i: Lipschitz constant of layer i
- Δ_i: Intrinsic roundoff error of layer i

See `docs/numgeom_ad_paper_simple.pdf` for full theoretical development.

## Citation

```bibtex
@article{numgeom-ad,
  title={NumGeom-AD: Certified Automatic Differentiation with Error Functionals},
  author={Anonymous},
  year={2026}
}
```

## License

Part of the Numerical Geometry framework.
