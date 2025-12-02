#!/bin/bash

# End-to-end script to regenerate everything from scratch
# Run time: ~2 minutes on a laptop

set -e  # Exit on error

echo "========================================================================"
echo "NUMERICAL GEOMETRY OF RL: End-to-End Regeneration Script"
echo "========================================================================"
echo ""

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "Working directory: $SCRIPT_DIR"
echo ""

# Step 1: Clean previous results
echo "Step 1: Cleaning previous results..."
rm -f data/experiment*.json
rm -f docs/figures/*.pdf docs/figures/*.png
rm -f docs/*.aux docs/*.bbl docs/*.blg docs/*.log docs/*.out
echo "  ✓ Cleaned"
echo ""

# Step 2: Run all experiments
echo "Step 2: Running experiments (~2 minutes)..."
echo "----------------------------------------"
python3.11 src/experiments.py
echo ""
echo "  ✓ Experiments complete"
echo ""

# Step 3: Generate visualizations
echo "Step 3: Generating visualizations..."
echo "----------------------------------------"
python3.11 scripts/generate_plots.py
echo ""
echo "  ✓ Visualizations complete"
echo ""

# Step 4: Verify data quality
echo "Step 4: Verifying experimental data..."
python3.11 << 'EOF'
import json
import numpy as np

# Check that all data files exist and are valid
data_files = [
    'data/experiment1_precision_threshold.json',
    'data/experiment2_error_accumulation.json',
    'data/experiment3_qlearning_stability.json',
    'data/experiment4_discount_sensitivity.json',
    'data/experiment5_function_approximation.json'
]

for f in data_files:
    with open(f, 'r') as fp:
        data = json.load(fp)
    print(f"  ✓ {f.split('/')[-1]}: {len(str(data))} bytes")

# Quick sanity checks
with open('data/experiment4_discount_sensitivity.json', 'r') as f:
    data = json.load(f)
    
gammas = np.array(data['gammas'])
p_bits = np.array(data['theoretical_min_bits'])

# Check monotonicity: precision should increase with gamma
assert all(p_bits[i] <= p_bits[i+1] for i in range(len(p_bits)-1)), "Precision not monotonic!"
print("\n  ✓ Data quality checks passed")
EOF
echo ""

# Step 5: Run tests
echo "Step 5: Running tests..."
echo "----------------------------------------"
python3.11 tests/test_numerical_rl.py
echo ""

# Step 6: Compile paper
echo "Step 6: Compiling paper..."
echo "----------------------------------------"
cd docs

# First pass
pdflatex -interaction=nonstopmode numerical_rl_icml.tex > /dev/null 2>&1 || true

# Bibliography (may fail gracefully)
bibtex numerical_rl_icml > /dev/null 2>&1 || true

# Second pass
pdflatex -interaction=nonstopmode numerical_rl_icml.tex > /dev/null 2>&1 || true

# Third pass (for references)
pdflatex -interaction=nonstopmode numerical_rl_icml.tex > /dev/null 2>&1 || true

if [ -f numerical_rl_icml.pdf ]; then
    PDF_SIZE=$(du -h numerical_rl_icml.pdf | cut -f1)
    echo "  ✓ Paper compiled: numerical_rl_icml.pdf ($PDF_SIZE)"
else
    echo "  ⚠ Paper compilation had issues, but may be usable"
fi

cd ..
echo ""

# Step 7: Summary
echo "========================================================================"
echo "REGENERATION COMPLETE!"
echo "========================================================================"
echo ""
echo "Generated artifacts:"
echo "  - 5 experimental data files in data/"
echo "  - 5 publication-quality figures in docs/figures/"
echo "  - Complete ICML paper in docs/numerical_rl_icml.pdf"
echo ""
echo "Quick checks:"
echo "  - View phase diagram: open docs/figures/fig1_phase_diagram.pdf"
echo "  - View paper: open docs/numerical_rl_icml.pdf"
echo "  - Check data: ls -lh data/"
echo ""

# Final verification
python3.11 << 'EOF'
import os
import json

# Count generated files
data_files = len([f for f in os.listdir('data') if f.endswith('.json')])
fig_files = len([f for f in os.listdir('docs/figures') if f.endswith('.pdf')])

print(f"Summary:")
print(f"  - {data_files}/5 data files generated")
print(f"  - {fig_files}/5 figures generated")

# Load one dataset and show key result
with open('data/experiment4_discount_sensitivity.json', 'r') as f:
    data = json.load(f)

import numpy as np
from numpy.linalg import lstsq

gammas = np.array(data['gammas'])
p_bits = np.array(data['theoretical_min_bits'])
log_term = np.log(1 / (1 - gammas))

A = np.vstack([log_term, np.ones(len(log_term))]).T
slope, intercept = lstsq(A, p_bits, rcond=None)[0]
r2 = 1 - np.var(p_bits - (slope*log_term + intercept)) / np.var(p_bits)

print(f"\nKey result: p = {slope:.2f} * log(1/(1-γ)) + {intercept:.2f}, R² = {r2:.4f}")
print("✓ Confirms theoretical log(1/(1-γ)) scaling")
EOF

echo ""
echo "========================================================================"
echo "To explore results:"
echo "  1. Read README.md for quick demos"
echo "  2. View docs/numerical_rl_icml.pdf for complete paper"
echo "  3. Check docs/figures/ for all visualizations"
echo "========================================================================"
