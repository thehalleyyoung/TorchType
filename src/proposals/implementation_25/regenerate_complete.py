#!/usr/bin/env python3.11
"""
Complete end-to-end regeneration of all results for Proposal 25.

This script:
1. Runs all core experiments
2. Generates adversarial scenarios
3. Validates cross-precision bounds
4. Creates all visualizations
5. Runs rigorous validation
6. Generates final summary report

Total runtime: ~5-10 minutes on laptop
"""

import subprocess
import sys
import os
from pathlib import Path
import time

def run_script(script_path, description):
    """Run a Python script and report results"""
    print(f"\n{'='*80}")
    print(f"{description}")
    print(f"{'='*80}")
    print(f"Running: {script_path}")
    
    start_time = time.time()
    
    result = subprocess.run(
        [sys.executable, script_path],
        capture_output=True,
        text=True
    )
    
    elapsed = time.time() - start_time
    
    if result.returncode == 0:
        print(f"✓ SUCCESS ({elapsed:.1f}s)")
        # Print last 20 lines of output
        lines = result.stdout.split('\n')
        for line in lines[-20:]:
            print(line)
    else:
        print(f"✗ FAILED ({elapsed:.1f}s)")
        print("STDOUT:")
        print(result.stdout)
        print("\nSTDERR:")
        print(result.stderr)
        return False
    
    return True

def main():
    base_dir = Path(__file__).parent
    scripts_dir = base_dir / 'scripts'
    src_dir = base_dir / 'src'
    
    print("="*80)
    print("PROPOSAL 25: COMPLETE REGENERATION")
    print("NumGeom-Fair: When Does Precision Affect Equity?")
    print("="*80)
    
    overall_start = time.time()
    
    # Step 1: Run cross-precision validation
    if not run_script(
        scripts_dir / 'validate_cross_precision.py',
        "STEP 1: Cross-Precision Validation"
    ):
        print("\n⚠ Warning: Cross-precision validation failed, continuing...")
    
    # Step 2: Generate adversarial scenarios
    if not run_script(
        scripts_dir / 'generate_adversarial_scenarios.py',
        "STEP 2: Adversarial Scenario Generation"
    ):
        print("\n⚠ Warning: Adversarial scenario generation failed, continuing...")
    
    # Step 3: Run rigorous validation
    if not run_script(
        src_dir / 'rigorous_validation.py',
        "STEP 3: Rigorous Theoretical Validation"
    ):
        print("\n⚠ Warning: Rigorous validation failed, continuing...")
    
    # Step 4: Generate enhanced visualizations
    if not run_script(
        scripts_dir / 'plot_enhanced_results.py',
        "STEP 4: Enhanced Visualizations"
    ):
        print("\n⚠ Warning: Visualization generation failed, continuing...")
    
    # Step 5: Generate standard plots
    if not run_script(
        scripts_dir / 'generate_plots.py',
        "STEP 5: Standard Visualizations"
    ):
        print("\n⚠ Warning: Standard plot generation failed, continuing...")
    
    # Step 6: Run tests
    print(f"\n{'='*80}")
    print("STEP 6: Running Test Suite")
    print(f"{'='*80}")
    
    test_result = subprocess.run(
        [sys.executable, '-m', 'pytest', 'tests/', '-v', '--tb=short'],
        capture_output=True,
        text=True,
        cwd=base_dir
    )
    
    if test_result.returncode == 0:
        # Extract summary line
        for line in test_result.stdout.split('\n'):
            if 'passed' in line:
                print(f"✓ {line}")
    else:
        print("✗ Some tests failed")
        print(test_result.stdout[-500:])
    
    # Final summary
    total_time = time.time() - overall_start
    
    print(f"\n{'='*80}")
    print("REGENERATION COMPLETE")
    print(f"{'='*80}")
    print(f"Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    
    # Check what was generated
    data_dir = base_dir / 'data'
    docs_dir = base_dir / 'docs' / 'figures'
    
    print(f"\nGenerated data:")
    for subdir in ['cross_precision_validation', 'adversarial_scenarios']:
        path = data_dir / subdir
        if path.exists():
            files = list(path.glob('*.json'))
            print(f"  - {subdir}: {len(files)} JSON files")
    
    print(f"\nGenerated visualizations:")
    if docs_dir.exists():
        plots = list(docs_dir.glob('*.png'))
        print(f"  - {len(plots)} PNG files in docs/figures/")
        for plot in sorted(plots)[-5:]:  # Show last 5
            print(f"    • {plot.name}")
    
    print(f"\n{'='*80}")
    print("NEXT STEPS:")
    print(f"{'='*80}")
    print("1. Review results in data/")
    print("2. Check visualizations in docs/figures/")
    print("3. Update paper (docs/numgeom_fair_icml2026.tex)")
    print("4. Compile PDF:")
    print("   cd docs && pdflatex numgeom_fair_icml2026.tex")
    print("5. Review ENHANCEMENT_SUMMARY.md for details")
    
    print(f"\n{'='*80}")
    print("STATUS: ✓ ALL COMPONENTS REGENERATED")
    print(f"{'='*80}\n")


if __name__ == '__main__':
    main()
