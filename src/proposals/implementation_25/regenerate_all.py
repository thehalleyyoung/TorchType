#!/usr/bin/env python3.11
"""
End-to-end pipeline for Proposal 25: NumGeom-Fair
Regenerates all experiments, plots, and paper from scratch.

Usage:
    python3.11 regenerate_all.py [--quick]
    
    --quick: Run minimal experiments for fast iteration (~30 seconds)
    (default): Run full experiments for publication (~2 minutes)
"""

import sys
import os
import subprocess
import time
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def print_section(title):
    """Print a formatted section header."""
    print("\n" + "="*70)
    print(title.center(70))
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and report timing."""
    print(f"▸ {description}...")
    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            capture_output=True,
            text=True
        )
        elapsed = time.time() - start
        print(f"  ✓ Complete in {elapsed:.1f}s")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start
        print(f"  ✗ Failed after {elapsed:.1f}s")
        print(f"    Error: {e.stderr[:200]}")
        return False

def main():
    """Run the complete pipeline."""
    quick_mode = '--quick' in sys.argv
    
    print_section("NUMGEOM-FAIR: COMPLETE PIPELINE")
    print(f"Mode: {'QUICK' if quick_mode else 'FULL'}")
    print(f"Working directory: {os.getcwd()}")
    
    total_start = time.time()
    
    # Step 1: Run all tests
    print_section("STEP 1: Running Tests")
    success = run_command(
        "python3.11 -m pytest tests/ -v --tb=short",
        "Running test suite (64 tests)"
    )
    if not success:
        print("\n⚠ Tests failed, but continuing...")
    
    # Step 2: Run core experiments
    print_section("STEP 2: Running Core Experiments")
    success = run_command(
        "python3.11 scripts/run_all_experiments.py",
        "Running 5 core experiments (~15s)"
    )
    if not success:
        print("\n✗ Core experiments failed. Aborting.")
        return 1
    
    # Step 3: Run rigorous validation
    print_section("STEP 3: Rigorous Validation")
    success = run_command(
        "python3.11 src/rigorous_validation.py",
        "Validating theoretical bounds empirically"
    )
    if not success:
        print("\n⚠ Validation had issues, but continuing...")
    
    # Step 4: Run practical benefits demo (if not quick mode)
    if not quick_mode:
        print_section("STEP 4: Practical Benefits Demo")
        success = run_command(
            "python3.11 -c 'from src.practical_benefits import PracticalBenefitsDemo; demo = PracticalBenefitsDemo(); demo.run_all_demonstrations(); demo.save_results(\"data/practical_benefits_results.json\")'",
            "Demonstrating memory/speed benefits on MNIST"
        )
        if not success:
            print("\n⚠ Practical benefits demo had issues, but continuing...")
    else:
        print_section("STEP 4: Practical Benefits Demo (SKIPPED in quick mode)")
    
    # Step 5: Generate plots
    print_section("STEP 5: Generating Plots")
    success = run_command(
        "python3.11 scripts/generate_plots.py",
        "Creating all publication-quality figures"
    )
    if not success:
        print("\n⚠ Some plots may be missing, but continuing...")
    
    # Step 6: Copy figures to docs
    print_section("STEP 6: Preparing Paper Assets")
    os.makedirs("docs/figures", exist_ok=True)
    success = run_command(
        "cp implementations/docs/proposal25/figures/*.png docs/figures/ 2>/dev/null || true",
        "Copying figures to docs directory"
    )
    
    # Step 7: Compile paper
    print_section("STEP 7: Compiling ICML Paper")
    os.chdir("docs")
    
    # First pass
    run_command(
        "pdflatex -interaction=nonstopmode numgeom_fair_icml2026.tex > /dev/null",
        "LaTeX compilation (pass 1/3)"
    )
    
    # BibTeX
    run_command(
        "bibtex numgeom_fair_icml2026 2>&1 | grep -v Warning || true",
        "Building bibliography"
    )
    
    # Second pass
    run_command(
        "pdflatex -interaction=nonstopmode numgeom_fair_icml2026.tex > /dev/null",
        "LaTeX compilation (pass 2/3)"
    )
    
    # Final pass
    success = run_command(
        "pdflatex -interaction=nonstopmode numgeom_fair_icml2026.tex > /dev/null",
        "LaTeX compilation (pass 3/3)"
    )
    
    os.chdir("..")
    
    if success and os.path.exists("docs/numgeom_fair_icml2026.pdf"):
        pdf_size = os.path.getsize("docs/numgeom_fair_icml2026.pdf") / 1024
        print(f"\n  ✓ Paper compiled successfully ({pdf_size:.0f} KB)")
        print(f"    Location: docs/numgeom_fair_icml2026.pdf")
    else:
        print("\n  ✗ Paper compilation may have failed")
    
    # Step 8: Generate summary
    print_section("STEP 8: Generating Summary Report")
    
    try:
        # Load validation results
        with open("data/rigorous_validation_results.json") as f:
            validation = json.load(f)
        
        # Load experiment results
        exp1_path = "data/experiment1/experiment1_precision_vs_fairness.json"
        if os.path.exists(exp1_path):
            with open(exp1_path) as f:
                exp1 = json.load(f)
        else:
            exp1 = {}
        
        print("\n📊 Results Summary:")
        print(f"  • Tests: {validation.get('tests_passed', '?')}/4 passed")
        print(f"  • Theory validation: {'✓ PASSED' if validation.get('all_passed') else '✗ FAILED'}")
        
        if 'summary' in exp1:
            print(f"  • Borderline assessments: {exp1['summary'].get('borderline_percentage', '?')}%")
        
        print(f"\n📁 Generated artifacts:")
        print(f"  • Experiment data: data/experiment1-5/")
        print(f"  • Validation results: data/rigorous_validation_results.json")
        print(f"  • Figures: docs/figures/ (7 figures)")
        print(f"  • Paper: docs/numgeom_fair_icml2026.pdf")
        
    except Exception as e:
        print(f"\n⚠ Could not generate full summary: {e}")
    
    # Final timing
    total_elapsed = time.time() - total_start
    
    print_section("PIPELINE COMPLETE")
    print(f"Total time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print()
    print("Next steps:")
    print("  1. Review paper: docs/numgeom_fair_icml2026.pdf")
    print("  2. Check figures: docs/figures/")
    print("  3. Examine data: data/")
    print()
    print("To run experiments with more samples/epochs:")
    print("  • Edit scripts/run_all_experiments.py")
    print("  • Modify n_samples, n_epochs parameters")
    print("  • Re-run: python3.11 regenerate_all.py")
    print()
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
