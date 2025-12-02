#!/usr/bin/env python3
"""
Script to run the getproposal prompt for proposals 1 through 10 sequentially.
"""

import os

proposal_ranking = [
    25,  # Fairness - Hot topic, societal impact, simple experiments, novel
    17,  # Certified Autodiff - Fundamental tool, error functionals are practical
    30,  # Numerical Complexity Classes - Novel taxonomy, compelling P/NP framing
    22,  # MCMC/Sampling - Foundational, ICML loves sampling papers
    19,  # Scientific ML - Neural ODEs/PINNs are hot, guarantees are needed
    21,  # RL/Bellman - Core ICML topic, clean theory, practical implications
    15,  # Training Dynamics - Homotopy geometry of training is novel & visual
    28,  # Probabilistic Inference - VI/MCMC staples with fresh numerical angle
    27,  # TDA - Growing interest, certified persistence is novel
    24,  # Interpretability - Hot topic, numerical saliency bounds are new
    18,  # Quantization-Aware Training - Practical, industry-relevant
    11,  # Bit Allocation - Curvature-guided quantization is novel & practical
    23,  # NumGeomCompile - Verified ML systems, niche but ICML-friendly
    16,  # Optimizer Equivalence - Interesting but may seem esoteric
    26,  # HPO - Well-studied but precision-aware acquisition is novel
    29,  # Dataset Distillation - Trendy topic, numerical fragility is fresh
    12,  # Stability Algebra - Beautiful math but may seem abstract
    14,  # Info-Precision Tradeoffs - Niche intersection
    13,  # Sheaves/Cohomology - Very abstract, hard sell at applied venue
    20,  # Benchmark Suite - Benchmarks alone are hard to publish at ICML
]

PROMPT_PATH = ".github/prompts/getproposal.prompt.md"
prompt = open(PROMPT_PATH, 'r').read().replace('"', '').replace('"', '')
def main():
    for j in range(5):
        for n in proposal_ranking[:5]:
            print(f"\n{'='*60}")
            print(f"Running proposal {n}")
            print(f"{'='*60}\n")
            
            cmd = f'copilot -p "{prompt.replace("{n}", str(n))}" --allow-all-tools'
            print(f"Executing: {cmd}")
            os.system(cmd)
            print(f"\n{'='*60}")
            print(f"Completed proposal {n}")
            print(f"{'='*60}\n")

if __name__ == "__main__":
    main()
