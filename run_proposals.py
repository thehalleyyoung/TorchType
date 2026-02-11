    #!/usr/bin/env python3
    """
    Script to run the getproposal prompt for proposals 1 through 10 sequentially.
    """

    import os

    proposal_ranking = [
        # === TIER 1: Near-Guaranteed Impressive Results ===
        # These have clear baselines, large gaps, and easy laptop reproduction
        
        23,  # NumGeomCompile: NaN rate 8.3% → 0.1%. HUGE gap, trivially reproducible,
             # baseline is literally "run PyTorch cross-entropy at float16". Anyone
             # can verify this fails and our rewrites fix it. Clear win.

        36,  # MuJoCo Sim: 34% → 91% success rate at float16. 
             # MuJoCo is industry-standard. Sim-to-sim failure is shocking and
             # easy to demonstrate. Gap is enormous (2.7x improvement).
        
        20,  # NumGeom-Bench Task 7: Only 23% of float16 runs succeed on Lorenz.
             # Ground truth exists (we know the attractor). Exposes a real problem
             # that practitioners hit. Easy to reproduce with torchdiffeq.
        
        # === TIER 2: Strong Results with Clear Baselines ===
        # Good benchmarks, solid improvements, some setup required
        
        33,  # NAS-Bench-201: ρ=0.89 vs ρ=0.52. Correlation is unambiguous,
             # NAS-Bench has precomputed ground truth. Improvement is 1.7x.
             # Just need to download the benchmark and run predictions.
        
        22,  # Banana distribution KL: 0.31 vs 0.02. Sampling is a classic benchmark,
             # banana is standard. 15x improvement is massive. Easy to run.
        
        19,  # Lorenz trajectory: 3.2x vs 48x theoretical bound. Chaotic systems
             # are dramatic - trajectory divergence is visually compelling.
             # Lorenz is famous; the 15x improvement is striking.
        
        37,  # Task arithmetic: 78% vs 52% success at float16. Task arithmetic
             # paper is well-cited, benchmark is established. 1.5x improvement
             # on a trendy topic (model merging). Very relevant to practitioners.
        
        31,  # Cora GNN: 81.2% vs 78.4% at float16. Cora is THE GNN benchmark.
             # 2.8% recovery is solid. Easy to run with PyTorch Geometric.
             # Baseline (naive float16) is exactly what practitioners would try.
        
        # === TIER 3: Good Results, Requires Careful Setup ===
        # Improvements are real but require more careful experimental design
        
        11,  # CIFAR-10 4-bit: 89.2% vs HAWQ 88.4%. Solid improvement but
             # quantization is crowded field. Need to match HAWQ's exact setup.
             # Still, beating a published method is impressive.
        
        39,  # NQ Recall@10: 78.4% vs 71.2%. Natural Questions is major benchmark.
             # 7.2% improvement is substantial. RAG is hot topic.
             # Requires DPR setup which is a bit heavy but doable.
        
        34,  # STL-10 SimCLR: 71.3% vs 66.8%. Contrastive learning benchmark.
             # 4.5% improvement is significant. SimCLR is well-known.
             # Temperature stability story is compelling.
        
        38,  # CIFAR-100 KD: 76.2% vs 73.8%. Knowledge distillation is mature.
             # 2.4% improvement is good. Standard benchmark with established
             # baselines from Hinton et al. Reproducible.
        
        40,  # Split CIFAR-100 EWC: 62.3% vs 54.7%. Continual learning benchmark.
             # 7.6% improvement is large. EWC is the canonical method.
             # Split CIFAR is standard. Fisher noise story is novel.
        
        # === TIER 4: Solid but Less Dramatic ===
        # Real improvements but smaller gaps or harder to verify
        
        35,  # ETTh1 MSE: 0.412 vs 0.523. Time series benchmark is good.
             # 21% improvement is solid. But MSE differences are less
             # "wow" than accuracy jumps or success/failure flips.
        
        32,  # LEAF-FEMNIST: 89 vs 142 rounds. Federated learning is important.
             # 37% faster convergence is good. But "convergence speed" is
             # less impressive than "accuracy" to most readers.
        
        17,  # Gradient bounds 7.3x vs 43x. Tightness is measurable but
             # "7.3x" requires explanation. Less intuitive than accuracy.
             # Still, beating interval arithmetic is meaningful to experts.
        
        24,  # MNIST attribution stability: 73% vs 94%. Interpretability metric.
             # The gap (21%) is real but "attribution stability" is niche.
             # Need to convince readers this matters.
        
        18,  # Training bit-ops: 42% reduction. Efficiency metric is good but
             # "bit-operations" is unusual. Most readers think in FLOPs or
             # wall-clock time. Solid but needs translation.
        
        # === TIER 5: Meaningful but Niche ===
        # Real contributions but smaller audience or harder to evaluate
        
        12,  # UCI Adult float16: 85.3% vs 82.1%. UCI is old-school benchmark.
             # 3.2% improvement is real but UCI isn't exciting to ML venues.
             # Pipeline ordering is useful but not flashy.
        
        27,  # Ripser H1 features: 67% preserved. TDA is niche field.
             # The metric (feature preservation) needs explanation.
             # Important to TDA people but small audience.
        
        25,  # Adult DPG sign flips: 12% of thresholds. Fairness is important
             # but "12% of thresholds flip" is confusing metric. The story
             # (fairness conclusions change with precision) is compelling
             # but needs careful presentation.
        
        28,  # German Credit posterior: 8.3% vs 0.4% error. Bayesian inference
             # audience is smaller. German Credit is old. But "coefficient
             # sign flips" is dramatic for practitioners.
        
        29,  # Dataset distillation variance: 1.2% vs 7.8%. Distillation is
             # trendy but variance (stability) is secondary metric. Mean
             # accuracy is what people care about. Trade-off story is nuanced.
        
        # === TIER 6: Hardest to Demonstrate Impressively ===
        # Valid contributions but metric requires most explanation
        
        14,  # C_num correlation r=0.94. Information theory metric needs
             # significant explanation. Beating CKA (r=0.71) is meaningful
             # but "numerical entropy" is unfamiliar to most readers.
        
        30,  # NC classification 92% (11/12 tasks). Meta-task about classifying
             # tasks is confusing. "We correctly predict precision sensitivity"
             # is second-order. Valid but hard to make impressive.
        
        26,  # HPO regret 31% reduction. HPO-Bench is specialized. "Regret"
             # requires definition. Improvement is solid but presentation
             # challenge is significant.
        
        15,  # XOR variance 0.31 vs 0.02. Toy problem (XOR) feels too simple.
             # The homotopy class insight is deep but XOR is undergraduate
             # homework. Need to show this scales to real problems.
        
        16,  # Optimizer trajectory divergence: 0.12 vs 0.47. The insight
             # (optimizers converge at float16) is surprising but "trajectory
             # distance" is not standard metric. Hard to contextualize.


        21,  # RL Precision: CartPole diverges at int8 (0 episodes vs 500+).
             # Binary success/failure is maximally clear. CartPole is THE standard
             # RL benchmark. Theoretical prediction matches exactly. Very impressive.
        
        13,  # SheafCheck: 94% bug detection (17/18) vs PyTorch's 6/18.
             # Clear ground truth (injected bugs), huge improvement (3x).
             # Easy to verify: inject bugs, run tool, count detections.
        
    ]


    PROMPT_PATH = ".github/prompts/getproposal.prompt.md"
    prompt = open(PROMPT_PATH, 'r').read().replace('"', '').replace('"', '')
    def main():
        while True:
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
# a3 CI enabled
