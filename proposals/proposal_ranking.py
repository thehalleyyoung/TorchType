# Ranking of Proposals 11-30 by ICML Publication Likelihood
# (No proposal 10 exists)
# Criteria: Novel contribution + clear experiments + low compute + strong theory + practical impact

# ACTUAL PROPOSAL TOPICS:
# 11: Curvature-Guided Bit Allocation in Neural Networks
# 12: The Stability Algebra of Learning Pipelines
# 13: Sheaves of Precision for ML Systems (Cohomological Debugging)
# 14: Information-Precision Tradeoffs in Representation Learning
# 15: Numerical Homotopy Paths and Training Dynamics Geometry
# 16: Numerical Equivalence of Training Algorithms
# 17: Certified Autodiff with Error Functionals
# 18: End-to-End Quantization-Aware Training via Numerical Geometry
# 19: Numerically Safe Scientific ML (Neural ODEs, PINNs)
# 20: Numerical Geometry Benchmark Suite
# 21: RL/Bellman Operator Stability
# 22: MCMC/Diffusion Sampling Precision
# 23: NumGeomCompile (Compiler Rewrites)
# 24: Interpretability/Saliency Numerical Bounds
# 25: Fairness Metrics Under Finite Precision
# 26: Precision-Aware HPO/AutoML
# 27: Certified Persistent Homology (TDA)
# 28: Numerically Aware VI/MCMC Inference
# 29: Dataset Distillation Numerical Stability
# 30: Numerical Complexity Classes Taxonomy

# Thinking process:
#
# TOP TIER (High novelty, clear story, laptop-friendly, strong theoretical grounding):
# - 25 (Fairness): Hot topic, clear societal impact, simple experiments, novel angle
# - 17 (Certified Autodiff): Autodiff is fundamental, error propagation is novel & practical
# - 30 (Numerical Complexity Classes): Novel taxonomy, P/NP-style framing is compelling
# - 22 (MCMC/Sampling): Foundational topic ICML loves, connects to sampling theory
#
# STRONG TIER (Good novelty, solid experiments, some competition in the space):
# - 19 (Scientific ML): Neural ODEs/PINNs are hot, numerical guarantees are needed
# - 21 (RL/Bellman): RL is core ICML, Bellman stability is fundamental
# - 15 (Training Dynamics): Homotopy/geometry of training is trendy, novel angle
# - 28 (Probabilistic Inference): VI/MCMC staples with fresh numerical angle
#
# GOOD TIER (Solid contributions, compelling but may face scrutiny):
# - 27 (TDA): Growing interest, certified persistence diagrams is novel
# - 24 (Interpretability): Hot topic, numerical saliency bounds are new
# - 18 (Quantization-Aware Training): Practical, industry-relevant, clear experiments
# - 11 (Bit Allocation): Curvature-guided quantization is novel & practical
#
# MODERATE TIER (Good ideas but may be harder to sell or more niche):
# - 23 (NumGeomCompile): Verified ML systems, niche but ICML-friendly
# - 16 (Optimizer Equivalence): Interesting theory, may seem esoteric
# - 26 (HPO): HPO is well-studied, but precision-aware acquisition is novel
# - 29 (Dataset Distillation): Trendy topic, numerical fragility is fresh angle
#
# LOWER TIER (More theoretical/niche, harder to motivate broad impact):
# - 12 (Stability Algebra): Beautiful math but may seem too abstract
# - 14 (Info-Precision Tradeoffs): Information theory + precision is niche
# - 13 (Sheaves/Cohomology): Very abstract, hard sell for applied venue
# - 20 (Benchmark Suite): Benchmarks are hard to publish alone at ICML

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

# Key factors in ranking:
# 1. TOPIC HEAT: Fairness, Scientific ML, RL, training dynamics are ICML favorites
# 2. NOVELTY: New taxonomies (30), new tools (17), new applications (25) rank higher
# 3. THEORY-PRACTICE BALANCE: Papers with both clean theorems AND usable artifacts rank higher
# 4. ACCESSIBILITY: Abstract math (sheaves, cohomology) ranks lower at applied venues
# 5. SCOPE: Full-paper-worthy contributions rank higher than incremental analyses
# 6. COMPUTE: All proposals are laptop-friendly, so this is roughly equal across all
