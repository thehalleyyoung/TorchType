# Proposal 29: Numerical Geometry of Dataset Distillation: Precision-Aware Coresets

## Abstract

We develop NumGeom-Distill, a framework for dataset distillation and coreset construction that accounts for finite-precision effects. Dataset distillation creates small synthetic datasets that preserve learning properties of large datasets. But distillation algorithms—gradient matching, trajectory matching, kernel methods—are numerical computations subject to precision errors. We model distillation as a numerical optimization problem and derive bounds on when distilled datasets are numerically reliable. Our key finding: distilled datasets can be numerically fragile, with small precision changes causing disproportionate changes in the synthetic examples. We provide certified distillation that ensures the synthetic dataset is robust across precisions. Experiments on MNIST/CIFAR-10 subsets show that standard distillation produces numerically sensitive synthetic examples, while our method improves robustness. All experiments run on a laptop in under 4 hours.

## 1. Introduction and Motivation

Dataset distillation compresses large datasets into small synthetic datasets that train models to similar performance. Applications include efficient training, data privacy, and neural architecture search. But distillation algorithms are themselves numerical: they optimize pixel values via gradient descent, match training trajectories, or solve kernel equations. When we distill MNIST to 10 synthetic images, are those images numerically robust? Would distillation at a different precision produce different images? We address this using Numerical Geometry. Each distillation method is a numerical optimization with its own error functional. We derive bounds on the distilled data and develop precision-aware distillation that produces numerically robust synthetic examples.

## 2. Technical Approach

### 2.1 Distillation as Numerical Optimization

**Gradient Matching (GradMatch):**
min_S Σ_{batch B} ||∇_θ L(θ; B) - ∇_θ L(θ; S)||²

where S is the distilled (synthetic) dataset.

**Numerical Analysis:**
- Computing ∇_θ L has error Φ_∇(ε)
- Matching gradients: two gradient errors compound
- Total error in matching objective: 2 · Φ_∇(ε) + curvature_term

**Trajectory Matching:**
min_S ||θ_T(S) - θ_T(D)||²

where θ_T is the model after T training steps.

**Numerical Analysis:**
- Training trajectory has accumulated error (see Proposal 11)
- Matching trajectories compounds errors from both training runs
- Error: 2 · Φ_{trajectory}(ε)

### 2.2 Error Bounds for Distilled Data

**Theorem (Distillation Sensitivity).** Let S* be the optimal distilled dataset at infinite precision, and S^{(p)} at precision p. Then:

||S^{(p)} - S*||_F ≤ (1/λ_min) · Φ_objective(ε)

where λ_min is the minimum eigenvalue of the Hessian of the distillation objective (controls conditioning).

**Corollary:** Ill-conditioned distillation (small λ_min) produces numerically sensitive synthetic data.

### 2.3 Certified Distillation Pipeline

**Algorithm: NumGeom-Distill**

```
Input: Dataset D, target size n, precision p

1. DISTILL with error tracking:
   - Initialize S randomly
   - For t = 1 to T:
     a. Compute objective gradient ∇_S L with error δ_∇
     b. Update S = S - η·∇_S L
     c. Track accumulated error δ_S

2. STABILITY check:
   - Compute Hessian condition number κ at S*
   - Estimate sensitivity: ||δS|| / ε_p

3. CERTIFICATION:
   - If sensitivity > threshold: flag as numerically fragile
   - Return robustness score

Output: Distilled dataset S with robustness certificate
```

### 2.4 Precision-Aware Distillation

Add regularization to improve numerical robustness:

**Objective:**
min_S L_distill(S) + λ · R_stability(S)

where R_stability penalizes:
- High gradient magnitudes (large Lipschitz)
- High curvature of the distillation objective at S
- Sensitivity to small perturbations

**Practical Implementation:**
R_stability(S) = ||∇²_S L||_F + λ' · Var(S perturbations)

This encourages distilled data that sits in a "flat" region of the distillation landscape.

### 2.5 Coreset Selection Stability

For coreset methods that select (not synthesize) data points:

**Stability Definition:** A coreset C is stable if small changes to distances/weights don't change the selection.

**Error Propagation:**
- Distance computations have error δ_d
- Selection depends on relative distances
- Samples with similar distances to the margin may swap

**Certified Coresets:**
Return coreset C with "margin" indicating how stably each point was selected.

## 3. Laptop-Friendly Implementation

Distillation certification is feasible at small scale:

1. **Datasets**: MNIST (1000 training samples), CIFAR-10 (5000 samples)
2. **Distillation target**: 10-50 synthetic images (standard for laptop-scale)
3. **Models**: 2-3 layer CNNs for training
4. **Distillation epochs**: 500-1000 (standard practice)
5. **Error tracking**: Track gradient norms, Hessian estimates via finite differences

Total experiment time: approximately 3-4 hours on a laptop.

## 4. Experimental Design

### 4.1 Setup

| Dataset | Train Size | Distilled Size | Methods |
|---------|------------|----------------|---------|
| MNIST | 1000 | 10, 50 | GradMatch, Trajectory |
| CIFAR-10 | 5000 | 10, 100 | GradMatch |
| Fashion-MNIST | 2000 | 20 | GradMatch, Coreset |

### 4.2 Experiments

**Experiment 1: Distillation Precision Sensitivity.** Run GradMatch at fp64, fp32, fp16. Compare resulting synthetic datasets (L2 distance, visual similarity). Measure downstream accuracy on models trained with each.

**Experiment 2: Robustness Score Validation.** Compute our robustness score for each distilled dataset. Correlate with empirical precision sensitivity.

**Experiment 3: Standard vs Precision-Aware Distillation.** Compare:
- Standard GradMatch
- NumGeom-Distill (with stability regularization)
Measure robustness score and downstream accuracy.

**Experiment 4: Conditioning Analysis.** Estimate Hessian condition number at the distillation optimum. Show correlation with numerical sensitivity.

**Experiment 5: Coreset Stability.** For coreset selection, identify samples with low selection margin. Show these are the ones that change between precisions.

### 4.3 Expected Results

1. Standard distillation at fp16 produces synthetic data differing from fp64 by 5-15% (L2 relative norm).
2. Downstream accuracy varies by 1-3% depending on distillation precision.
3. NumGeom-Distill produces synthetic data with 2-5x lower precision sensitivity.
4. High Hessian condition number (κ > 100) correlates with numerical fragility.
5. 10-20% of coreset selections are numerically borderline.

**High-Impact Visualizations (< 20 min compute):**
- **Distilled image gallery** (10 min): 3×10 grid showing same 10 distilled MNIST digits at fp64/fp32/fp16. Differences are often visible to the naked eye—most striking figure.
- **Robustness score bar chart** (3 min): Horizontal bars for each distilled image, length = robustness score. Some images are stable (long bars), others fragile (short bars).
- **Downstream accuracy vs precision** (5 min): Line plot showing test accuracy of models trained on distilled data. Steeper slope = more fragile distillation.
- **Hessian conditioning scatter** (2 min): Each distilled image as a point, x = Hessian condition number, y = empirical precision sensitivity. Shows strong correlation.

## 5. Theoretical Contributions Summary

1. **Distillation Error Model**: First analysis of numerical precision effects in dataset distillation.
2. **Sensitivity Bound**: Relates distillation conditioning to precision sensitivity.
3. **Precision-Aware Distillation**: Regularization approach for numerically robust synthetic data.
4. **Coreset Stability Analysis**: Margin-based reliability for subset selection.

## 5.1 Usable Artifacts

1. **RobustDistill**: Drop-in replacement for GradMatch/trajectory matching that adds stability regularization, producing numerically robust synthetic datasets.
2. **DistillationRobustnessScorer**: Function that takes a distilled dataset and estimates its numerical robustness score, useful for comparing distillation methods.
3. **CoresetStabilityAnalyzer**: Given a coreset selection, outputs margin scores for each selected sample indicating how stably it was selected.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error model development | 1 week | None |
| NumGeom-Distill implementation | 1.5 weeks | Laptop |
| Distillation experiments | 3 days | 3 hrs laptop |
| Stability analysis | 2 days | 45 min laptop |
| Visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **5 weeks** | **~4 hrs laptop** |
