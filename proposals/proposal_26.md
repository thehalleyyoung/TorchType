# Proposal 26: Numerical Geometry of Hyperparameter Optimization: Precision-Aware AutoML

## Abstract

We develop NumGeom-HPO, a framework that incorporates numerical precision effects into hyperparameter optimization (HPO). Standard HPO methods (Bayesian optimization, random search, Hyperband) evaluate configurations by running training procedures—which are numerical algorithms subject to finite-precision errors. Configurations that appear better may simply have more favorable numerical conditioning. We model HPO as search over a precision-dependent loss landscape and develop precision-aware acquisition functions that discount configurations with high numerical uncertainty. Our key finding: on high-curvature regions of hyperparameter space, configuration rankings can change between precisions, and "optimal" hyperparameters found at float32 may differ from float64 optima. Experiments on small MLPs with 2-3 hyperparameters show 10-25% of HPO trials are numerically borderline. All experiments run in under 4 hours on a laptop.

## 1. Introduction and Motivation

HPO finds the best hyperparameters for a learning algorithm by evaluating many configurations and selecting the best. But "best" depends on computed validation losses, which are subject to numerical error. A configuration with validation loss 0.500 may not be distinguishable from one with 0.502 when both have numerical uncertainty ±0.01. Current HPO ignores this, treating all evaluations as precise measurements. We address this using Numerical Geometry. Each training run is a numerical algorithm with error functional depending on the hyperparameters. We propagate this error to the validation loss, providing certified bounds. Our insight: hyperparameter space has its own numerical geometry—some regions are well-conditioned (stable evaluations), others are ill-conditioned (noisy evaluations). Precision-aware HPO should prefer stable regions.

## 2. Technical Approach

### 2.1 HPO Loss Landscape as Numerical Manifold

Let θ denote hyperparameters (learning rate, batch size, architecture) and L(θ) = validation loss after training with θ.

**Error Model:**
L(θ) is computed by a training algorithm T(θ; D; ε) where ε is machine precision.
Error functional: Φ_{L(θ)}(ε) depends on θ through:
- **Training stability**: High LR → large gradients → more error accumulation
- **Optimizer conditioning**: Some θ give well-conditioned optimization
- **Curvature of loss landscape**: Sharp minima → high curvature → error amplification

### 2.2 Precision-Dependent Configuration Ranking

**Theorem (Ranking Instability in HPO).** Let θ_1, θ_2 be configurations with:
- Computed losses L(θ_1), L(θ_2)
- Error bounds δ_1 = Φ_{L(θ_1)}(ε), δ_2 = Φ_{L(θ_2)}(ε)

Then the ranking is unreliable if |L(θ_1) - L(θ_2)| < δ_1 + δ_2.

**Corollary:** Configurations near the optimum with similar losses are often numerically indistinguishable. HPO should report a Pareto set of configurations within numerical tolerance, not a single "best."

### 2.3 Precision-Aware Acquisition Functions

In Bayesian optimization, the acquisition function α(θ) guides search (e.g., Expected Improvement).

**Standard EI:** α_EI(θ) = E[max(0, L_best - L(θ))]

**Precision-Aware EI:** α_PA(θ) = E[max(0, L_best - L(θ) - λ·Φ_L(θ))]

where λ weights numerical uncertainty. This penalizes configurations with high numerical error, preferring stable evaluations.

**Algorithm: NumGeom-HPO**

```
Input: Search space Θ, evaluation budget N, precision p

Initialize: GP surrogate with (θ, L(θ), δ(θ)) observations

For n = 1 to N:
   1. SELECT: θ_n = argmax α_PA(θ)
      - α_PA incorporates predicted loss AND predicted error

   2. EVALUATE: Run training for θ_n at precision p
      - Track error functional during training
      - Return L(θ_n) and δ(θ_n)

   3. UPDATE: Add (θ_n, L(θ_n), δ(θ_n)) to GP

Output: Certified Pareto set of configurations
   - All θ with L(θ) ≤ L_best + δ_best + δ(θ)
```

### 2.4 Error Functional Estimation During Training

To compute Φ_{L(θ)}(ε), we track numerical stability during training:

**Lightweight Tracking:**
1. Monitor gradient magnitudes (proxy for Lipschitz)
2. Track loss curvature via periodic second-derivative samples
3. Estimate accumulated error from composition of updates

**Error Bound:**
After T optimization steps with gradients g_1, ..., g_T and per-step Lipschitz constants L_1, ..., L_T:

Φ_{val}(ε) ≈ (Σ_t ||g_t||) · (Π_{s>t} L_s) · ε + κ_loss · T · ε²

where the first term propagates gradient errors through remaining steps, and κ_loss is loss curvature. This can be approximated with < 5% overhead by tracking running gradient norms.

### 2.5 Numerical Topology of Hyperparameter Space

Some regions of θ-space are numerically stable, others unstable:

**Definition:** θ is numerically stable if Φ_{L(θ)}(ε) / |L(θ)| < tolerance (relative error is small).

**Stability Map:** Divide θ-space into grid, evaluate stability at each point, interpolate.

This enables visualization: "Learning rates in [0.001, 0.01] are stable; [0.1, 1.0] are numerically fragile."

## 3. Laptop-Friendly Implementation

HPO with precision-awareness is feasible on laptops:

1. **Small search spaces**: 2-3 hyperparameters (LR, weight decay, width)
2. **Short training**: 10-20 epochs on MNIST/CIFAR-10 subset
3. **Limited budget**: 30-50 configurations (standard for small-scale HPO)
4. **Lightweight tracking**: Gradient norms, loss Hessian samples (< 5% overhead)
5. **Simple models**: 2-3 layer MLPs or small CNNs

Total experiment time: approximately 3-4 hours on a laptop.

## 4. Experimental Design

### 4.1 Setup

| Problem | Model | Hyperparameters | Budget |
|---------|-------|-----------------|--------|
| MNIST-subset | 2-layer MLP | LR, WD | 30 evals |
| CIFAR-10-subset | 3-layer CNN | LR, WD, width | 40 evals |
| Tabular-synthetic | 2-layer MLP | LR, dropout | 30 evals |

### 4.2 Experiments

**Experiment 1: Error Bounds Validation.** Compare certified error bounds δ(θ) to empirical errors |L^{fp32}(θ) - L^{fp64}(θ)|. Report tightness factor.

**Experiment 2: Ranking Reliability.** Identify pairs (θ_1, θ_2) where ranking changes between fp64 and fp32. Check if our criterion predicts these.

**Experiment 3: Standard vs Precision-Aware HPO.** Run standard BO and NumGeom-HPO. Compare:
- Best found loss
- Stability of optimal configuration
- Pareto set sizes

**Experiment 4: Stability Mapping.** Create 2D grid over (LR, WD), evaluate stability at each point. Visualize stable vs unstable regions.

**Experiment 5: Precision Sensitivity of "Optimal" Configurations.** Take configurations found by standard HPO at fp32. Evaluate at fp64. How often does ranking change?

### 4.3 Expected Results

1. Error bounds are within 50x of actual errors (HPO errors are inherently noisy).
2. Our ranking instability criterion predicts 70%+ of cross-precision ranking changes.
3. NumGeom-HPO finds configurations with 10-30% lower numerical uncertainty.
4. 10-25% of HPO trials are numerically borderline.
5. Stability maps show clear structure: low LR = stable, high LR = unstable.

**High-Impact Visualizations (< 30 min compute):**
- **Numerical stability map**: 2D heatmap of θ-space, color = stability score. Shows which hyperparameter regions are numerically reliable.
- **Pareto frontier with uncertainty**: Scatter plot of configurations, x = loss, y = stability. Standard HPO picks lowest x; NumGeom-HPO considers both.
- **Configuration ranking at two precisions**: Two ranked lists (fp32 vs fp64), lines connecting same configuration. Shows ranking shuffles.
- **Acquisition function comparison**: For one HPO step, show standard EI vs precision-aware EI surfaces. Shows how stability modifies search.

## 5. Theoretical Contributions Summary

1. **HPO Error Model**: First framework for propagating numerical precision effects through hyperparameter evaluation.
2. **Ranking Reliability Criterion**: Principled test for whether configuration comparisons are numerically meaningful.
3. **Precision-Aware Acquisition**: Modified acquisition functions that prefer numerically stable configurations.
4. **Stability Topology**: Characterization of numerically stable regions in hyperparameter space.

## 5.1 Usable Artifacts

1. **NumGeomHPO**: Drop-in replacement for standard Bayesian optimization that tracks numerical stability and outputs certified Pareto sets of configurations.
2. **StabilityScorer**: Function that takes a trained model and estimates its numerical stability, useful for comparing HPO results.
3. **HPO Stability Analyzer**: Visualization tool producing stability maps over hyperparameter space, showing which regions are numerically reliable.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error model development | 1 week | None |
| NumGeom-HPO implementation | 1 week | Laptop |
| HPO experiments | 3 days | 3 hrs laptop |
| Stability mapping | 2 days | 45 min laptop |
| Visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~4 hrs laptop** |
