# Proposal 22: Numerical Geometry of Sampling: Stable Low-Precision MCMC and Diffusion

## Abstract

We apply Numerical Geometry to sampling algorithms, modeling MCMC and diffusion samplers as numerical morphisms with explicit error functionals. We prove that finite-precision arithmetic introduces bias in sampled distributions that compounds over sampling steps according to the Stability Composition Theorem. Our key result: below a precision threshold determined by the curvature of the log-density, numerical error dominates discretization error, making additional sampling steps counterproductive. We derive precision bounds for Unadjusted Langevin Algorithm (ULA), MALA, and simple diffusion samplers. Experiments on 1D/2D Gaussian mixtures, banana distributions, and small Bayesian posteriors verify that low-precision sampling fails when predicted and that curvature of the target predicts sensitivity. All experiments run on a laptop in under 2 hours.

## 1. Introduction and Motivation

Sampling algorithms power Bayesian inference, generative models, and uncertainty quantification. But sampling is iterative: each step of Langevin dynamics or MCMC involves gradient computations, random number generation, and state updates—all subject to finite-precision error. In diffusion models, thousands of denoising steps compound these errors. When does precision become the bottleneck? We answer this using Numerical Geometry. Each sampler step is a numerical morphism on state space with Lipschitz constant L (related to step size and curvature) and intrinsic error Δ (from floating-point arithmetic). The Stability Composition Theorem tracks error accumulation. Our key insight: for targets with high curvature κ in the log-density, the precision threshold scales as p* ∝ log(κ). Below this threshold, samples drift away from the true distribution regardless of how many steps you take.

## 2. Technical Approach

### 2.1 Samplers as Numerical Morphisms

**Unadjusted Langevin Algorithm (ULA)**: 
x_{k+1} = x_k + η∇log π(x_k) + √(2η) ξ_k

where π is the target density and ξ_k ~ N(0, I).

We model the ULA step as a numerical morphism with:
- **Lipschitz constant**: L_ULA = 1 + η · L_∇log π ≈ 1 + η · κ where κ is the curvature of log π
- **Intrinsic error**: Δ_ULA = ε_mach · (||x|| + η · ||∇log π|| + √(2η) · ||ξ||)

**MALA (Metropolis-Adjusted Langevin)**:
Same proposal as ULA, but with accept/reject step. The accept probability computation introduces additional numerical error.

**Simple Diffusion Sampler** (1D/2D toy):
For score-based sampling, x_{t-Δt} = x_t + Δt · s_θ(x_t, t) + √(Δt) ξ

where s_θ is a learned score function (small MLP for toy problems).

### 2.2 Distributional Error from Numerical Noise

**Definition (Numerical Sampling Error).** For a sampler producing samples {x_k} at precision p, define the numerical sampling error as the Wasserstein-2 distance:

E_num(k, p) = W_2(Law(x_k^{(p)}), Law(x_k^{(∞)}))

where x_k^{(∞)} is the infinite-precision sampler output.

**Theorem (Sampling Error Accumulation).** For ULA with step size η on a target with curvature κ = ||∇²log π||, after k steps:

E_num(k, p) ≤ min((1 + ηκ)^k, C_mix/η) · ε_p · C_init + Δ_ULA / (ηκ)

where ε_p = 2^{-p}, C_init depends on initialization, and C_mix is a mixing constant.

**Proof Strategy.** Apply the Stability Composition Theorem to k iterations. The Lipschitz constant L_ULA ≈ 1 + ηκ governs error amplification, but mixing of the chain bounds the growth: after the mixing time τ_mix ≈ 1/(ηκ), errors saturate rather than growing exponentially. The min() reflects that error growth is geometric initially but bounded by mixing.

### 2.3 Precision Lower Bound for Sampling

**Theorem (Sampling Precision Lower Bound).** To achieve sampling accuracy ε in W_2 distance after k steps of ULA:

p ≥ log₂((1 + ηκ)^k · ||x||_scale / ε)

where ||x||_scale is the typical sample magnitude.

**Corollary (Numerical vs Discretization Error).** ULA has discretization error O(η). Numerical error is O(ε_p · (1+ηκ)^k). For the numerical error to be subdominant:

ε_p < η / ((1+ηκ)^k · C)

This sets a minimum precision for a given step size and chain length.

### 2.4 Curvature Sensitivity Analysis

**Key Insight**: High curvature regions of log π are numerically sensitive.

For multimodal distributions (Gaussian mixtures), curvature is high between modes. Samples transitioning between modes are most affected by numerical error.

For heavy-tailed distributions, curvature is low in the tails but high near the mode. Numerical error concentrates near the mode.

**Proposition.** For a Gaussian mixture with separation Δ between modes and variance σ²:
κ_max ≈ 1/σ² + (Δ/σ²)² · exp(-Δ²/8σ²)

The first term is within-mode curvature; the second is the saddle point between modes. For well-separated modes (Δ >> σ), the saddle curvature dominates, requiring higher precision for between-mode transitions.

## 3. Laptop-Friendly Implementation

All experiments use low-dimensional sampling problems:

1. **1D targets**: Gaussian, bimodal mixture, heavy-tailed
2. **2D targets**: Gaussian mixture (2-4 modes), banana distribution, funnel
3. **Small Bayesian posteriors**: Logistic regression with < 10 parameters
4. **Toy diffusion**: 1D/2D score-based model with 2-layer MLP score network
5. **Precision simulation**: Float64 with explicit rounding + noise injection

Ground truth obtained via long-run high-precision chains or analytical formulas.

Total compute: approximately 2 hours on a laptop.

## 4. Experimental Design

### 4.1 Target Distributions

| Target | Dimension | Curvature Profile | Ground Truth |
|--------|-----------|-------------------|--------------|
| Gaussian | 1D, 2D | Constant | Analytical |
| Mixture-2 | 1D, 2D | High between modes | Analytical |
| Mixture-4 | 2D | Very high | Long chain |
| Banana | 2D | Varying | Long chain |
| Bayesian LogReg | 5D | Moderate | Long chain |

### 4.2 Experiments

**Experiment 1: Precision vs Sampling Error.** For each target, run ULA at different precisions (8, 16, 32, 64 bit). Measure W_2 distance to ground truth after 10K steps.

**Experiment 2: Error Accumulation Curve.** Track E_num(k, p) over sampling iterations. Compare to theoretical prediction.

**Experiment 3: Curvature Correlation.** Estimate κ for each target. Plot precision threshold p* vs κ. Verify log-linear relationship.

**Experiment 4: MALA vs ULA.** Compare ULA (no accept/reject) to MALA (with accept/reject). Show that accept/reject computation introduces additional precision sensitivity.

**Experiment 5: Toy Diffusion Sampler.** Train a tiny score network on 2D Gaussian mixture. Sample at different precisions. Show that low-precision sampling produces mode collapse or spurious modes.

### 4.3 Expected Results

1. Sampling error shows characteristic "floor" at precision-dependent level.
2. Error accumulation matches Φ theory within 5x for most targets.
3. Curvature κ correlates strongly (r > 0.85) with required precision.
4. MALA is slightly more precision-sensitive than ULA due to accept probability.
5. Toy diffusion at float16 shows visible artifacts; float32 matches float64.

**High-Impact Visualizations (< 20 min compute):**
- **Sample cloud gallery** (5 min): 2×4 grid showing 2D Gaussian mixture samples at 64/32/16/8 bit. Visible mode collapse at low precision—most visually striking figure.
- **Precision-W₂ elbow plot** (3 min): For banana distribution, W₂ error vs precision bits. Sharp elbow at theoretical threshold. Simple, clear message.
- **Curvature-error overlay** (5 min): 2D heatmap of log π curvature as background, scatter overlay of sample errors (colored by magnitude). Shows errors concentrate in high-κ regions.
- **Mode weight bar chart** (3 min): For 4-mode mixture, grouped bars showing true vs estimated mode weights at each precision. Low precision shows mode weight drift.

## 5. Theoretical Contributions Summary

1. **Sampler as Numerical Morphism**: First rigorous numerical error model for MCMC and diffusion sampling.
2. **Precision Lower Bound for Sampling**: Theorem relating target curvature to minimum bit-depth.
3. **Numerical vs Discretization Tradeoff**: Analysis of when precision becomes the bottleneck in sampling.
4. **Curvature-Sensitivity Theory**: Explanation of why multimodal and curved targets are precision-sensitive.

## 5.1 Usable Artifacts

1. **CurvatureEstimator**: Function that estimates log-density curvature κ from samples, enabling precision requirement prediction for any target distribution.
2. **PrecisionAwareSampler**: Drop-in replacement for ULA/MALA that tracks numerical error and warns when samples may be unreliable.
3. **Sampling Precision Calculator**: Given target curvature κ, step size η, and desired W_2 accuracy, outputs minimum required precision.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Theory development | 1 week | None |
| 1D/2D sampling experiments | 3 days | 30 min laptop |
| Bayesian posterior experiments | 2 days | 30 min laptop |
| Toy diffusion experiments | 3 days | 1 hr laptop |
| Visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~2 hrs laptop** |
