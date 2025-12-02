# Proposal 28: Numerically Aware Probabilistic Inference: Error Functionals for VI and MCMC

## Abstract

We develop NumGeom-Inference, a framework that incorporates finite-precision error analysis into probabilistic inference algorithms. Variational inference (VI) and Markov Chain Monte Carlo (MCMC) are numerical algorithms—optimizing ELBOs, computing gradients, generating random samples—all subject to floating-point errors. We model each inference algorithm as a numerical morphism and derive explicit error bounds on posterior approximations. Our key finding: for models with high-curvature posteriors, the ELBO can be numerically unstable, and VI can converge to different local optima at different precisions. For MCMC, acceptance probabilities near 0 or 1 are numerically fragile. We provide certified bounds on inference quality and flag when posterior estimates are numerically uncertain. Experiments on small Bayesian neural networks and Gaussian mixture models show that 5-15% of inference runs have numerically borderline conclusions. All experiments run on a laptop in under 3 hours.

## 1. Introduction and Motivation

Probabilistic inference is fundamentally numerical: we optimize objectives (VI), sample from distributions (MCMC), or compute normalizing constants. All these operations have finite-precision implementations that introduce numerical error. When we say "the posterior probability of θ > 0 is 0.73," this is a computed number subject to numerical uncertainty. We study this using Numerical Geometry. Each inference algorithm is a composition of numerical morphisms—gradients, exponentials, random number generation—with Lipschitz constants and curvatures that determine error propagation. Our insight: probabilistic inference has special numerical challenges because it involves log-probabilities (which can underflow), softmax/exp (which can overflow), and stochastic estimates (which interact with numerical noise).

## 2. Technical Approach

### 2.1 Variational Inference as Numerical Optimization

VI optimizes the Evidence Lower Bound (ELBO):
L(φ) = E_{q_φ}[log p(x, z)] - E_{q_φ}[log q_φ(z)]

**Error Sources:**
1. **Gradient computation**: ∇_φ L(φ) via reparameterization or score function
2. **Log-probability evaluation**: log p(x, z) involves products → sums of logs
3. **Entropy computation**: -E[log q] can involve numerical integration
4. **Optimization trajectory**: Errors accumulate over updates

**Error Model for ELBO:**
Φ_{ELBO}(ε) = (L_p + L_q) · ε + T · (η · ||∇L||_avg) · ε

where L_p, L_q are Lipschitz constants of log p, log q, T is the number of optimization steps, η is the learning rate, and ||∇L||_avg is average gradient norm. The first term is per-evaluation error; the second is accumulated optimization error.

### 2.2 MCMC as Numerical Sampling

Metropolis-Hastings accepts proposal x' → x with probability:
α = min(1, p(x')q(x|x') / p(x)q(x'|x))

**Numerical Issues:**
1. **Ratio computation**: p(x')/p(x) computed as exp(log p(x') - log p(x))
2. **Near-boundary acceptance**: When α ≈ 0 or α ≈ 1, numerical noise dominates
3. **Random number comparison**: u < α where u ~ Uniform(0,1)

**Theorem (Acceptance Probability Error).** Let α be the true acceptance probability and α^{(p)} the computed value at precision p. Then:

|α^{(p)} - α| ≤ α · (exp(δ_log) - 1) where δ_log = ε · (|log p(x')| + |log p(x)|)

For small δ_log, this simplifies to |α^{(p)} - α| ≈ α · δ_log.

This bound is large when log-probabilities are large (concentrated posteriors) or when α is near 1.

### 2.3 Certified Inference Pipeline

**Algorithm: NumGeom-VI**

```
Input: Model p(x,z), variational family q_φ, precision p

1. INITIALIZE with error tracking:
   - φ_0 = initial parameters
   - δ_0 = 0 (no accumulated error)

2. OPTIMIZE with tracking:
   For t = 1 to T:
     - Compute ∇L(φ_t) with error δ_∇
     - Update: φ_{t+1} = φ_t - η·∇L(φ_t)
     - Accumulate: δ_{t+1} = δ_t + η·δ_∇

3. FINAL bounds:
   - ELBO = L(φ_T) ± Φ_L(ε)
   - Posterior mean = E_{q_{φ_T}}[z] ± Φ_E(ε)

Output: Variational posterior with certified bounds
```

**Algorithm: NumGeom-MCMC**

```
Input: Target p(x), proposal q, precision p, samples N

1. SAMPLE with acceptance tracking:
   For n = 1 to N:
     - Propose x' ~ q(·|x_n)
     - Compute α with error δ_α
     - Flag if |α - 0.5| < δ_α (numerically borderline)
     - Accept/reject

2. ESTIMATE with bounds:
   - Mean = (1/N)Σ x_n ± standard_error + numerical_error
   - Numerical error from flagged samples

Output: Posterior samples with reliability flags
```

### 2.4 Log-Space Stability Analysis

Probabilistic computations often use log-space for stability:
log p(x) = Σ_i log f_i(x)

**LogSumExp Stability:**
log(Σ_i exp(a_i)) = max(a) + log(Σ_i exp(a_i - max(a)))

Error: δ ≈ n · ε where n is number of terms.

**Underflow/Overflow Thresholds:**
At float32: exp(x) underflows for x < -88, overflows for x > 88.
These create "numerical cliffs" in probability space.

**Certified Probability Bounds:**
When log p(x) is computed, we track:
- log p(x) ± δ_{log}
- This gives p(x) ∈ [exp(log p - δ), exp(log p + δ)]
- Multiplicative uncertainty: p(x) · [e^{-δ}, e^{+δ}]

### 2.5 Curvature of Posteriors

High-curvature posteriors amplify numerical errors:

**Definition (Posterior Curvature).** κ_post = max eigenvalue of -∇² log p(z|x)

Regions with high κ_post have:
- Sharp probability gradients → large L_p
- Sensitive acceptance ratios → unreliable MCMC
- Narrow variational optima → sensitive VI

## 3. Laptop-Friendly Implementation

Probabilistic inference certification is feasible on small models:

1. **Models**: Small BNNs (2 layers, < 1000 params), GMMs (2-5 components)
2. **VI**: 1000-5000 ELBO iterations
3. **MCMC**: 1000-5000 samples (after burn-in)
4. **Error tracking**: Track log-probability ranges, gradient norms
5. **Comparison**: fp64 vs fp32 as ground truth validation

Total experiment time: approximately 2-3 hours on a laptop.

## 4. Experimental Design

### 4.1 Models

| Model | Type | Parameters | Inference |
|-------|------|------------|-----------|
| BNN-MNIST | Bayesian MLP | 1000 weights | VI (mean-field) |
| GMM-2D | Mixture model | 20 (2 components) | MCMC (Gibbs) |
| Logistic Regression | Linear classifier | 50 | VI, MCMC |
| Hierarchical Normal | Hierarchical model | 10 | MCMC (HMC-lite) |

### 4.2 Experiments

**Experiment 1: ELBO Error Bounds.** Compare certified bounds on ELBO to empirical |ELBO^{fp32} - ELBO^{fp64}|. Report tightness factor.

**Experiment 2: Posterior Mean Stability.** Compute posterior mean at fp64, fp32, fp16. Compare differences to certified bounds.

**Experiment 3: Acceptance Probability Reliability.** For MCMC, flag samples where acceptance was numerically borderline (|α - 0.5| < δ_α). Report fraction.

**Experiment 4: Convergence Precision Sensitivity.** Run VI at different precisions. Do they converge to the same optimum? Plot ELBO trajectories.

**Experiment 5: Posterior Curvature Mapping.** Estimate posterior curvature at different regions. Show correlation with numerical uncertainty.

### 4.3 Expected Results

1. ELBO bounds are within 20x of actual errors for smooth posteriors.
2. Posterior means differ by 1-5% between fp32 and fp64 for high-curvature models.
3. 5-15% of MCMC acceptances are numerically borderline.
4. VI at fp16 sometimes converges to different optima than fp64.
5. High posterior curvature correlates with numerical instability.

**High-Impact Visualizations (< 20 min compute):**
- **ELBO with uncertainty band** (5 min): ELBO vs iteration with shaded error band. Shows when "convergence" is within numerical noise vs genuine.
- **Posterior contour triptych** (5 min): Three 2D posterior contour plots (fp64/fp32/fp16) side by side for GMM. Shape changes are visually obvious.
- **Acceptance reliability scatter** (5 min): 2D posterior samples colored by acceptance reliability (green = solid, red = borderline). Fragile regions cluster near mode boundaries.
- **Log-probability cliff diagram** (3 min): 1D slice of log p(z) with horizontal lines at underflow (-88) and overflow (+88) thresholds. Shows where numerical cliffs lie.

## 5. Theoretical Contributions Summary

1. **Inference Error Model**: First systematic error analysis for VI and MCMC under finite precision.
2. **Certified ELBO Bounds**: Error propagation through variational optimization.
3. **MCMC Reliability Criterion**: Flags samples where acceptance was numerically uncertain.
4. **Log-Space Stability Analysis**: Bounds on probabilities computed via log-space operations.

## 5.1 Usable Artifacts

1. **CertifiedVI**: Variational inference wrapper that tracks ELBO uncertainty and outputs posterior with error bounds: `certified_vi(model) -> (posterior_params, uncertainties)`.
2. **MCMCReliabilityTracker**: Wrapper for MCMC samplers that flags samples with numerically borderline acceptance, enabling post-hoc reliability analysis.
3. **PosteriorPrecisionChecker**: Given a probabilistic model, estimates the posterior curvature and recommends minimum precision for reliable inference.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error model derivation | 1.5 weeks | None |
| NumGeom-Inference implementation | 1 week | Laptop |
| VI experiments | 2 days | 1.5 hrs laptop |
| MCMC experiments | 2 days | 1 hr laptop |
| Visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **4.5 weeks** | **~3 hrs laptop** |
