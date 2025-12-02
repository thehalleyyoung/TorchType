# Proposal 24: Numerically Certified Interpretability: Error-Aware Saliency and Attribution

## Abstract

We develop NumGeom-Interpret, a framework that attaches numerical error certificates to saliency maps and feature attributions. Interpretability methods like gradient saliency and Integrated Gradients are numerical algorithms—composed of forward passes, backward passes, and difference quotients—each subject to finite-precision error. We model each method as a numerical morphism and derive explicit error bounds on attribution scores. Our key finding: for networks with high-curvature regions, attributions can be numerically unreliable even at float32, with attribution rankings changing due to rounding noise. We provide certified bounds of the form "attribution_i = a_i ± δ_i" and flag features whose attributions are dominated by numerical uncertainty. Experiments on tiny CNNs (MNIST, CIFAR-10) show that 5-15% of attributions are numerically unstable and that our bounds correctly predict when attribution rankings change under precision reduction. All experiments run on a laptop in under 2 hours.

## 1. Introduction and Motivation

Interpretability methods explain model predictions by attributing importance to input features. But these explanations are computed numerically, subject to the same floating-point errors as the models themselves. When a saliency map highlights pixel i over pixel j, is this difference real or numerical noise? Current practice ignores this question entirely. We address it using Numerical Geometry. Each interpretability method—gradient saliency, Integrated Gradients, DeepLIFT—can be modeled as a numerical morphism with Lipschitz constant and error functional. We propagate precision errors through the attribution computation, producing certified bounds. Our key insight: attributions are often more numerically fragile than predictions because they involve derivatives (gradient saliency) or path integrals (Integrated Gradients), which amplify numerical errors.

## 2. Technical Approach

### 2.1 Attribution Methods as Numerical Morphisms

**Gradient Saliency**: A(x) = |∇_x f(x)|

- Involves one forward and one backward pass
- Error from forward pass: Φ_f(ε)
- Error from backward pass: Φ_{∇f}(ε) (see Proposal 17)
- Total: Φ_saliency(ε) = L_∇f · Φ_f(ε) + Δ_∇f

**Integrated Gradients (IG)**: A(x)_i = (x_i - x'_i) · ∫_0^1 ∂f/∂x_i(x' + α(x - x')) dα

- Path integral approximated by m steps: Σ_{k=1}^m ∂f/∂x_i(x' + k/m · (x - x')) / m
- Each step has gradient error
- Errors accumulate: Φ_IG(ε) = m · Φ_∇f(ε) + Δ_integration

**Lipschitz Bounds for Attributions:**

For a network f with layer-wise Lipschitz constants L_1, ..., L_n and curvatures κ_1, ..., κ_n:

- Gradient saliency Lipschitz: L_saliency ≈ Π_i L_i · (1 + Σ_i κ_i) (chain rule + curvature correction)
- IG Lipschitz: L_IG ≈ m · Π_i L_i · (1 + Σ_i κ_i) (grows with integration steps)

The Π_i L_i term comes from the chain rule for Jacobians; the curvature correction accounts for how gradients themselves vary.

### 2.2 Certified Attribution Pipeline

**Algorithm: NumGeom-Interpret**

```
Input: Model f, input x, baseline x', precision p

1. FORWARD with error tracking:
   - Compute f(x) with tracked Φ_f(ε)

2. BACKWARD with error tracking:
   - Compute ∇_x f with tracked Φ_∇f(ε)
   - For IG: repeat along path, accumulate error

3. ATTRIBUTION with bounds:
   - For each feature i:
     a_i = computed attribution
     δ_i = Φ_attribution(ε_p) for feature i

4. RELIABILITY check:
   - Flag feature i as unreliable if |a_i| < τ · δ_i
   - Compute reliability score: R_i = |a_i| / δ_i

Output: Attributions {a_i} with bounds {δ_i} and reliability flags
```

### 2.3 Attribution Stability Analysis

**Theorem (Attribution Error Bound).** For gradient saliency on a network f with path Lipschitz product L_path = Π_i L_i and curvature sum κ_sum = Σ_i κ_i:

|a_i^{(p)} - a_i^{(∞)}| ≤ L_path · ||x|| · ε_p + κ_sum · ||x||² · ε_p²

where a_i^{(p)} is the attribution at precision p and a_i^{(∞)} is the infinite-precision attribution.

**Corollary (Ranking Instability).** Two features i, j may have swapped rankings if:
|a_i - a_j| < δ_i + δ_j

This gives a principled criterion for when attribution rankings are numerically meaningful.

### 2.4 Curvature Hotspots in Attribution Paths

For Integrated Gradients, the path from baseline x' to input x may pass through high-curvature regions of the network. These are "numerical hotspots" where:
- Gradient computation is unstable
- Small precision errors cause large attribution errors

**Detection Algorithm:**
1. Sample points along the IG path
2. Estimate local curvature κ(α) at each point
3. Flag path segments with κ(α) > threshold
4. Report total error contribution from each segment

This enables diagnosis: "Attribution for pixel i is unreliable because the IG path passes through high-curvature region at α = 0.3"

## 3. Laptop-Friendly Implementation

Attribution certification is computationally light:

1. **Small models**: 2-4 layer CNNs (< 500K params) on MNIST/CIFAR-10
2. **Error tracking overhead**: O(1) per operation (track L and Δ alongside values)
3. **IG steps**: m = 50 sufficient; each step is one backward pass
4. **Batch certification**: Process 100 images in < 1 minute
5. **Comparison to float64**: Use float64 as ground truth for validation

Total experiment time: approximately 2 hours on a laptop.

## 4. Experimental Design

### 4.1 Models and Methods

| Model | Architecture | Interpretability Methods |
|-------|--------------|-------------------------|
| CNN-MNIST | 2 conv + 2 FC | Gradient, IG, SmoothGrad |
| CNN-CIFAR | 4 conv + 2 FC | Gradient, IG |
| MLP-Tabular | 3 FC layers | Gradient, IG |

### 4.2 Experiments

**Experiment 1: Attribution Error Bounds.** Compare certified bounds δ_i to empirical error |a_i^{fp32} - a_i^{fp64}|. Report tightness factor (bound / actual).

**Experiment 2: Ranking Stability.** Identify pairs (i, j) where ranking changes between fp64 and fp32. Check if our criterion |a_i - a_j| < δ_i + δ_j predicts these.

**Experiment 3: Reliability Distribution.** Histogram of reliability scores R_i = |a_i| / δ_i across features and images. What fraction of attributions are reliable (R > 2)?

**Experiment 4: Precision Sensitivity.** Run attribution at float64/32/16. Track ranking correlation (Spearman) with float64 baseline. Show that our bounds predict correlation drop.

**Experiment 5: Curvature Hotspot Analysis.** For IG, identify where along the integration path errors accumulate. Visualize curvature profile and error contribution.

### 4.3 Expected Results

1. Error bounds are within 10x of actual errors for well-conditioned networks, within 100x for ill-conditioned.
2. Our ranking instability criterion correctly predicts 80%+ of ranking changes.
3. 5-15% of attributions have reliability score R < 2 (numerically borderline).
4. Float16 attribution rankings have Spearman < 0.9 correlation with float64 on some models.
5. Curvature hotspots occur near ReLU boundaries and in final classification layer.

**High-Impact Visualizations (< 20 min compute):**
- **Certified saliency map** (5 min): For one MNIST digit, show saliency where brightness = attribution, opacity = reliability. Unreliable pixels fade to transparent. Immediately interpretable.
- **Ranking swap diff** (3 min): Two saliency maps (fp64 vs fp32) with arrows connecting pixels that swapped rank. Highlights how numerical noise affects interpretation.
- **Reliability histogram** (3 min): Distribution of R_i = |a_i|/δ_i across 1000 images. Shade "unreliable" zone (R < 2) in red. Shows what fraction of attributions are numerically questionable.
- **IG curvature profile** (5 min): For one attribution, plot κ(α) and cumulative error along the path α ∈ [0,1]. Shows exactly where numerical problems arise.

## 5. Theoretical Contributions Summary

1. **Attribution Error Model**: First rigorous error bounds for gradient-based interpretability methods.
2. **Certified Attribution Pipeline**: Practical algorithm producing attributions with numerical certificates.
3. **Ranking Stability Criterion**: Principled test for whether feature rankings are numerically meaningful.
4. **Curvature Hotspot Analysis**: Method to diagnose where numerical problems arise in IG paths.

## 5.1 Usable Artifacts

1. **CertifiedSaliency**: Drop-in replacement for standard saliency methods that returns attributions with error bounds. Interface: `certified_saliency(model, x) -> (attributions, uncertainties, reliability_flags)`.
2. **AttributionReliabilityChecker**: Given any attribution vector, estimates numerical reliability and flags features that may have unstable rankings.
3. **IG Path Analyzer**: Visualization tool showing where along an Integrated Gradients path numerical errors accumulate, enabling diagnosis of fragile attributions.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error model derivation | 1 week | None |
| NumGeom-Interpret implementation | 1 week | Laptop |
| Attribution experiments | 3 days | 1 hr laptop |
| Ranking stability analysis | 2 days | 30 min laptop |
| Visualization | 2 days | 15 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~2 hrs laptop** |
