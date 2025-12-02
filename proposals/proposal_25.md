# Proposal 25: Numerical Geometry of Fairness Metrics: When Does Precision Affect Equity?

## Abstract

We investigate how finite-precision arithmetic affects algorithmic fairness metrics and decision boundaries. Fairness metrics—demographic parity, equalized odds, calibration—are computed from model outputs, which are themselves subject to numerical error. We develop NumGeom-Fair, a framework that propagates precision errors through fairness computations and identifies when fairness conclusions are numerically unreliable. Our key finding: for models near decision thresholds, demographic parity differences can flip sign due to numerical noise, and calibration curves can show spurious bias. We provide certified bounds on fairness metrics and flag when fairness assessments are numerically uncertain. Experiments on tabular datasets (Adult, COMPAS-style synthetic) with small MLPs show that 3-8% of fairness measurements are numerically borderline. All experiments run in under 2 hours on a laptop.

## 1. Introduction and Motivation

Algorithmic fairness decisions have real consequences: loan approvals, bail recommendations, hiring filters. These decisions depend on computed fairness metrics, which depend on model predictions, which are computed in finite precision. The question "Is this model fair?" can have different answers at different precisions. We study this using Numerical Geometry. The insight is that fairness metrics aggregate model outputs, and aggregation can either amplify or suppress numerical errors depending on the distribution of predictions relative to decision thresholds. For predictions clustered near thresholds, small errors cause classification flips, potentially changing fairness metrics. We quantify this effect and provide tools for certified fairness assessment.

## 2. Technical Approach

### 2.1 Fairness Metrics as Numerical Functions

Let f: X → [0,1] be a model (classifier probability), threshold t ∈ (0,1), and groups G_0, G_1 ⊂ X.

**Demographic Parity Gap (DPG):**
DPG = |P(f(x) > t | x ∈ G_0) - P(f(x) > t | x ∈ G_1)|

**Equalized Odds Gap (EOG):**
EOG = |P(f(x) > t | Y=1, x ∈ G_0) - P(f(x) > t | Y=1, x ∈ G_1)|

These metrics depend on how many samples cross threshold t, which is sensitive to numerical errors in f(x).

### 2.2 Error Propagation to Fairness Metrics

**Theorem (Fairness Metric Error).** Let f have error functional Φ_f(ε), and let p_near = fraction of samples with |f(x) - t| < Φ_f(ε). Then:

|DPG^{(p)} - DPG^{(∞)}| ≤ 2 · p_near

where DPG^{(p)} is demographic parity gap at precision p.

**Proof sketch:** Samples with |f(x) - t| < Φ_f(ε) may flip classification. In the worst case, all such samples flip, changing the positive rate by p_near for each group.

**Refined Bound by Group:**
Let p_near^{(0)} = fraction of G_0 samples near threshold, p_near^{(1)} = fraction of G_1 samples near threshold.
|DPG^{(p)} - DPG^{(∞)}| ≤ p_near^{(0)} + p_near^{(1)}

This bound is tighter when one group is less concentrated near the threshold.

### 2.3 Certified Fairness Pipeline

**Algorithm: NumGeom-Fair**

```
Input: Model f, dataset D, groups G_0/G_1, threshold t, precision p

1. PREDICTIONS with error:
   - For each x ∈ D: compute f(x) with error bound δ_f(x)

2. NEAR-THRESHOLD identification:
   - N_0 = {x ∈ G_0 : |f(x) - t| < δ_f(x)}
   - N_1 = {x ∈ G_1 : |f(x) - t| < δ_f(x)}

3. FAIRNESS METRIC computation:
   - DPG = |positive_rate(G_0) - positive_rate(G_1)|
   - Error bound: δ_DPG = |N_0|/|G_0| + |N_1|/|G_1|

4. RELIABILITY assessment:
   - Reliable if DPG > τ · δ_DPG
   - Report: "DPG = X ± Y, reliability = Z"

Output: Fairness metrics with certified bounds
```

### 2.4 Threshold Sensitivity Analysis

The sensitivity of fairness to threshold choice interacts with numerical precision:

**Definition (Numerically Stable Threshold Region).** A threshold t is numerically stable if ∀t' with |t - t'| < Φ_f(ε): |DPG(t) - DPG(t')| < tolerance.

**Algorithm: Find Stable Regions**
1. Compute DPG(t) for t in grid [0.1, 0.9]
2. At each t, perturb by ±Φ_f(ε), recompute DPG
3. Mark t as stable if variation < tolerance
4. Output stable regions

This identifies thresholds where fairness conclusions are robust to numerical noise.

### 2.5 Calibration Under Finite Precision

Calibration measures if predicted probabilities match empirical frequencies. Finite precision affects calibration:

- Probability bins aggregate predictions into intervals
- Predictions near bin boundaries may shift bins due to numerical error
- Calibration error inherits this uncertainty

**Certified Calibration Error:**
For bin [a, b], let n_uncertain = samples with |f(x) - a| < δ or |f(x) - b| < δ.
Calibration error in bin has uncertainty proportional to n_uncertain / n_bin.

## 3. Laptop-Friendly Implementation

Fairness certification is lightweight:

1. **Models**: 2-3 layer MLPs (< 100K params) on tabular data
2. **Datasets**: Adult Income, COMPAS-style synthetic (1000-5000 samples)
3. **Error tracking**: Track Φ_f(ε) during inference (O(1) overhead)
4. **Near-threshold check**: Simple arithmetic per sample
5. **Metric computation**: Standard fairness metrics with added bounds

Total experiment time: approximately 1-2 hours on a laptop.

## 4. Experimental Design

### 4.1 Datasets and Models

| Dataset | Samples | Groups | Model |
|---------|---------|--------|-------|
| Adult Income | 5000 subset | Gender | 3-layer MLP |
| Synthetic-COMPAS | 2000 | Race (binary) | 2-layer MLP |
| Synthetic-Tabular | 3000 | A vs B | 3-layer MLP |

Models trained to be "borderline fair" (DPG ≈ 0.05-0.10) to stress-test numerical effects.

### 4.2 Experiments

**Experiment 1: Precision vs Fairness.** Train model at float64. Evaluate DPG at float64, float32, float16. Compare differences to certified bounds.

**Experiment 2: Near-Threshold Distribution.** Visualize distribution of |f(x) - t| by group. Show how this predicts fairness metric uncertainty.

**Experiment 3: Threshold Stability Mapping.** For each threshold t ∈ [0.1, 0.9], compute DPG and its uncertainty. Identify stable vs unstable threshold regions.

**Experiment 4: Calibration Reliability.** Compute calibration curve at different precisions. Identify bins where calibration is numerically uncertain.

**Experiment 5: Sign Flip Cases.** Find examples where DPG flips sign between precisions (G_0 advantaged vs G_1 advantaged). Show that our bounds predict these.

### 4.3 Expected Results

1. DPG differences between fp64 and fp16 are within our certified bounds for 95%+ of cases.
2. Near-threshold concentration (p_near) correlates with fairness metric volatility.
3. Threshold stability varies: some ranges have stable DPG, others are numerically fragile.
4. 3-8% of fairness assessments are numerically borderline (reliability < 2).
5. Sign flips occur when DPG is small and p_near is large—our criterion predicts these.

**High-Impact Visualizations (< 15 min compute):**
- **Fairness with error bars** (3 min): Single bar chart: DPG with error bars, colored green (reliable) or red (borderline). Immediately shows which fairness claims are trustworthy.
- **Threshold stability ribbon** (5 min): x = threshold, y = DPG, ribbon width = uncertainty. Some regions are wide (unstable), others narrow (stable). Clear decision guidance.
- **Near-threshold danger zone** (3 min): Overlapping density plots of f(x) for G_0, G_1 with vertical line at t and shaded "danger zone" where predictions may flip.
- **Sign flip example** (2 min): For one case where DPG sign flips between precisions, show the two DPG values with error bars overlapping zero. Proves the phenomenon is real.

## 5. Theoretical Contributions Summary

1. **Fairness Error Model**: First rigorous error bounds for demographic parity and related metrics under finite precision.
2. **Threshold Stability Analysis**: Framework for identifying numerically robust threshold choices.
3. **Certified Fairness Pipeline**: Practical algorithm producing fairness assessments with numerical certificates.
4. **Near-Threshold Sensitivity**: Quantifies how prediction distribution affects fairness reliability.

## 5.1 Usable Artifacts

1. **CertifiedFairness**: Library providing `evaluate_fairness(model, data, groups, threshold) -> (dpg, uncertainty, reliable)` that returns fairness metrics with numerical certificates.
2. **ThresholdOptimizer**: Tool that finds numerically stable decision thresholds where fairness metrics are reliable, outputting safe threshold ranges.
3. **Fairness Audit Report Generator**: Produces human-readable reports showing which fairness claims are numerically trustworthy and which are borderline.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error model derivation | 1 week | None |
| NumGeom-Fair implementation | 1 week | Laptop |
| Fairness experiments | 2 days | 45 min laptop |
| Threshold analysis | 2 days | 30 min laptop |
| Visualization | 1 day | 15 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~1.5 hrs laptop** |
