# Proposal 27: Topological Data Analysis Under Finite Precision: Certified Persistent Homology

## Abstract

We develop NumGeom-TDA, a framework for computing persistent homology with certified numerical guarantees. Persistent homology extracts topological features (connected components, loops, voids) from data, but the computations—distance matrices, filtration values, boundary matrix reductions—are all subject to finite-precision errors. We model TDA computations as numerical morphisms and derive explicit bounds on persistence diagrams. Our key finding: features with short "persistence" (death - birth) may be indistinguishable from numerical noise, and feature rankings can change under precision reduction. We provide certified persistence diagrams of the form "feature has persistence p ± δ" and flag features that are numerically uncertain. Experiments on 2D point clouds (1000-5000 points) and small images show that 5-20% of persistent features are numerically borderline. All experiments run on a laptop in under 2 hours.

## 1. Introduction and Motivation

Topological Data Analysis (TDA) uses persistent homology to identify robust topological features in data. A feature's "persistence" (lifetime in the filtration) indicates its significance—long-lived features are real, short-lived are noise. But what if the noise is numerical rather than statistical? When computing distances, filtration values, and boundary matrices in finite precision, small errors propagate and can create or destroy topological features. We address this using Numerical Geometry. Each step of the TDA pipeline is a numerical morphism with Lipschitz constant and error functional. We compose these to get certified bounds on the final persistence diagram. Our insight: persistence diagrams have their own numerical geometry, and features near the diagonal (low persistence) are often numerically uncertain.

## 2. Technical Approach

### 2.1 TDA Pipeline as Composed Numerical Morphisms

**Step 1: Distance Matrix.**
D_{ij} = ||x_i - x_j||_2

Absolute error: δ_{ij} = ε · (||x_i|| + ||x_j||) (from floating-point norm computation)
Relative error: δ_{ij} / D_{ij} = ε · (||x_i|| + ||x_j||) / ||x_i - x_j|| (large for nearby points)

**Step 2: Filtration.**
For Vietoris-Rips, edge (i,j) appears at filtration value D_{ij}/2.
Error in filtration value = δ_{ij}/2.

**Step 3: Boundary Matrix Reduction.**
Gaussian elimination on sparse boundary matrix over Z/2Z.
Error: Can amplify when near-zero pivots arise (rare for Z/2Z, more common for weighted variants).

### 2.2 Certified Persistence Diagrams

**Theorem (Persistence Error Bound).** Let (b, d) be a point in the persistence diagram with birth time b and death time d. Let δ_b, δ_d be the errors in the filtration values that determine birth and death. Then:

- True birth time: b ± δ_b
- True death time: d ± δ_d
- True persistence: (d - b) ± (δ_b + δ_d)

**Corollary (Feature Reliability).** A feature is numerically reliable if:
(d - b) > τ · (δ_b + δ_d)

where τ ≥ 2 gives reasonable confidence. Features with persistence below this threshold are indistinguishable from numerical noise.

### 2.3 Certified TDA Pipeline

**Algorithm: NumGeom-TDA**

```
Input: Point cloud X ⊂ R^d, precision p, max dimension k

1. DISTANCE MATRIX with error:
   - D_{ij} = ||x_i - x_j||
   - δ_{ij} = error bound from norm computation

2. FILTRATION with error:
   - Build Vietoris-Rips filtration
   - Track δ for each simplex's filtration value

3. PERSISTENT HOMOLOGY:
   - Standard persistence algorithm
   - For each feature (b, d), inherit δ_b, δ_d from simplices

4. CERTIFICATION:
   - For each (b, d): compute reliability = (d-b) / (δ_b + δ_d)
   - Flag unreliable features

Output: Persistence diagram with certified bounds and reliability flags
```

### 2.4 Stability of Persistence Diagrams

Classical stability theorem says bottleneck distance between diagrams is bounded by sup-norm of filtration value changes. We extend this:

**Theorem (Numerical Stability of Diagrams).** Let PD^{(p)} be the diagram at precision p, PD^{(∞)} at infinite precision. Then:

d_B(PD^{(p)}, PD^{(∞)}) ≤ max_{simplex} δ_{filt}

where δ_{filt} is the maximum filtration value error across all simplices.

This gives a global bound, but our per-feature bounds are more informative for identifying which features are affected.

### 2.5 Curvature Effects in TDA

Point cloud geometry affects numerical stability:

- **Clustered points**: Nearby points have small distances, high relative error
- **Sparse regions**: Large distances, lower relative error
- **Uniform density**: Most stable configuration

**Curvature-Aware Filtration:**
Weight filtration values by local point density to balance numerical error across scales.

## 3. Laptop-Friendly Implementation

TDA certification is tractable on small datasets:

1. **Point clouds**: 1000-5000 points in R^2 or R^3 (image patches also work)
2. **Homology dimension**: H_0 and H_1 only (connected components and loops)
3. **Efficient libraries**: Ripser, Gudhi for fast persistence computation
4. **Error tracking**: Maintain δ for filtration values alongside values
5. **Sparse distance matrices**: Truncate at max filtration value

Total experiment time: approximately 1-2 hours on a laptop.

## 4. Experimental Design

### 4.1 Datasets

| Dataset | Points | Dimension | Known Topology |
|---------|--------|-----------|----------------|
| Noisy circle | 500-2000 | 2D | 1 loop (H_1) |
| Noisy torus | 1000-3000 | 3D | 2 loops, 1 void |
| Figure-8 | 500-1500 | 2D | 2 loops |
| Random points | 1000 | 2D | No significant features |
| MNIST digit patches | 500 patches | 64D | Unknown |

### 4.2 Experiments

**Experiment 1: Error Bound Validation.** Compare certified bounds δ on persistence to empirical differences |PD^{fp32} - PD^{fp64}| (using bottleneck distance or per-feature comparison).

**Experiment 2: Feature Reliability Distribution.** Histogram of reliability scores = persistence / (δ_b + δ_d). What fraction of features are reliable (R > 2)?

**Experiment 3: Precision Sensitivity.** Compute persistence at float64/32/16. Track how many features appear/disappear or change rank. Show bounds predict this.

**Experiment 4: Near-Diagonal Features.** Specifically study features with small persistence. Show correlation between persistence and numerical uncertainty.

**Experiment 5: Curvature Effects.** Vary point cloud density (uniform vs clustered). Show that clustered configurations have higher numerical uncertainty.

### 4.3 Expected Results

1. Error bounds are within 10x of actual per-feature errors for well-separated features.
2. 5-20% of features have reliability score < 2 (numerically borderline).
3. Features with persistence < 5·max(δ) frequently change between precisions.
4. Near-diagonal features (small persistence) are systematically less reliable.
5. Clustered point clouds have 2-3x higher numerical uncertainty than uniform.

**High-Impact Visualizations (< 20 min compute):**
- **Certified persistence diagram** (5 min): Standard PD with horizontal/vertical error bars on each point. Color by reliability (green/yellow/red). Near-diagonal points with large bars are obviously suspect.
- **Precision comparison triptych** (5 min): Three PDs (fp64/fp32/fp16) for same noisy circle. Highlight features that appear/disappear—visually obvious at a glance.
- **Reliability histogram** (3 min): Distribution of persistence/uncertainty. Shade "unreliable" zone. Shows what fraction of "features" are numerical noise.
- **Point cloud with error heat** (5 min): Original 2D point cloud colored by local distance error magnitude. Clustered regions glow red (high error), sparse regions blue (stable).

## 5. Theoretical Contributions Summary

1. **TDA Error Model**: First systematic error analysis for the complete TDA pipeline under finite precision.
2. **Certified Persistence Diagrams**: Per-feature bounds distinguishing real topology from numerical artifacts.
3. **Feature Reliability Criterion**: Principled threshold for when persistence features are numerically meaningful.
4. **Curvature-Aware TDA**: Analysis of how point cloud geometry affects numerical stability.

## 5.1 Usable Artifacts

1. **CertifiedPersistence**: Wrapper around Ripser/Gudhi that returns persistence diagrams with error bars: `certified_persistence(point_cloud) -> (diagram, uncertainties, reliability_flags)`.
2. **TopologyReliabilityFilter**: Post-processor that takes a persistence diagram and filters to only numerically reliable features.
3. **TDA Precision Advisor**: Given point cloud statistics (density, spread), recommends minimum precision for reliable homology computation.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error model derivation | 1 week | None |
| NumGeom-TDA implementation | 1 week | Laptop |
| Persistence experiments | 2 days | 1 hr laptop |
| Reliability analysis | 2 days | 30 min laptop |
| Visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~2 hrs laptop** |
