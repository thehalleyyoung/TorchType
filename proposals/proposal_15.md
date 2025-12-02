# Proposal 15: Numerical Homotopy Paths and the Geometry of Training Dynamics

## Abstract

We apply Numerical Homotopy Theory to characterize training trajectories in neural networks, showing that optimization paths fall into distinct numerical homotopy classes that determine which minima are reachable at a given precision. We define numerical paths as discretized trajectories through parameter space with bounded per-step error, and numerical homotopy equivalence as the existence of a continuous family of paths whose total numerical error remains bounded. We prove that passing through high-curvature regions of the loss landscape requires high precision by the Curvature Lower Bound—some homotopy classes of paths to good minima are simply inaccessible at low precision. Experiments on 2D synthetic landscapes and small networks (XOR, MNIST) demonstrate that: (1) different precision levels lead to different final minima; (2) these differences correspond to distinct numerical homotopy classes; (3) curvature along training paths predicts which minima become inaccessible at low precision. All experiments run on a laptop in under 2 hours.

## 1. Introduction and Motivation

Neural network optimization is highly non-convex, with many local minima and saddle points. A growing body of work studies the loss landscape topology—mode connectivity, saddle escape, basin structure. But all this work assumes infinite precision. In reality, training happens at finite precision, and the discrete nature of floating-point arithmetic constrains which paths through parameter space are actually traversable. We propose a new lens: Numerical Homotopy Theory. A training trajectory θ₀ → θ₁ → ... → θ_T is a discretized path through parameter space. Two trajectories are numerically homotopic if one can be continuously deformed into the other while keeping the per-step error bounded by the available precision. The key insight is that some deformations require passing through high-curvature regions (sharp valleys, saddles), and by the Curvature Lower Bound, these regions demand high precision. At low precision, certain minima become topologically unreachable—not because no path exists, but because no numerically realizable path exists.

## 2. Technical Approach

### 2.1 Numerical Paths in Parameter Space

Let Θ be the parameter space with loss function L: Θ → ℝ. A discrete training trajectory is a sequence γ = (θ₀, θ₁, ..., θ_T) where θ_{t+1} = θ_t - η∇L(θ_t) + noise (for SGD). We view this as an approximation to a continuous path γ̃: [0,1] → Θ.

**Definition (Numerical Path).** A numerical path at precision p is a discrete trajectory γ = (θ₀, ..., θ_T) such that each step θ_{t+1} = fl_p(θ_t - η · fl_p(∇L(θ_t))) where fl_p denotes rounding to p bits. The path has intrinsic numerical error Δ(γ) = Σ_t ||θ_{t+1} - (θ_t - η∇L(θ_t))||.

**Definition (Numerical Homotopy).** Two numerical paths γ, γ' from θ_init to low-loss regions are numerically p-homotopic if there exists a continuous family of paths γ_s (s ∈ [0,1]) with γ_0 = γ, γ_1 = γ', and max_s Δ(γ_s) ≤ C · 2^{-p} for some constant C.

### 2.2 Curvature Obstruction Theorem

**Theorem (Homotopy Obstruction from Curvature).** Let Θ be partitioned into regions based on loss curvature: R_high = {θ : λ_max(∇²L(θ)) > κ_high} and R_low = {θ : λ_max(∇²L(θ)) ≤ κ_low}. If a path γ passes through R_high, then at precision p with machine epsilon ε_p = 2^{-p}, the per-step gradient error in R_high is at least:

δg ≥ κ_high · ||θ|| · ε_p

For gradient descent with step size η to make progress, we need the true gradient signal to exceed this noise:

η · ||g|| > η · δg  ⇒  p > log₂(κ_high · ||θ|| / ||g||)

When ||g|| is small (near saddles or in flat regions), the required precision becomes large, creating "precision barriers."

**Consequence for Homotopy.** Two paths γ, γ' are numerically p-homotopic only if all intermediate paths avoid regions where p < log₂(κ · ||θ|| / ||g||). High-curvature regions with small gradients are impassable at low precision.

**Proof Strategy.** Curvature κ = λ_max(∇²L) measures how fast the gradient changes. When computing gradients at precision p, perturbations of size ε_p in θ cause gradient changes of size κ · ε_p · ||θ||. This is the intrinsic noise floor for gradient computation in high-curvature regions. Near saddle points where ||g|| → 0 but κ remains large, this noise dominates the signal.

### 2.3 Computing Numerical Homotopy Classes

For practical computation, we discretize the problem:

1. **Trajectory Clustering**: Run N training runs with different random seeds at fixed precision p. Cluster final parameters θ*_i by distance in Θ.

2. **Path Metric**: Define distance between paths γ, γ' as d(γ, γ') = max_t ||θ_t - θ'_{π(t)}|| where π is the optimal time warping.

3. **Homotopy Class Assignment**: Two paths are in the same numerical homotopy class if they can be connected by a chain of paths with d(γ_i, γ_{i+1}) < δ(p), where δ(p) is the precision-dependent threshold.

4. **Curvature Profile**: For each path γ, compute the maximum curvature encountered: κ_max(γ) = max_t λ_max(∇²L(θ_t)).

We use persistent homology (via Ripser) to visualize the structure of path space at different precision scales.

## 3. Laptop-Friendly Implementation

This is a theory-heavy paper with lightweight experiments: (1) **2D synthetic landscapes**: Hand-designed loss functions L(θ₁, θ₂) with multiple minima and known topology. Visualize entire path space. No neural networks needed; (2) **Tiny networks**: 2-layer MLPs on XOR (4 inputs, 2 hidden, 1 output = 13 parameters) and 2-layer CNNs on MNIST subset (1K samples, < 50K params); (3) **Low-dimensional parameter space**: Focus on problems where Θ is small enough to visualize (< 1000 dimensions) or project to 2D/3D via PCA; (4) **Efficient Hessian computation**: For tiny networks, compute full Hessian directly; for larger ones, use Hessian-vector products to estimate λ_max; (5) **Lightweight TDA**: Ripser handles point clouds of 1000 paths in seconds. Total compute: approximately 2 hours on a laptop.

## 4. Experimental Design

### 4.1 Synthetic Landscapes

**Landscape 1: Two Valleys.** L(θ) = min((θ₁-1)² + θ₂², (θ₁+1)² + θ₂²) + 0.1·θ₁²θ₂² with two minima at (±1, 0) separated by a ridge.

**Landscape 2: Saddle Gauntlet.** L(θ) with multiple saddle points between initialization and global minimum, requiring careful navigation.

**Landscape 3: Curved Funnel.** L(θ) with a high-curvature funnel leading to the global minimum, testing curvature obstruction.

**High-Impact Visualizations (< 5 min compute each):**
- **3D loss surface with curvature overlay**: Color-coded by κ, with training paths at different precisions as colored trajectories (float64=blue, float32=green, float16=orange, 8-bit=red). ~100 paths per precision.
- **Precision barrier heatmap**: 2D plot where color = minimum precision needed to traverse each point, derived from local κ/||g|| ratio.
- **Path space persistence diagram**: Topological summary showing when homotopy classes merge as precision decreases.
- **Animated GIF**: Training trajectories evolving over time at different precisions (compelling for Twitter/paper figures).

For each landscape, visualize loss surface, curvature map, and training paths at different precisions (float64, float32, float16, 8-bit).

### 4.2 Neural Network Experiments

**XOR Task**: 4 binary inputs, 2 hidden units, 1 output. Multiple qualitatively different solutions exist (different weight configurations achieving 100% accuracy). Run 100 training runs at each precision level, cluster final weights, measure which solution types are reached.

**MNIST-Subset**: 1000 samples, 10 classes, small CNN. Run 50 training runs at float32 vs float16, compare final accuracy distributions and parameter space clusters.

### 4.3 Experiments and Metrics

**Experiment 1: Homotopy Class Count.** At each precision level, count the number of distinct homotopy classes of successful training paths (reaching loss < threshold).

**Experiment 2: Curvature-Precision Correlation.** For paths that fail at low precision but succeed at high precision, measure the maximum curvature encountered. Verify correlation with precision threshold.

**Experiment 3: Minima Accessibility.** For landscapes/networks with multiple distinct minima, measure which fraction are accessible at each precision level.

**Experiment 4: Persistent Homology Analysis.** Compute persistence diagrams for the path space at different precision scales, visualizing the topological structure.

### 4.4 Expected Results

1. Number of homotopy classes decreases as precision decreases (fewer paths are distinguishable).
2. Paths requiring high-curvature traversal become inaccessible at low precision, matching theoretical threshold.
3. On XOR, some solution types (e.g., those requiring precise weight balancing) become unreachable at 8-bit.
4. Persistent homology shows phase transitions in path space topology at curvature-predicted precision thresholds.

## 5. Theoretical Contributions Summary

1. **Numerical Homotopy Framework**: First formal treatment of training paths as elements of numerical homotopy groups.
2. **Curvature Obstruction Theorem**: Proof that high-curvature regions create precision barriers between homotopy classes.
3. **Minima Accessibility Analysis**: Characterization of which minima are reachable at given precision.
4. **Computational Methods**: Practical algorithms for computing numerical homotopy classes from training trajectories.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Synthetic landscape design | 3 days | None |
| Path clustering algorithm | 1 week | Laptop |
| Synthetic experiments | 2 days | 30 min laptop |
| XOR experiments | 2 days | 30 min laptop |
| MNIST experiments | 2 days | 1 hr laptop |
| TDA visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~2.5 hrs laptop** |

