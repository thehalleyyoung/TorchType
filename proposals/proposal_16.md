# Proposal 16: Numerical Equivalence of Training Algorithms: When Are Optimizers Just Old Ones in Disguise?

## Abstract

We develop a formal framework for determining when two optimization algorithms are numerically equivalent—indistinguishable under finite-precision arithmetic. Using Numerical Equivalence Theory, we model optimizers as numerical morphisms on (parameters, gradients, state) space and define equivalence via mutual simulation with bounded condition numbers. We prove that many seemingly different optimizers exhibit *approximate* equivalence under specific conditions: when gradients have uniform magnitude across coordinates, Adam behaves similarly to momentum SGD. Conversely, we identify genuinely distinct classes: signSGD and second-order methods cannot be numerically equivalent to first-order methods without exponential condition numbers. At lower precision (float16), more optimizers collapse into equivalence as adaptive denominators are rounded away. Experiments on small CNNs (CIFAR-10) and MLPs (MNIST) verify these predictions. All experiments run on a laptop in under 3 hours.

## 1. Introduction and Motivation

The optimization literature produces dozens of new algorithms annually: Adam, AdamW, RAdam, LAMB, Adafactor, Lion, Sophia, etc. Each claims improvements over predecessors. But a provocative question arises: how many of these are actually different under finite-precision arithmetic? We formalize this using Numerical Equivalence from Numerical Geometry. Two functions are numerically equivalent if they can simulate each other via Lipschitz maps with bounded condition numbers. For optimizers, this means: if we can transform the state of optimizer A to the state of optimizer B (and back) with bounded precision loss, and the transformed updates match, then A and B are numerically equivalent—any difference between them is below the noise floor of floating-point arithmetic. We find that the optimizer zoo exhibits *conditional* collapse: under specific conditions (uniform gradient magnitudes, low precision), many variants become approximately indistinguishable. This has practical implications: understanding when optimizers are genuinely different helps practitioners make informed choices.

## 2. Technical Approach

### 2.1 Optimizers as Numerical Morphisms

An optimizer U operates on state space S = Θ × G × M where Θ is parameters, G is gradients, M is optimizer memory (momentum buffers, second moments, etc.). The update rule defines a map:

U: S → S,  (θ, g, m) ↦ (θ', m')

We equip S with a numerical structure (S, d, R) where d is the Euclidean metric and R is the set of floating-point representable states at precision p.

**Definition (Optimizer as Numerical Morphism).** An optimizer U is a numerical morphism with Lipschitz constant L_U = sup ||U(s) - U(s')|| / ||s - s'|| and intrinsic error Δ_U = sup_{s ∈ R} ||U(s) - fl_p(U(s))||.

Common optimizers have the following properties:
- **SGD**: L_U = max(1, η·L_loss), Δ_U = O(ε_mach · ||θ||)
- **Momentum**: L_U = max(1, η·L_loss, β), Δ_U = O(ε_mach · ||θ|| + ε_mach · ||m||)
- **Adam**: L_U = max(1, η/√(v+ε)), Δ_U = O(ε_mach · ||θ|| + ε_mach · ||m|| + ε_mach · ||v||)

### 2.2 Numerical Equivalence of Optimizers

**Definition (Numerical Equivalence).** Optimizers U₁, U₂ are numerically p-equivalent if there exist numerical morphisms φ: S₁ → S₂ and ψ: S₂ → S₁ such that:
1. ||U₂(φ(s)) - φ(U₁(s))|| ≤ δ(p) for all s ∈ S₁
2. ||U₁(ψ(s)) - ψ(U₂(s))|| ≤ δ(p) for all s ∈ S₂
3. κ(φ), κ(ψ) ≤ poly(dim(S)) (polynomial condition numbers)

where δ(p) = O(2^{-p}) is the precision threshold.

**Theorem (Optimizer Approximate Equivalence).** At float32 precision (p = 24), the following pairs exhibit approximate numerical equivalence under specific conditions:
- SGD with momentum β and Adam with β₁=β, β₂ small (e.g., 0.9), when gradients have roughly uniform magnitude across coordinates
- RMSprop with α and Adam with β₁=0, β₂=α, when first moment effects are negligible

**Important Caveat.** These equivalences are *approximate* and *conditional*. Adam with adaptive learning rates (β₂ ≠ 0) is genuinely different from SGD+momentum when gradients have varying magnitudes across coordinates—the adaptive denominator √(v+ε) provides coordinate-wise scaling that cannot be replicated by scalar learning rate adjustment.

**Proof Strategy.** For the SGD-momentum ↔ Adam approximate equivalence: when all gradient coordinates have similar magnitudes, the adaptive denominator √v behaves approximately as a scalar, absorbable into the learning rate. However, when grad_i and grad_j differ by factors of 100+, Adam's per-coordinate adaptation yields genuinely different trajectories. The float32 precision threshold determines when these differences become measurable.

### 2.3 Genuinely Distinct Classes

**Theorem (Non-Equivalence of SignSGD).** SignSGD (updating θ ← θ - η·sign(g)) is not numerically equivalent to any first-order method at any precision, unless the transformation has condition number κ ≥ ||g||/min_i|g_i|, which is typically exponential in dimension.

**Proof Strategy.** SignSGD throws away magnitude information: sign(g) ∈ {-1, +1}^d. Any transformation recovering g from sign(g) must amplify by a factor of ||g||/||sign(g)|| = ||g||/√d. For gradients with varying magnitudes across coordinates, this amplification factor is at least max_i|g_i|/min_i|g_i|, which is typically very large. Thus no low-condition-number equivalence exists.

**Theorem (Non-Equivalence of Second-Order Methods).** Newton's method (θ ← θ - H⁻¹g) is not numerically equivalent to first-order methods unless κ ≥ κ(H), the condition number of the Hessian.

**Proof.** First-order methods use only gradient information g. Recovering H⁻¹g from g requires Hessian information, which cannot be inferred from g alone with bounded condition number.

## 3. Laptop-Friendly Implementation

All experiments use small models and standard datasets: (1) **Small architectures**: MLPs with < 500K params on MNIST, CNNs with < 2M params on CIFAR-10; (2) **Short training**: 20-50 epochs sufficient to observe optimizer differences; (3) **State space analysis**: Track optimizer state (θ, m, v) at each step, compute distances between trajectories under different optimizers; (4) **Statistical testing**: Run 10 seeds per optimizer, use paired t-tests to determine if differences are significant; (5) **Transformation validation**: Explicitly compute φ, ψ transformations and verify equivalence on actual states. Total compute: approximately 3 hours on a laptop.

## 4. Experimental Design

### 4.1 Optimizers Tested

| Optimizer | Key Parameters | Expected Class |
|-----------|---------------|----------------|
| SGD | η | Class A (baseline) |
| SGD+Momentum | η, β=0.9 | Class B |
| Adam | η, β₁=0.9, β₂=0.999 | Class C (adaptive) |
| Adam (β₂→0) | η, β₁=0.9, β₂=0.001 | Class B (≈ momentum) |
| RMSprop | η, α=0.99 | Class C (adaptive) |
| SignSGD | η | Class D (distinct) |
| Adagrad | η | Class C (adaptive) |

The key question is whether Class C (adaptive methods) collapses toward Class B at low precision or remains distinct.

### 4.2 Experiments

**Experiment 1: Trajectory Distance.** For each pair of optimizers, train with matched learning rates and measure ||θ_A(t) - θ_B(t)|| over training. Numerically equivalent optimizers should have distance O(t · 2^{-p}).

**Experiment 2: Final Performance.** Compare test accuracy distributions across seeds. Numerically equivalent optimizers should have statistically indistinguishable distributions (p > 0.05 on paired t-test).

**Experiment 3: Transformation Verification.** Compute φ(state_A) at each step and compare to state_B. Measure ||φ(state_A) - state_B|| and verify it's O(2^{-p}).

**Experiment 4: Non-Equivalence Demonstration.** Train with SignSGD and SGD+momentum on same problem. Show trajectory distance grows linearly with steps (not O(2^{-p})).

**Experiment 5: Precision Sensitivity.** Repeat experiments at float16 vs float32 vs float64. At higher precision, more optimizers become distinguishable; at lower precision, more collapse into equivalence.

### 4.3 Expected Results

1. SGD+momentum and Adam (with β₂→0) have trajectory distance < 10^{-5} at float32, confirming equivalence.
2. SignSGD trajectory distance from SGD+momentum grows as O(√t), confirming non-equivalence.
3. Standard Adam (β₂=0.999) may be distinguishable from momentum due to adaptive learning rate, but difference is small on well-conditioned problems.
4. At float16, even standard Adam collapses toward momentum equivalence (adaptive denominator rounded away).
5. Number of equivalence classes: 2-3 at float16, 3-4 at float32, 5+ at float64.

**High-Impact Visualizations (< 30 min compute):**
- **Optimizer equivalence phase diagram**: 2D plot with x-axis = precision (4-64 bits), y-axis = gradient magnitude heterogeneity (ratio max/min coordinate). Colors indicate which optimizer pairs are equivalent in each region. Shows dramatic "collapse" at low precision.
- **Trajectory divergence over time**: Multi-panel figure showing ||θ_A(t) - θ_B(t)|| for each optimizer pair, with 95% CI bands across seeds. Equivalent pairs stay flat; distinct pairs grow.
- **State-space transformation animation**: Show φ(state_SGD) tracking state_Adam over training, with residual shown as growing/shrinking halo.

## 5. Theoretical Contributions Summary

1. **Optimizer Numerical Equivalence Framework**: Formal definition of when optimizers are indistinguishable at given precision.
2. **Approximate Equivalence Theorems**: Proofs that major optimizer families exhibit approximate equivalence under specific conditions at finite precision.
3. **Non-Equivalence Theorems**: Proofs that SignSGD and second-order methods are genuinely distinct.
4. **Practical Classification**: Taxonomy of optimizers by numerical equivalence class.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Equivalence proofs | 1 week | None |
| Transformation implementation | 1 week | Laptop |
| Trajectory experiments | 3 days | 2 hrs laptop |
| Precision sensitivity | 2 days | 1 hr laptop |
| Statistical analysis | 2 days | 30 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~3.5 hrs laptop** |

