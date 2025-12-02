# Proposal 30: Numerical Complexity Classes: A Curvature-Based Taxonomy of Learning Tasks

## Abstract

We propose a novel taxonomy of machine learning tasks based on their numerical complexity—the intrinsic difficulty of computing solutions in finite precision. Just as computational complexity classifies problems by time/space requirements, numerical complexity classifies by precision requirements and curvature characteristics. We define complexity classes based on error functional scaling: NC-Poly (polynomial error growth), NC-Exp (exponential error growth), and NC-Crit (critical precision thresholds). We develop theoretical criteria for classification and validate empirically on a suite of standard ML tasks. Our key finding: task numerical complexity correlates with but is distinct from computational complexity—some computationally easy tasks are numerically hard, and vice versa. Experiments on toy tasks (linear regression, logistic regression, small neural nets) demonstrate the taxonomy and show that numerical complexity predicts practical precision requirements. All experiments run on a laptop in under 3 hours.

## 1. Introduction and Motivation

Machine learning practice implicitly assumes that "float32 is enough" for most tasks. But why? Some tasks—ill-conditioned linear systems, computing small eigenvalues—require higher precision. Others are numerically robust even at float16. We lack a principled framework for understanding which tasks need which precision. We address this by developing Numerical Complexity Theory for ML. Inspired by computational complexity (P, NP, etc.), we classify tasks by their precision requirements and numerical sensitivity. The key insight: a task's numerical complexity depends on the curvature of its loss landscape, the condition number of its data, and the stability of its optimization algorithm. We formalize this using the Numerical Geometry framework.

## 2. Technical Approach

### 2.1 Numerical Complexity Classes

**Definition (Error Functional Class).** A learning task T with algorithm A has error functional class determined by how Φ_A(ε) scales with problem size n and precision ε:

**NC-Poly (Numerically Polynomial):**
Φ_A(ε) = O(poly(n) · ε)

Examples: Well-conditioned linear regression, low-curvature neural nets.

**NC-Exp (Numerically Exponential):**
Φ_A(ε) = O(exp(n) · ε) or O(poly(n) · ε^{-1})

Examples: Ill-conditioned systems, training near sharp minima.

**NC-Crit (Numerically Critical):**
∃ ε_crit such that for ε > ε_crit, the algorithm fails entirely.

Examples: Overflow/underflow problems, degeneracy issues.

### 2.2 Characterizing Numerical Complexity

**Theorem (Curvature-Complexity Correspondence).** For an optimization-based learning task with loss L(θ):

1. If κ_max = max |eigenvalue of ∇²L| is bounded: Task is NC-Poly
2. If κ_max grows polynomially with n: Task is NC-Poly with large constants
3. If κ_max grows exponentially or ∇²L has near-zero eigenvalues: Task is NC-Exp

**Theorem (Condition Number Determines Precision).** For a task with condition number κ, achieving relative error δ requires precision ε satisfying:

ε < δ / κ

This gives the "precision budget" for any target accuracy.

### 2.3 Classification Criteria

**Algorithm: ClassifyNumericalComplexity**

```
Input: Learning task T (dataset, model, loss, algorithm)

1. CURVATURE analysis:
   - Estimate κ_max from Hessian samples
   - Check if κ_max is bounded, poly, or exp in n

2. CONDITIONING analysis:
   - Compute condition number of key matrices (data, Hessian)
   - Check for near-zero eigenvalues

3. STABILITY analysis:
   - Run algorithm at multiple precisions
   - Measure error scaling with ε

4. CLASSIFY:
   - NC-Poly: Bounded curvature, good conditioning, linear error scaling
   - NC-Exp: High curvature or poor conditioning, superlinear scaling
   - NC-Crit: Algorithm failure at some precision threshold

Output: Complexity class + supporting metrics
```

### 2.4 Subclasses and Refinements

**NC-Poly Subclasses:**
- NC-Poly-Easy: Low curvature (κ < 10), works at float16
- NC-Poly-Moderate: Medium curvature (10 < κ < 1000), needs float32
- NC-Poly-Hard: High but bounded curvature (κ > 1000), needs float64

**NC-Exp Subclasses:**
- NC-Exp-Recoverable: High precision recovers, error controlled
- NC-Exp-Unstable: Error growth makes task impractical at any precision

### 2.5 Relationships to Standard Complexity

**Proposition (Computational vs Numerical Complexity).**
1. Computationally easy, numerically easy: Linear regression (well-conditioned)
2. Computationally easy, numerically hard: Linear regression (ill-conditioned)
3. Computationally hard, numerically easy: Some discrete optimization problems
4. Computationally hard, numerically hard: Many NP-hard continuous problems

This shows numerical complexity is an orthogonal axis to computational complexity.

## 3. Laptop-Friendly Implementation

Numerical complexity classification is lightweight:

1. **Tasks**: Small-scale versions of standard ML problems
2. **Curvature estimation**: Hessian-vector products via finite differences
3. **Multi-precision runs**: fp64, fp32, fp16 comparisons
4. **Small models**: < 10K parameters for tractable Hessian analysis

Total experiment time: approximately 2-3 hours on a laptop.

## 4. Experimental Design

### 4.1 Task Suite

| Task | Dataset | Model | Expected Class |
|------|---------|-------|----------------|
| Linear Regression (well-conditioned) | Synthetic (κ=10) | Linear | NC-Poly-Easy |
| Linear Regression (ill-conditioned) | Synthetic (κ=10^6) | Linear | NC-Exp |
| Logistic Regression | MNIST-binary | Logistic | NC-Poly-Moderate |
| MLP (small) | MNIST-subset | 2-layer MLP | NC-Poly-Moderate |
| MLP (deep) | MNIST-subset | 5-layer MLP | NC-Poly-Hard |
| Neural Net (sharp minimum) | Synthetic | MLP + high LR | NC-Exp |
| Eigenvalue computation | Synthetic matrices | Power method | NC-Exp (for small eigs) |

### 4.2 Experiments

**Experiment 1: Classification Validation.** For each task, run classification algorithm. Compare predicted class to empirical precision requirements.

**Experiment 2: Curvature-Complexity Correlation.** Measure κ_max for each task. Plot against empirical error scaling. Show correlation.

**Experiment 3: Precision Budget Verification.** Predict required precision from condition number. Verify by running at different precisions.

**Experiment 4: Cross-Task Comparison.** Systematic comparison of error scaling across all tasks. Visualize task placement in numerical complexity space.

**Experiment 5: Failure Mode Analysis.** For NC-Crit tasks, identify the precision threshold where failure occurs. Characterize failure mode (overflow? underflow? divergence?).

### 4.3 Expected Results

1. Classification algorithm correctly identifies task class for 80%+ of tasks.
2. Strong correlation (r > 0.8) between κ_max and empirical error scaling.
3. Precision budget predictions are accurate within 2 bits of actual requirements.
4. Clear separation in numerical complexity space between task categories.
5. NC-Crit tasks have sharp failure thresholds, predictable from analysis.

**High-Impact Visualizations (< 20 min compute):**
- **Numerical complexity map** (5 min): 2D scatter with x = log(curvature), y = log(condition number). Each task is a labeled point, colored by empirical precision class (green/yellow/red). Taxonomy is visually obvious.
- **Error scaling comparison** (10 min): One subplot per task showing error vs precision. NC-Poly tasks are straight lines, NC-Exp are curved, NC-Crit show sharp cliffs. Visual proof of taxonomy.
- **Precision prediction accuracy table** (2 min): Simple table: task | predicted precision | actual precision | correct?. High accuracy validates the framework.
- **Computational vs numerical complexity quadrant** (3 min): 2×2 grid placing tasks by computational/numerical difficulty. Shows the two axes are independent.

## 5. Theoretical Contributions Summary

1. **Numerical Complexity Framework**: First formal taxonomy of ML tasks by precision requirements.
2. **Curvature-Complexity Correspondence**: Links loss landscape geometry to numerical hardness.
3. **Precision Budget Formula**: Practical prediction of required precision from condition number.
4. **Orthogonality Result**: Shows numerical complexity is independent of computational complexity.

## 5.1 Usable Artifacts

1. **NumericalComplexityClassifier**: Python tool that takes a PyTorch model + dataset and outputs its numerical complexity class (NC-Poly-Easy/Moderate/Hard, NC-Exp, NC-Crit) with confidence scores.
2. **PrecisionBudgetCalculator**: Function `required_precision(model, data, target_error)` that returns minimum precision for achieving target accuracy.
3. **Task Complexity Database**: Pre-classified complexity ratings for 20+ common ML tasks/architectures, usable as a reference guide for practitioners.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Framework development | 1.5 weeks | None |
| Classification algorithm | 1 week | Laptop |
| Task suite experiments | 3 days | 2 hrs laptop |
| Cross-task analysis | 2 days | 45 min laptop |
| Visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **5 weeks** | **~3 hrs laptop** |

## 7. Broader Implications

This taxonomy has practical implications:
1. **Precision selection**: Given a task, predict whether float16/32/64 is appropriate
2. **Hardware design**: Identify which tasks benefit from high-precision accelerators
3. **Algorithm design**: Guide development of numerically stable algorithms for hard classes
4. **Curriculum learning**: Train first on NC-Poly-Easy tasks, then NC-Poly-Hard

The numerical complexity perspective complements existing ML theory and provides actionable guidance for practitioners.
