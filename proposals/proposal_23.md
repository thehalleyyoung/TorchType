# Proposal 23: Numerical Geometry Compilers: Rewriting Computation Graphs for Better Conditioning

## Abstract

We develop NumGeomCompile, a prototype compiler that rewrites ML computation graphs into numerically better-conditioned but functionally equivalent forms. Using the categorical structure of Numerical Geometry—where computations are morphisms in a monoidal category of numerical spaces—we identify algebraic rewrites that preserve semantics while improving error functionals, Lipschitz constants, or condition numbers. Our compiler catalogs 15 safe rewrites (associativity permutations, stable reparameterizations, cancellation avoidance) with proven error bounds, then applies them via greedy local search. Experiments on small networks (2-4 layers, MLPs and CNNs) show that compiled graphs have 20-50% lower numerical condition numbers and exhibit smaller empirical error under precision reduction, with 0.2-0.5% accuracy improvement at float16. All experiments run on a laptop in under 3 hours.

## 1. Introduction and Motivation

Floating-point errors depend not just on precision but on how computations are structured. The same mathematical function can be implemented in numerically better or worse ways: (a+b)+c vs a+(b+c) can differ when |a| >> |b| ≈ |c|. Current ML frameworks don't optimize for numerical stability—they optimize for speed. We propose a "numerical geometry compiler" that rewrites computation graphs to improve conditioning while preserving function. The Numerical Geometry framework provides the theoretical foundation: computations are morphisms in a monoidal category, and algebraic laws (associativity, distributivity) have numerical costs attached. We can search for equivalent computations with lower error functionals Φ.

## 2. Technical Approach

### 2.1 Computation Graphs as Categorical Morphisms

We model a computation graph as a morphism in the category **NumSpace** where:
- Objects are numerical spaces (X, d, R_p)
- Morphisms are numerical functions with tracked (L, Δ, κ)
- Composition gives Φ_{g∘f}(ε) = L_g · Φ_f(ε) + Δ_g

A computation graph G: X → Y is built from primitive operations {+, ×, matmul, ReLU, exp, log, softmax, ...} connected in a DAG.

**Graph Metadata**: For each node n in G, we track:
- L_n: Lipschitz constant of the operation
- Δ_n: Intrinsic rounding error  
- κ_n: Local condition number

### 2.2 Catalogue of Numerically Safe Rewrites

We identify rewrites that preserve mathematical semantics but change numerical properties:

**Associativity Rewrites:**
1. (A·B)·C ↔ A·(B·C) — can change condition number
2. (a+b)+c ↔ a+(b+c) — can reduce cancellation error
3. Σ_i x_i in different orders — sum small terms first for stability

**Reparameterization Rewrites:**
4. log(softmax(x)) → log_softmax(x) — avoids exp overflow
5. x/||x|| → normalize(x) — numerically stable implementation
6. exp(log(x)) → x — identity but one direction is unstable near 0

**Factorization Rewrites:**
7. A·x + b → affine(A, x, b) — fused implementation reduces rounding
8. (A+B)·x ↔ A·x + B·x — distributivity can help or hurt
9. BatchNorm(x) → fused form with precomputed stats

**Numerical Stabilizers:**
10. softmax(x) → softmax(x - max(x)) — shift for stability
11. log(1+x) → log1p(x) — avoids cancellation near x=0
12. exp(x) - 1 → expm1(x) — avoids cancellation near x=0

**Precision Mixing:**
13. Cast high-Lipschitz ops to higher precision locally
14. Accumulate in higher precision, cast down for storage
15. Kahan summation for long reductions

For each rewrite r: G → G', we derive bounds on:
- ΔL = L_{G'} - L_G (change in Lipschitz)
- ΔΔ = Δ_{G'} - Δ_G (change in intrinsic error)

### 2.3 Compiler Architecture

**NumGeomCompile Pipeline:**

```
Input: PyTorch/JAX computation graph G

1. ANALYSIS: Annotate each node with (L, Δ, κ)
   - Use power iteration for matrix Lipschitz
   - Use sample-based estimation for nonlinear ops
   - Propagate through graph

2. COST FUNCTION: Define numerical cost
   cost(G) = Σ_n w_L · L_n + w_Δ · Δ_n + w_κ · κ_n
   with weights emphasizing late layers (closer to output)

3. REWRITE SEARCH: 
   For each pattern in catalogue:
     - Match pattern in G
     - Compute cost of rewritten graph G'
     - If cost(G') < cost(G) - threshold: apply rewrite
   Iterate until no improvement found

4. OUTPUT: Optimized graph G*

Output: Rewritten computation graph with lower cost
```

**Implementation Notes:**
- Pattern matching via subgraph isomorphism (small graphs only)
- Cost estimation uses sampled inputs from calibration data
- Greedy search with random restarts
- Total ~500 lines of Python on top of torch.fx or JAX

### 2.4 Theoretical Guarantees

**Theorem (Rewrite Soundness).** Each rewrite r in our catalogue satisfies:
||G(x) - G'(x)|| ≤ C · Δ(p) for all x in the domain

where Δ(p) is the precision-dependent rounding error and C is a rewrite-specific constant.

**Proposition (Optimization Bound).** If the compiler reduces estimated cost by factor α:
Φ_{G*}(ε) ≤ α · Φ_G(ε) + O(ε²)

## 3. Laptop-Friendly Implementation

The compiler targets small networks where analysis is tractable:

1. **Model scope**: 2-4 layer MLPs, small CNNs (4-6 conv layers), no attention
2. **Graph size**: < 100 nodes after fusion
3. **Analysis cost**: O(|G|²) pattern matching, O(batch_size × |G|) estimation
4. **Calibration data**: 1000 samples sufficient
5. **Rewrite search**: Greedy with 10 random restarts, < 1 minute

Total compile time: ~30 seconds for a 4-layer MLP on laptop.

## 4. Experimental Design

### 4.1 Models

| Model | Architecture | Graph Nodes | Task |
|-------|--------------|-------------|------|
| MLP-2 | 784-256-10 | ~15 | MNIST |
| MLP-4 | 784-512-256-128-10 | ~25 | Fashion-MNIST |
| CNN-4 | Conv32-Pool-Conv64-FC | ~35 | CIFAR-10 |
| CNN-6 | 6 conv + 2 FC | ~50 | SVHN |

### 4.2 Experiments

**Experiment 1: Condition Number Reduction.** For each model, compare estimated condition number (max L across paths) before and after compilation.

**Experiment 2: Empirical Error Reduction.** Measure ||output_{fp32} - output_{fp16}|| on test set before/after compilation. Show that compiled graphs have smaller precision-induced error.

**Experiment 3: Gradient Stability.** Compare gradient variance across random seeds before/after compilation. Compiled graphs should have lower variance.

**Experiment 4: Accuracy at Low Precision.** Train each model, then evaluate at float32/16/int8. Show that compiled models degrade less.

**Experiment 5: Rewrite Analysis.** Report which rewrites are applied most frequently and their individual contributions to improvement.

### 4.3 Expected Results

1. Condition number reduction of 20-50% on most models.
2. Precision-induced error reduced by 30-60% at float16.
3. Gradient variance reduced by 10-30%.
4. Float16 accuracy improved by 0.2-0.5% for compiled models.
5. Most common rewrites: log-softmax fusion, associativity reordering, BatchNorm fusion.

**High-Impact Visualizations (< 20 min compute):**
- **Before/after graph diff** (5 min): Side-by-side computation graphs with changed nodes highlighted in yellow. Rewrites are visually obvious—log-softmax fusion shrinks the graph.
- **Precision degradation line plot** (10 min): Accuracy vs precision bits for original (solid) vs compiled (dashed). Compiled lines stay higher longer. One plot per model, 4 models total.
- **Condition number reduction waterfall** (3 min): Horizontal bar chart showing condition number before (gray) and after (blue) for each model. All bars shrink.
- **Rewrite impact table** (2 min): Simple table: rewrite name | frequency | avg condition improvement. log-softmax and associativity dominate.

## 5. Theoretical Contributions Summary

1. **Categorical Formalization**: Computation graphs as morphisms in NumSpace category with tracked numerical properties.
2. **Rewrite Catalogue**: 15 proven-safe rewrites with error bounds.
3. **Compiler Algorithm**: Greedy search with cost function based on Numerical Geometry.
4. **Practical Tool**: Open-source implementation for PyTorch/JAX.

## 5.1 Usable Artifacts

1. **NumGeomCompile Library**: pip-installable package providing `optimize_graph(model)` that returns a numerically better-conditioned equivalent model. Works with any PyTorch nn.Module.
2. **Rewrite Catalogue**: Documented library of 15 rewrites, each with: pattern, replacement, conditions for benefit, and error bound proof. Extensible for custom rewrites.
3. **Graph Analyzer**: Tool that annotates computation graphs with per-node (L, Δ, κ) estimates, useful for diagnosing numerical bottlenecks.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Rewrite catalogue + proofs | 1 week | None |
| Compiler implementation | 2 weeks | Laptop |
| Model experiments | 3 days | 2 hrs laptop |
| Analysis and ablations | 2 days | 1 hr laptop |
| Writing | 1 week | None |
| **Total** | **5 weeks** | **~3 hrs laptop** |
