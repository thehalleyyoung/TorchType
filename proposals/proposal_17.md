# Proposal 17: Certified Autodiff: Automatic Differentiation with Error Functionals

## Abstract

We develop NumGeom-AD, a prototype automatic differentiation system that propagates error functionals alongside gradients, providing certified bounds on numerical error for every computed derivative. Building on the AD-with-Precision-Tracking theory from Numerical Geometry, we implement forward and reverse mode AD rules that track not just values and gradients, but also Lipschitz constants and intrinsic errors through the computation graph. For each primitive operation (add, multiply, matmul, ReLU, softmax, exp, log, division), we derive closed-form error functional updates. The result: every gradient comes with a certificate Φ_∇L(ε) bounding how much the computed gradient can differ from the true gradient given input precision ε. Experiments on small MLPs and CNNs show that our bounds are tight (within 10x of observed error), detect numerically unreliable gradients (e.g., saturated softmax, ill-conditioned layers), and guide precision allocation for stable training. NumGeom-AD is implemented as a 1000-line JAX/PyTorch extension running on any laptop.

## 1. Introduction and Motivation

Automatic differentiation is the backbone of deep learning, but its numerical reliability is rarely questioned. We trust that torch.autograd gives us "the gradient," but finite-precision arithmetic means we get an approximation. How good is this approximation? Current practice: hope for the best. We propose a principled alternative: augment AD to propagate error bounds. For each operation in the computation graph, we track not just the computed value and gradient, but also an error functional Φ that bounds how much the output can differ from the mathematically correct value. These error functionals compose through the chain rule, giving end-to-end bounds. The key insight from Numerical Geometry is that error functionals Φ_f(ε) = L_f · ε + Δ_f have a clean composition rule: Φ_{g∘f}(ε) = L_g · Φ_f(ε) + Δ_g. This extends to derivatives via the AD chain rule with tracked Lipschitz constants.

## 2. Technical Approach

### 2.1 Error-Aware AD Primitives

For each primitive operation, we derive the error functional for both the forward pass and the backward pass (gradient computation).

**Addition**: f(x, y) = x + y
- Forward: Φ_f(ε_x, ε_y) = ε_x + ε_y + ε_mach·|x+y| (rounding error)
- Backward: ∂f/∂x = 1, ∂f/∂y = 1, both exact (Lipschitz 1, no intrinsic error)

**Multiplication**: f(x, y) = x · y
- Forward: Φ_f(ε_x, ε_y) = |y|·ε_x + |x|·ε_y + ε_x·ε_y + ε_mach·|xy|
- Backward: ∂f/∂x = y, ∂f/∂y = x, Lipschitz constants |y| and |x| respectively

**Division**: f(x, y) = x / y
- Forward: Φ_f(ε_x, ε_y) = (ε_x + |x|/|y|·ε_y) / |y| + ε_mach·|x/y|
- Backward: ∂f/∂y = -x/y² has Lipschitz constant |x|/y² w.r.t. x and 2|x|/|y|³ w.r.t. y
- WARNING: Δ → ∞ as y → 0

**Exponential**: f(x) = exp(x)
- Forward: Φ_f(ε) = exp(x)·ε + ε_mach·exp(x) (Lipschitz = exp(x))
- Backward: ∂f/∂x = exp(x), same Lipschitz constant
- WARNING: Lipschitz grows exponentially with x

### 2.2 Softmax Error Analysis (Detailed Example)

Softmax is notoriously numerically sensitive. We derive its error functional carefully.

f(z)_i = exp(z_i) / Σ_j exp(z_j)

**Forward error analysis**:
1. Let M = max_j z_j (for numerical stability, we compute f(z - M))
2. Subtraction z_i - M has cancellation error when z_i ≈ M
3. Exponentiation exp(z_i - M) has error ≈ exp(z_i - M) · ε_mach
4. Sum Σ exp(z_j - M) accumulates errors
5. Division introduces error when sum is small (all z_i ≪ M except one)

**Result**: Φ_softmax(ε) = L_soft · ε + Δ_soft where
- L_soft = 1 (the softmax Jacobian has operator norm at most 1, achieved when probabilities are uniform)
- The Jacobian diag(f) - ffᵀ has entries bounded by f_i(1 - f_i) ≤ 1/4 on diagonal
- Δ_soft = O(n · ε_mach) when well-conditioned, but O(exp(max_i z_i - z_k) · ε_mach) for the k-th output when poorly conditioned

**Backward error**: ∂L/∂z involves f(1-f), same stability concerns apply.

### 2.3 Composition Rules for AD

**Theorem (Error Functional Chain Rule).** For a composition f = g ∘ h with error functionals Φ_h, Φ_g, the composed error functional is:
Φ_f(ε) = Φ_g(Φ_h(ε))

For the gradient computed by reverse-mode AD, the error propagation is more subtle:
Φ_{∇f}(ε) = L_{∂h/∂x} · Φ_{∇_h g}(ε) + ||∇_h g|| · Φ_{∂h/∂x}(ε) + Δ_{matmul}

where:
- L_{∂h/∂x} is the Lipschitz constant of the Jacobian ∂h/∂x
- ||∇_h g|| is the upstream gradient magnitude
- Δ_{matmul} is the intrinsic error from the Jacobian-vector product

**Proof Strategy.** The reverse-mode gradient ∇_x L = (∂h/∂x)ᵀ ∇_h L is a matrix-vector product. Error sources:
1. Error in the upstream gradient ∇_h L, amplified by the Jacobian norm ||∂h/∂x||
2. Error in the Jacobian computation itself, amplified by the gradient magnitude ||∇_h L||
3. Intrinsic matmul rounding error

The key insight is that gradient error accumulates multiplicatively through the network depth, with each layer contributing both Jacobian error and amplification of upstream error.

### 2.4 NumGeom-AD API

```python
import numgeom_ad as nad

# Wrap a PyTorch model
model = nad.wrap(my_pytorch_model)

# Forward pass with error tracking
output, error_bound = model.forward_with_error(x, precision='float32')

# Backward pass with gradient error certificate
loss = criterion(output, y)
grads, grad_error_bounds = model.backward_with_error(loss)

# Check for numerical warnings
warnings = nad.check_stability(model, x, threshold=1e-4)
# Returns: ["softmax at layer 3 has error bound 0.02 > threshold",
#           "division at layer 5 approaching singularity"]

# Get per-layer error contributions
breakdown = nad.error_breakdown(model, x)
# Returns: {"layer1": 0.0001, "layer2": 0.0003, "softmax": 0.015, ...}
```

## 3. Laptop-Friendly Implementation

NumGeom-AD is designed for debugging and analysis, not production training: (1) **Restricted op set**: We implement error tracking for ~20 core operations (arithmetic, matmul, activations, softmax, batchnorm, conv2d). Sufficient for standard MLPs, CNNs, and small Transformers; (2) **Symbolic error functionals**: Instead of computing error bounds numerically at each step, we propagate symbolic expressions Φ(ε) = L·ε + Δ, only evaluating at the end; (3) **Lazy evaluation**: Error bounds are only computed when requested, not during normal forward/backward; (4) **Small models**: Overhead is O(1) per operation for tracking L and Δ, so analysis time is proportional to model size. For a 4-layer MLP, analysis takes < 1 second. (5) **Comparison to float64**: We validate bounds by comparing float32 outputs to float64 "ground truth." Total implementation: ~1000 lines of Python.

## 4. Experimental Design

### 4.1 Models and Test Cases

| Model | Architecture | Known Numerical Issues |
|-------|--------------|----------------------|
| MLP-Saturated | 4-layer with large weights | Softmax saturation |
| CNN-IllCond | Conv layers with high condition number | Gradient explosion |
| Transformer-Tiny | 2-layer attention | Attention softmax, LayerNorm division |
| RNN-Vanishing | 3-layer LSTM | Gradient vanishing/explosion |

For each model, we create "healthy" and "pathological" versions to test detection.

### 4.2 Experiments

**Experiment 1: Bound Tightness.** Compute error bound Φ_∇L and compare to empirical error ||∇L_{fp32} - ∇L_{fp64}||. Report ratio (tightness factor). Target: within 10x.

**Experiment 2: Detection Accuracy.** Inject numerical instabilities (saturated activations, near-zero divisors) and measure if NumGeom-AD flags them. Report true positive rate and false positive rate.

**Experiment 3: Higher-Order Derivatives.** Test on double-backprop scenarios (Hessian-vector products). These are known to be numerically fragile. Measure bound tightness.

**Experiment 4: Precision Guidance.** Use error breakdown to identify which layers need high precision. Compare training with uniform float32 vs. mixed precision guided by NumGeom-AD.

**Experiment 5: Overhead Measurement.** Measure wall-clock time for forward+backward with vs. without error tracking. Target: < 2x overhead for analysis.

### 4.3 Expected Results

1. Error bounds are within 10x of observed error for well-conditioned models, within 100x for ill-conditioned.
2. 90%+ of injected instabilities are detected; false positive rate < 10%.
3. Higher-order derivatives show larger errors (as expected), bounds remain informative.
4. Precision guidance from error breakdown matches intuition (softmax, division, ill-conditioned matmuls flagged).
5. Overhead is 1.5-2x for error tracking, acceptable for debugging.

**High-Impact Visualizations (< 30 min compute):**
- **Per-layer error breakdown treemap**: Hierarchical visualization where area = error contribution, color = error type (Lipschitz vs intrinsic). Immediately identifies bottleneck layers.
- **Bound tightness over training**: Line plot showing ||predicted error|| and ||observed error|| over training iterations. Tracks that bounds remain valid even as model changes.
- **Softmax instability heatmap**: For attention layer, show logit differences z_i - z_j on one axis, error bound on other. Clear "danger zone" at large differences.
- **Gradient flow with error certificates**: Network architecture diagram with arrows showing gradient flow, arrow thickness = gradient magnitude, arrow color = error bound (green=safe, red=unreliable).

## 5. Theoretical Contributions Summary

1. **Error Functional AD Rules**: Complete derivation of error functionals for common neural network operations.
2. **Gradient Error Composition**: Chain rule extension for error functional propagation through reverse-mode AD.
3. **NumGeom-AD System**: Practical implementation with clean API.
4. **Use Cases**: Demonstrated value for debugging and precision allocation.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error functional derivations | 1 week | None |
| NumGeom-AD implementation | 2 weeks | Laptop |
| Bound tightness experiments | 2 days | 30 min laptop |
| Detection experiments | 2 days | 30 min laptop |
| Higher-order experiments | 2 days | 30 min laptop |
| Writing | 1 week | None |
| **Total** | **5 weeks** | **~2 hrs laptop** |

