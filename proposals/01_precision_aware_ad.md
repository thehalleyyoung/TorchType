# Project 1: Precision-Aware Automatic Differentiation Library

## Transformer Application: Debug Attention NaNs During Training

**The Problem:** Training transformers frequently produces NaN/Inf values, especially in attention layers during early training or when using float16/bfloat16. Currently, debugging these requires trial-and-error insertion of `.float()` calls.

**Our Solution:** Track precision requirements through the transformer computation graph automatically, identifying exactly which operations (QK^T computation, softmax, LayerNorm) need higher precision before training begins.

**Concrete Use Cases:**
1. Before training: "Your attention layer needs float32 for the softmax, but FFN can stay in bfloat16"
2. During training: "Step 1247: attention.layers.12.softmax curvature spiked, increase precision or reduce LR"
3. Debugging NaNs: "The NaN originated in layer 8's attention softmax due to large logits"

---

## Executive Summary

Build a JAX/PyTorch extension that tracks precision requirements through computation graphs using the curvature bounds from Theorem 5.7 of the HNF paper. The system computes per-operation numerical curvature and propagates precision requirements through compositions, outputting recommendations like "layer 12 needs float64, layers 1-11 are fine with float16."

---

## Theoretical Foundation

### The Core Insight

Every differentiable operation $f: A \to B$ has an associated **curvature** that measures how quickly the linearization changes:

$$\kappa_f^{\mathrm{curv}} = \sup_{x \in D} \|D^2f(x)\| \cdot \|Df(x)^{-1}\|^2$$

This curvature directly determines precision requirements. From Theorem 5.7:

$$p_{\min} \geq \log_2\left(\frac{c \cdot \kappa_f^{\mathrm{curv}} \cdot D^2}{\varepsilon}\right)$$

where $D$ is the domain diameter and $\varepsilon$ is the target accuracy.

### Why This Matters

Current practice: Run model in float16, observe NaNs, add `.float()` calls until it works.

With this tool: Know *before training* which layers need higher precision, and why.

---

## Technical Approach

### 1. PrecisionTensor Wrapper

Create a tensor class that tracks precision metadata alongside values:

```python
class PrecisionTensor:
    def __init__(self, data, curvature_bound=1.0, precision_req=None):
        self.data = data  # underlying tensor
        self.curvature = curvature_bound  # κ^curv for operations producing this
        self.precision_req = precision_req  # minimum bits needed
        self.domain_diameter = None  # estimated from data statistics
```

### 2. Curvature Database for Primitives

For each PyTorch/JAX primitive, precompute or derive curvature bounds:

| Operation | Curvature $\kappa^{\mathrm{curv}}$ | Notes |
|-----------|-----------------------------------|-------|
| `matmul(A, B)` | $O(\kappa(A) \cdot \kappa(B))$ | Condition numbers of inputs |
| `exp(x)` | $e^{2\|x\|_\infty}$ | Exponentially bad for large inputs |
| `log(x)` | $1/x_{\min}^2$ | Bad near zero |
| `softmax(x)` | $O(e^{\|x\|_\infty})$ | Dominated by max element |
| `relu(x)` | $0$ | Piecewise linear, no curvature |
| `sigmoid(x)` | $O(1)$ | Bounded curvature |
| `layer_norm(x)` | $O(1/\sigma_{\min}^2)$ | Bad when variance is small |
| `attention(Q,K,V)` | $O(e^{\|QK^T\|_\infty / \sqrt{d}})$ | Softmax dominates |

### 3. Composition Rules

When composing $g \circ f$, the curvature propagates:

$$\kappa_{g \circ f}^{\mathrm{curv}} \leq \kappa_g^{\mathrm{curv}} \cdot L_f^2 + \kappa_f^{\mathrm{curv}} \cdot \|Dg\|$$

Precision requirements compose as:

$$p_{g \circ f} \geq \max(p_f, p_g) + \log_2(1 + \kappa_g \cdot L_f^2 / \kappa_f)$$

### 4. Tracing System

Hook into PyTorch's autograd or use FX tracing:

```python
def trace_precision_requirements(model, sample_input):
    """
    Trace model and compute precision requirements for each op.
    
    Returns:
        dict: {op_name: {'curvature': κ, 'precision_bits': p, 'recommendation': str}}
    """
    # Use torch.fx to get computation graph
    traced = torch.fx.symbolic_trace(model)
    
    precision_map = {}
    for node in traced.graph.nodes:
        if node.op == 'call_function':
            κ = compute_curvature(node.target, node.args)
            p = compute_precision_requirement(κ, target_accuracy=1e-6)
            precision_map[node.name] = {
                'curvature': κ,
                'precision_bits': p,
                'recommendation': 'float16' if p <= 11 else 'float32' if p <= 24 else 'float64'
            }
    
    return precision_map
```

---

## Implementation Plan

### Phase 1: Core Infrastructure (Week 1-2)

**Deliverables:**
- `PrecisionTensor` class with basic operations
- Curvature database for 10 core operations (matmul, exp, log, softmax, relu, sigmoid, tanh, add, mul, div)
- Simple composition tracking

**Validation:**
- Unit tests comparing predicted vs. actual numerical errors
- Test on toy examples: $f(x) = \exp(\exp(x))$ should predict very high precision needs

### Phase 2: PyTorch Integration (Week 3-4)

**Deliverables:**
- FX-based tracing for arbitrary `nn.Module`
- Hook system for dynamic precision tracking during forward pass
- Pretty-printed precision reports

**Validation:**
- Trace ResNet-18 and identify layers with highest precision requirements
- Compare predictions against empirical precision failure experiments

### Phase 3: Lipschitz Estimation (Week 5-6)

**Deliverables:**
- Spectral norm estimation for weight matrices
- Running statistics for activation ranges (domain diameter)
- Integration with batch statistics from BatchNorm/LayerNorm

**Validation:**
- Accuracy of Lipschitz estimates on trained models
- Correlation between predicted curvature and actual gradient explosion events

### Phase 4: Recommendations Engine (Week 7-8)

**Deliverables:**
- Automatic mixed-precision config generation
- Integration with PyTorch AMP
- Warning system for precision-critical operations

**Validation:**
- A/B test: AMP alone vs. AMP + our recommendations
- Measure: stability failures, memory usage, accuracy

---

## Curvature Computations for Key Operations

### Matrix Multiplication

For $f(A, B) = AB$ where $A \in \mathbb{R}^{m \times n}$, $B \in \mathbb{R}^{n \times p}$:

The Hessian in the direction $(dA, dB)$ is:
$$D^2f(A,B)[(dA_1, dB_1), (dA_2, dB_2)] = dA_1 \cdot dB_2 + dA_2 \cdot dB_1$$

The curvature bound becomes:
$$\kappa_{\mathrm{matmul}}^{\mathrm{curv}} \leq \frac{\|A\| \cdot \|B\|}{\sigma_{\min}(AB)^2}$$

For well-conditioned matrices, this is $O(\kappa(A) \cdot \kappa(B))$.

### Softmax

For $f(x)_i = e^{x_i} / \sum_j e^{x_j}$:

The Hessian has the structure:
$$D^2 f(x) = \text{diag}(f) - f f^T - (\text{diag}(f) - f f^T) \otimes \mathbf{1}$$

The curvature bound:
$$\kappa_{\mathrm{softmax}}^{\mathrm{curv}} \leq O(n \cdot e^{2(x_{\max} - x_{\min})})$$

This explains why softmax with large logits causes precision issues.

### Attention

For attention $f(Q, K, V) = \mathrm{softmax}(QK^T / \sqrt{d}) V$:

The composition of matmul → scale → softmax → matmul gives:
$$\kappa_{\mathrm{attn}}^{\mathrm{curv}} \leq O\left(\frac{\|Q\| \|K\| \|V\|}{d} \cdot e^{2\|QK^T\|_\infty / \sqrt{d}}\right)$$

This predicts that:
- Large attention logits cause precision issues (known)
- Longer sequences (larger $\|QK^T\|$) need more precision (known)
- Scaling by $\sqrt{d}$ helps (known)

---

## Validation Strategy

### Experiment 1: Precision Failure Prediction

**Setup:**
1. Take ResNet-18, BERT-base, GPT-2 small
2. For each layer, compute predicted precision requirement
3. Run model in float16, record which layers produce NaN/Inf
4. Compare predictions to failures

**Success Metric:** >0.8 correlation between predicted requirements and failure locations

### Experiment 2: Minimum Viable Precision

**Setup:**
1. Compute per-layer precision requirements for target accuracy $\varepsilon = 10^{-4}$
2. Deploy model with per-layer precision as predicted
3. Measure actual output accuracy

**Success Metric:** Accuracy matches prediction within 1 order of magnitude

### Experiment 3: Training Dynamics

**Setup:**
1. Track curvature during training of Transformer
2. Log curvature spikes and loss spikes
3. Measure temporal correlation

**Success Metric:** Curvature spikes precede loss spikes by 10-100 steps with >80% precision

---

## API Design

### High-Level API

```python
from hnf import PrecisionAnalyzer

# Analyze a model
analyzer = PrecisionAnalyzer(target_accuracy=1e-6)
report = analyzer.analyze(model, sample_input)

# Print recommendations
print(report.summary())
# Output:
# Layer              Curvature    Bits Needed   Recommendation
# attention.softmax  1.2e+08      28            float32 (critical)
# ffn.relu           0.0          0             float16 (safe)
# layer_norm         4.5e+02      12            float16 (marginal)

# Generate mixed-precision config
config = report.to_amp_config()
# {'attention.softmax': torch.float32, 'ffn': torch.float16, ...}

# Check specific operation
curvature = analyzer.curvature_of(torch.softmax, sample_logits)
```

### Integration with Training

```python
from hnf import PrecisionMonitor

monitor = PrecisionMonitor(model, warn_threshold=1e6)

for batch in dataloader:
    loss = model(batch)
    
    # Check for precision issues before they cause NaNs
    if monitor.check():
        warnings = monitor.get_warnings()
        # ['attention_layer.softmax curvature exceeded threshold: 2.3e7']
    
    loss.backward()
    optimizer.step()
```

---

## Compute Requirements

| Task | Time | Hardware |
|------|------|----------|
| Curvature DB construction | 1 day | Laptop |
| ResNet-18 analysis | <1 min | Laptop |
| BERT-base analysis | <5 min | Laptop |
| GPT-2 analysis | <10 min | Laptop |
| Training monitoring | 2x overhead | GPU (optional) |

All development and validation feasible on Mac laptop. Larger models can be analyzed layer-by-layer.

---

## Expected Impact

### Immediate Benefits

1. **Debugging tool**: When training fails with NaNs, immediately identify the culprit layer
2. **Architecture design**: Know before training whether a design will have precision issues
3. **Hardware selection**: Principled choice between float16/bfloat16/float32

### Longer-Term Impact

1. **Foundation for other projects**: Projects 2, 5, 6, 9 all build on this infrastructure
2. **Validation of HNF theory**: Empirical confirmation that curvature bounds are practically useful
3. **New research directions**: Curvature-aware architecture design, precision-aware optimization

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Curvature bounds too loose | Medium | Use tighter local bounds, empirical calibration |
| Lipschitz estimation inaccurate | Medium | Use multiple estimation methods, cross-validate |
| Overhead too high for training | Low | Batch analysis, sampling, caching |
| PyTorch internals change | Low | Abstract over tracing backend |

---

## Success Criteria

1. **Correlation test**: Predicted precision requirements correlate >0.8 with empirical failures
2. **Utility test**: Finds at least 3 real precision bugs in popular open-source models
3. **Adoption test**: At least 100 GitHub stars within 6 months of release

---

## Next Steps

1. Implement `PrecisionTensor` class with matmul, exp, softmax
2. Build curvature database for 10 core operations
3. Create minimal FX tracer for simple models
4. Validate on toy examples before scaling up
