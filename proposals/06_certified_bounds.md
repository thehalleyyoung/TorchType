# Project 6: Certified Precision Bounds for Transformer Inference

## Transformer Application: Know Exactly What Precision You Need for Deployment

**Use case:** Before deploying a transformer to edge devices, mobile, or specialized hardware, get guaranteed answers: "This model requires at least FP16 for 99.9% accuracy" or "INT8 is safe for this use case." No more trial-and-error deployment.

### The Problem with Transformer Deployment

Deploying transformers involves precision trade-offs:
- **FP32:** Safe but slow, uses 4x memory
- **FP16/BF16:** Fast but attention can overflow on long sequences
- **INT8:** Great for efficiency but may degrade quality unpredictably
- **FP8 (new):** Even faster but almost no guidance on when it's safe

Current practice: Deploy in lower precision, observe errors, increase precision if broken. Expensive, slow, unreliable.

### This Tool Gives Formal Guarantees Before Deployment

```python
from certified_bounds import TransformerCertifier

certifier = TransformerCertifier(model)

# Specify your deployment scenario
cert = certifier.certify(
    input_spec=InputSpec(
        max_sequence_length=2048,
        vocabulary_size=50000,
        embedding_range=(-3, 3)  # After normalization
    ),
    accuracy_target=1e-4,  # Max acceptable output difference
)

# Output:
# ╔══════════════════════════════════════════════════════════════╗
# ║ PRECISION CERTIFICATE                                         ║
# ╠══════════════════════════════════════════════════════════════╣
# ║ Minimum Required Precision:  17 bits (FP16 insufficient!)     ║
# ║ Recommendation:              BF16 or FP32                     ║
# ║                                                                ║
# ║ Bottleneck Layers:                                            ║
# ║   - Attention Layer 11: κ = 2.3e6 (softmax overflow risk)     ║
# ║   - FFN Layer 8: κ = 1.1e4 (GELU boundary)                    ║
# ║                                                                ║
# ║ If FP16 required:                                             ║
# ║   - Reduce max_sequence_length to 512                         ║
# ║   - Or apply attention scaling fix (see suggestions)          ║
# ╚══════════════════════════════════════════════════════════════╝
```

---

## Theoretical Foundation

### The Precision Bound Theorem

From Theorem 5.7, for any computation $f$ with curvature $\kappa_f^{\mathrm{curv}}$ on domain $D$:

$$p_{\min} \geq \log_2\left(\frac{c \cdot \kappa_f^{\mathrm{curv}} \cdot \mathrm{diam}(D)^2}{\varepsilon}\right)$$

where:
- $p_{\min}$ is the minimum precision (bits of mantissa)
- $c$ is an explicit constant depending on the computation
- $\mathrm{diam}(D)$ is the diameter of the input domain
- $\varepsilon$ is the target accuracy

### Transformer-Specific Precision Requirements

| Component | Curvature Formula | Precision Driver |
|-----------|-------------------|------------------|
| Softmax attention | $\kappa \propto e^{2 \cdot \text{seqlen} \cdot \|QK\|}$ | Sequence length |
| LayerNorm | $\kappa \propto 1/\sigma^2$ | Token variance |
| Embedding | $\kappa \propto \text{vocab\_size}$ | Vocabulary size |
| Cross-entropy | $\kappa \propto 1/p_{\min}$ | Rare token handling |

### Why Certification Matters for Transformers

| Without Certification | With Certification |
|----------------------|-------------------|
| Deploy FP16, find 5% accuracy drop | Know FP16 loses 5% before deployment |
| Rollback to FP32, lose 2x speed | Know BF16 is safe, keep speed |
| Edge deployment fails mysteriously | Prove INT8 works for your input range |
| No confidence in production quality | Formal guarantee with proof |

---

## Technical Approach

### 1. Input Domain Specification

```python
@dataclass
class InputDomain:
    """Specification of valid inputs."""
    
    # Bounding box
    lower_bounds: torch.Tensor  # Shape: (input_dim,)
    upper_bounds: torch.Tensor
    
    # Optional: distribution assumptions
    distribution: Optional[str] = None  # 'uniform', 'gaussian', etc.
    mean: Optional[torch.Tensor] = None
    std: Optional[torch.Tensor] = None
    
    def diameter(self) -> float:
        """Compute diameter of domain."""
        return torch.norm(self.upper_bounds - self.lower_bounds).item()
    
    def sample(self, n: int) -> torch.Tensor:
        """Sample n points from domain."""
        if self.distribution == 'uniform':
            return torch.rand(n, len(self.lower_bounds)) * \
                   (self.upper_bounds - self.lower_bounds) + self.lower_bounds
        elif self.distribution == 'gaussian':
            return torch.randn(n, len(self.mean)) * self.std + self.mean
        else:
            # Default: uniform in bounding box
            return torch.rand(n, len(self.lower_bounds)) * \
                   (self.upper_bounds - self.lower_bounds) + self.lower_bounds

    @classmethod
    def from_dataset(cls, dataset, percentile=99.9):
        """Construct domain from dataset statistics."""
        all_data = torch.stack([x for x, _ in dataset])
        lower = torch.quantile(all_data, (100 - percentile) / 200, dim=0)
        upper = torch.quantile(all_data, (100 + percentile) / 200, dim=0)
        return cls(lower_bounds=lower, upper_bounds=upper)
```

### 2. Interval Arithmetic

Use interval arithmetic for rigorous curvature bounds:

```python
class Interval:
    """Interval arithmetic for rigorous bounds."""
    
    def __init__(self, lower: torch.Tensor, upper: torch.Tensor):
        self.lower = lower
        self.upper = upper
    
    def __add__(self, other):
        return Interval(
            self.lower + other.lower,
            self.upper + other.upper
        )
    
    def __mul__(self, other):
        products = torch.stack([
            self.lower * other.lower,
            self.lower * other.upper,
            self.upper * other.lower,
            self.upper * other.upper
        ])
        return Interval(products.min(dim=0)[0], products.max(dim=0)[0])
    
    def exp(self):
        return Interval(torch.exp(self.lower), torch.exp(self.upper))
    
    def log(self):
        # Requires lower > 0
        return Interval(torch.log(self.lower), torch.log(self.upper))
    
    def norm_bound(self):
        """Upper bound on ||x|| for x in interval."""
        return torch.max(torch.abs(self.lower), torch.abs(self.upper)).norm()
```

### 3. Layer-wise Curvature Bounds

```python
def bound_layer_curvature(layer, input_interval: Interval) -> float:
    """
    Compute rigorous upper bound on layer curvature.
    
    Returns:
        Upper bound on κ^curv for this layer on given input domain.
    """
    if isinstance(layer, nn.Linear):
        # Linear: ||D²f|| = 0, but need to track Jacobian for composition
        W = layer.weight
        # Curvature is 0, but we track Lipschitz constant
        return 0.0, W.norm(2).item()
    
    elif isinstance(layer, nn.ReLU):
        # ReLU: piecewise linear, κ = 0
        return 0.0, 1.0
    
    elif isinstance(layer, nn.Softmax):
        # Softmax: κ ≈ exp(2 * max_logit)
        max_val = input_interval.upper.max().item()
        return np.exp(2 * max_val), 1.0
    
    elif isinstance(layer, nn.LayerNorm):
        # Layer norm: κ ≈ 1 / var_min²
        # Need to bound variance from below
        var_lower = estimate_var_lower_bound(input_interval)
        return 1.0 / var_lower ** 2, 1.0
    
    else:
        raise NotImplementedError(f"Curvature bound not implemented for {type(layer)}")

def propagate_intervals(model, input_domain: InputDomain) -> List[Interval]:
    """
    Propagate intervals through model to get activation bounds.
    """
    x = Interval(input_domain.lower_bounds, input_domain.upper_bounds)
    intervals = [x]
    
    for layer in model.children():
        x = apply_layer_interval(layer, x)
        intervals.append(x)
    
    return intervals

def total_curvature_bound(model, input_domain: InputDomain) -> float:
    """
    Compute rigorous upper bound on total model curvature.
    """
    intervals = propagate_intervals(model, input_domain)
    
    total_curv = 0.0
    total_lip = 1.0
    
    for i, layer in enumerate(model.children()):
        layer_curv, layer_lip = bound_layer_curvature(layer, intervals[i])
        
        # Composition rule: κ_{g∘f} ≤ κ_g * L_f² + κ_f * ||Dg||
        total_curv = total_curv * layer_lip ** 2 + layer_curv * total_lip
        total_lip *= layer_lip
    
    return total_curv
```

### 4. Certificate Generation

```python
@dataclass
class PrecisionCertificate:
    """Certificate guaranteeing precision requirements."""
    
    model_hash: str  # Hash of model weights
    input_domain: InputDomain
    target_accuracy: float
    curvature_bound: float
    precision_requirement: int  # Bits of mantissa
    
    # Audit trail
    timestamp: str
    computation_details: dict
    
    def verify(self, model) -> bool:
        """Verify certificate is valid for given model."""
        # Check model hash matches
        if hash_model(model) != self.model_hash:
            return False
        
        # Recompute curvature bound and check
        κ = total_curvature_bound(model, self.input_domain)
        if κ > self.curvature_bound:
            return False
        
        # Check precision formula
        D = self.input_domain.diameter()
        p_required = np.log2(
            self.curvature_bound * D ** 2 / self.target_accuracy
        )
        return p_required <= self.precision_requirement
    
    def to_json(self) -> str:
        """Serialize certificate for distribution."""
        return json.dumps({
            'model_hash': self.model_hash,
            'input_domain': self.input_domain.to_dict(),
            'target_accuracy': self.target_accuracy,
            'curvature_bound': self.curvature_bound,
            'precision_requirement': self.precision_requirement,
            'timestamp': self.timestamp,
            'computation_details': self.computation_details
        })

def certify_model(model, 
                  input_domain: InputDomain,
                  target_accuracy: float) -> PrecisionCertificate:
    """
    Generate precision certificate for model.
    
    Args:
        model: PyTorch model to certify
        input_domain: Specification of valid inputs
        target_accuracy: Maximum allowed output error
    
    Returns:
        PrecisionCertificate with guaranteed precision requirements
    """
    # Compute curvature bound
    κ = total_curvature_bound(model, input_domain)
    
    # Compute domain diameter
    D = input_domain.diameter()
    
    # Compute precision requirement
    p_min = int(np.ceil(np.log2(κ * D ** 2 / target_accuracy)))
    
    # Add safety margin
    p_min += 2  # Extra 2 bits for rounding
    
    return PrecisionCertificate(
        model_hash=hash_model(model),
        input_domain=input_domain,
        target_accuracy=target_accuracy,
        curvature_bound=κ,
        precision_requirement=p_min,
        timestamp=datetime.now().isoformat(),
        computation_details={
            'diameter': D,
            'curvature_method': 'interval_arithmetic',
            'safety_margin_bits': 2
        }
    )
```

---

## Implementation Plan

### Phase 1: Interval Arithmetic (Week 1-2)

**Deliverables:**
- `Interval` class with all standard operations
- Interval propagation for linear layers, ReLU, softmax
- Basic tests for correctness

**Validation:**
- Compare interval bounds against sampling-based bounds
- Verify intervals contain all sampled outputs

### Phase 2: Curvature Bounds (Week 3-4)

**Deliverables:**
- Layer-wise curvature bound functions
- Composition rules for total curvature
- Integration with PyTorch models

**Validation:**
- Compare bounds against empirical curvature estimates
- Verify bounds are not too loose (within 10x of empirical)

### Phase 3: Certificate Generation (Week 5-6)

**Deliverables:**
- `PrecisionCertificate` dataclass
- `certify_model` function
- Verification and serialization

**Validation:**
- Generate certificates for small models
- Verify certificates by testing at predicted precision

### Phase 4: Deployment Integration (Week 7-8)

**Deliverables:**
- Export to ONNX/TFLite with precision annotations
- Integration with quantization workflows
- Documentation and examples

**Validation:**
- Deploy models at certified precision
- Verify accuracy matches certificate

---

## Example Certificate

```json
{
    "model_hash": "a1b2c3d4e5f6...",
    "input_domain": {
        "type": "bounding_box",
        "lower_bounds": [0.0, 0.0, 0.0, ...],
        "upper_bounds": [1.0, 1.0, 1.0, ...],
        "dimension": 784
    },
    "target_accuracy": 1e-6,
    "curvature_bound": 1.234e8,
    "precision_requirement": 28,
    "timestamp": "2024-01-15T10:30:00Z",
    "computation_details": {
        "diameter": 28.0,
        "curvature_method": "interval_arithmetic",
        "safety_margin_bits": 2,
        "per_layer_curvature": {
            "conv1": 0.0,
            "relu1": 0.0,
            "conv2": 0.0,
            "relu2": 0.0,
            "fc1": 0.0,
            "softmax": 1.234e8
        }
    },
    "conclusion": "This model requires at least 28 bits of mantissa (float32) for inputs in the specified domain to achieve 1e-6 accuracy. Float16 (11 bits) is NOT sufficient. Float64 (52 bits) provides ample margin."
}
```

---

## Precision Recommendations

Based on precision requirement $p$, recommend:

| $p$ (bits) | Recommendation | Data Type |
|------------|----------------|-----------|
| ≤ 8 | int8 sufficient | int8 |
| ≤ 11 | float16 sufficient | float16/bfloat16 |
| ≤ 24 | float32 required | float32 |
| ≤ 52 | float64 required | float64 |
| > 52 | Extended precision or reformulation | — |

---

## Validation Strategy

### Experiment 1: Certificate Accuracy

**Setup:**
1. Generate certificates for 10 models on 10 input domains
2. Deploy at certified precision
3. Measure actual accuracy

**Success Metric:** Actual accuracy ≤ certified accuracy in 100% of cases

### Experiment 2: Tightness

**Setup:**
1. Compare certified precision against empirical minimum
2. Empirical minimum = lowest precision that achieves accuracy

**Success Metric:** Certified precision within 4 bits of empirical minimum

### Experiment 3: Real-World Models

**Setup:**
1. Certify popular models: ResNet, BERT, ViT
2. Compare against known precision requirements
3. Identify where our bounds are tight vs loose

---

## API Design

```python
from hnf.certify import certify_model, InputDomain

# Define input domain
domain = InputDomain.from_dataset(test_dataset)
# Or manually:
domain = InputDomain(
    lower_bounds=torch.zeros(784),
    upper_bounds=torch.ones(784)
)

# Generate certificate
cert = certify_model(
    model,
    input_domain=domain,
    target_accuracy=1e-4
)

print(f"Precision requirement: {cert.precision_requirement} bits")
print(f"Recommendation: {'float16' if cert.precision_requirement <= 11 else 'float32'}")

# Verify certificate
assert cert.verify(model)

# Save certificate
with open('model_cert.json', 'w') as f:
    f.write(cert.to_json())

# Deploy with confidence
if cert.precision_requirement <= 11:
    model = model.half()  # Safe to use float16
```

---

## Advanced Features

### 1. Per-Layer Certificates

Generate certificates for each layer to enable mixed precision:

```python
def certify_per_layer(model, input_domain, target_accuracy):
    """Generate per-layer precision certificates."""
    certs = {}
    
    intervals = propagate_intervals(model, input_domain)
    
    for i, (name, layer) in enumerate(model.named_children()):
        κ, L = bound_layer_curvature(layer, intervals[i])
        D_local = intervals[i].diameter()
        
        p = int(np.ceil(np.log2(κ * D_local ** 2 / target_accuracy)))
        certs[name] = {
            'curvature': κ,
            'precision_bits': p,
            'recommendation': bits_to_dtype(p)
        }
    
    return certs
```

### 2. Probabilistic Certificates

For tighter bounds, allow probabilistic guarantees:

```python
def certify_probabilistic(model, input_domain, target_accuracy, confidence=0.99):
    """
    Generate certificate with probabilistic guarantee.
    
    Guarantee: With probability ≥ confidence, accuracy ≤ target
    """
    # Sample inputs and estimate curvature distribution
    samples = input_domain.sample(10000)
    curvatures = [estimate_local_curvature(model, x) for x in samples]
    
    # Use percentile instead of max
    κ = np.percentile(curvatures, confidence * 100)
    
    # Rest of certification
    ...
```

### 3. Input-Dependent Certification

Some inputs need more precision than others:

```python
def certify_input(model, x, target_accuracy):
    """
    Certify precision requirement for specific input.
    
    Returns tighter bound than domain-wide certificate.
    """
    # Local curvature at x
    κ = local_curvature(model, x)
    
    # Local precision requirement
    p = int(np.ceil(np.log2(κ / target_accuracy)))
    
    return p
```

---

## Comparison with Existing Approaches

### vs. Empirical Testing

| Aspect | Empirical | Our Certificates |
|--------|-----------|------------------|
| Guarantee | None | Formal |
| Coverage | Tested inputs only | All inputs in domain |
| Cost | Many inference runs | One-time analysis |
| Confidence | Statistical | Mathematical |

### vs. Sensitivity Analysis

| Aspect | Sensitivity | Our Certificates |
|--------|-------------|------------------|
| Order | First-order (Jacobian) | Second-order (curvature) |
| Bound type | Local | Global on domain |
| Precision prediction | Indirect | Direct |

---

## Compute Requirements

| Task | Time | Hardware |
|------|------|----------|
| Interval propagation | O(forward pass) | Laptop |
| Curvature bound | O(n_layers × forward) | Laptop |
| Full certification | <1 min for typical model | Laptop |

Memory: 2-3x model size (for interval bounds)

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Bounds too loose | Medium | Use tighter interval methods (affine, zonotopes) |
| Some layers not supported | Medium | Add bounds incrementally, fall back to sampling |
| Certification too slow | Low | Cache intermediate results, parallelize |
| Users don't trust certificates | Medium | Provide verification, detailed audit trail |

---

## Expected Impact

### For Deployment

- Know precision requirements before deployment
- Principled hardware selection
- Avoid precision-related production failures

### For Quantization

- Guide quantization decisions
- Identify layers that can't be quantized
- Formal guarantees on quantized models

### For Safety-Critical Applications

- Certifiable numerical behavior
- Audit trail for precision decisions
- Compliance with numerical requirements

---

## Next Steps

1. Implement `Interval` class with core operations
2. Build interval propagation for MLP, CNN
3. Implement layer-wise curvature bounds
4. Create `certify_model` function
5. Validate on small models with known precision requirements
