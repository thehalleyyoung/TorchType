# Project 9: Curvature-Guided Transformer Quantization

## Transformer Application: Optimal Per-Layer Quantization for Efficient Inference

**Use case:** Instead of uniform INT8 quantization that loses quality on sensitive layers, use curvature analysis to allocate bits per layer: attention projections get 8 bits, LayerNorm stays FP16, FFN can use 4 bits. Result: same quality as uniform INT8 with 30% fewer bits, or better quality at the same bit budget.

### The Problem with Uniform Quantization

Standard transformer quantization (GPTQ, AWQ, etc.) uses the same bit width everywhere:
- **INT8 everywhere:** Simple but wastes bits on easy layers, loses quality on hard ones
- **INT4 everywhere:** Aggressive compression, significant quality loss
- **Manual tuning:** Experts try different configs, expensive and model-specific

**Hidden assumption:** All layers need the same precision. This is false.

### The Reality: Different Layers Have Different Precision Needs

| Layer Type | Typical Curvature | Precision Sensitivity |
|------------|-------------------|----------------------|
| Embedding lookup | Low | Can use INT4 |
| Q/K projections | Medium-High | Needs INT8 |
| V/O projections | Medium | Can use INT6 |
| Attention softmax | Very High | Keep FP16 |
| LayerNorm | High | Keep FP16 or BF16 |
| FFN (up projection) | Low | Can use INT4 |
| FFN (down projection) | Low | Can use INT4 |

### This Tool Automatically Finds Optimal Bit Allocation

```python
from curvature_quantization import TransformerQuantizer

model = load_llama_7b()
quantizer = TransformerQuantizer(model)

# Analyze curvature to determine per-layer precision
config = quantizer.analyze(
    calibration_data=calibration_dataset,
    bit_budget=4.5,  # Average 4.5 bits per weight (vs 8 for INT8)
)

# Output:
# ╔════════════════════════════════════════════════════════════════╗
# ║ CURVATURE-OPTIMAL QUANTIZATION CONFIG                          ║
# ╠════════════════════════════════════════════════════════════════╣
# ║ Layer                    │ Curvature │ Bits │ Savings         ║
# ╠══════════════════════════╪═══════════╪══════╪═════════════════╣
# ║ embed_tokens             │    12.3   │   4  │  50%            ║
# ║ layers.*.self_attn.q_proj│   892.1   │   8  │   0%            ║
# ║ layers.*.self_attn.k_proj│   743.2   │   8  │   0%            ║
# ║ layers.*.self_attn.v_proj│   234.5   │   6  │  25%            ║
# ║ layers.*.self_attn.o_proj│   187.3   │   6  │  25%            ║
# ║ layers.*.mlp.up_proj     │    45.2   │   4  │  50%            ║
# ║ layers.*.mlp.down_proj   │    38.7   │   4  │  50%            ║
# ║ layers.*.mlp.gate_proj   │    52.1   │   4  │  50%            ║
# ║ lm_head                  │   567.8   │   8  │   0%            ║
# ╠════════════════════════════════════════════════════════════════╣
# ║ Total: 4.3 bits/weight (vs 4.5 budget)                         ║
# ║ Expected quality: 99.2% of FP16 (vs 97.1% for uniform INT4)    ║
# ╚════════════════════════════════════════════════════════════════╝

# Apply quantization
quantized_model = quantizer.quantize(model, config)
```

---

## Theoretical Foundation

### The Precision-Curvature Relationship

From Theorem 5.7, for a layer $f$ with curvature $\kappa_f^{\mathrm{curv}}$:

$$p_f \geq \log_2\left(\frac{\kappa_f^{\mathrm{curv}} \cdot D^2}{\varepsilon}\right)$$

where $D$ is the domain diameter and $\varepsilon$ is the target accuracy.

### Why Uniform Quantization is Suboptimal

Standard quantization: all layers get same bit width (e.g., INT8).

Problem: 
- High-curvature layers (attention projections, LayerNorm) lose critical precision
- Low-curvature layers (FFN, embeddings) waste bits

**Solution:** Allocate bits proportionally to $\log \kappa_f^{\mathrm{curv}}$.

### Bit Budget Optimization

Given total bit budget $B$, optimize allocation:
$$\min_{\{b_\ell\}} \sum_\ell \text{QuantError}_\ell(b_\ell) \quad \text{s.t.} \quad \sum_\ell b_\ell \cdot |\theta_\ell| \leq B$$

Using curvature as proxy for quantization error:
$$\text{QuantError}_\ell(b_\ell) \approx \kappa_\ell^{\mathrm{curv}} \cdot 2^{-b_\ell}$$

### Transformer-Specific Curvature Formulas

| Component | Curvature Formula | Typical Value |
|-----------|-------------------|---------------|
| Linear projection | $\kappa(W) = \sigma_{\max}(W) / \sigma_{\min}(W)$ | $10^1$ to $10^3$ |
| LayerNorm | $\kappa \approx d / \sigma^2$ | $10^2$ to $10^4$ |
| Softmax | $\kappa \approx e^{2 \cdot \max(x)}$ | $10^2$ to $10^6$ |
| GELU | $\kappa \approx 2$ (smooth) | $\sim 2$ |

---

## Technical Approach

### 1. Curvature Analysis

```python
class QuantizationAnalyzer:
    """Analyze model for precision-aware quantization."""
    
    def __init__(self, model, calibration_data):
        self.model = model
        self.calibration_data = calibration_data
        self.layer_stats = {}
    
    def analyze(self) -> Dict[str, 'LayerStats']:
        """Compute per-layer statistics for quantization."""
        
        # Register hooks to collect activation statistics
        hooks = []
        for name, module in self.model.named_modules():
            if self._is_quantizable(module):
                hook = module.register_forward_hook(
                    partial(self._stats_hook, name=name)
                )
                hooks.append(hook)
        
        # Run calibration data
        self.model.eval()
        with torch.no_grad():
            for batch in self.calibration_data:
                self.model(batch)
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Compute curvature for each layer
        for name in self.layer_stats:
            self._compute_curvature(name)
        
        return self.layer_stats
    
    def _stats_hook(self, module, input, output, name):
        """Collect activation statistics."""
        if name not in self.layer_stats:
            self.layer_stats[name] = LayerStats(name)
        
        stats = self.layer_stats[name]
        
        # Update running statistics
        with torch.no_grad():
            x = input[0].detach()
            y = output.detach()
            
            stats.update_input_stats(x)
            stats.update_output_stats(y)
    
    def _compute_curvature(self, name: str):
        """Compute curvature estimate for layer."""
        stats = self.layer_stats[name]
        module = dict(self.model.named_modules())[name]
        
        if isinstance(module, nn.Linear):
            # For linear layers, curvature comes from weight matrix condition
            W = module.weight.data
            stats.curvature = torch.linalg.cond(W).item()
        
        elif isinstance(module, nn.Conv2d):
            # For convolutions, estimate curvature from reshaped weight
            W = module.weight.data.view(module.out_channels, -1)
            stats.curvature = torch.linalg.cond(W).item()
        
        elif isinstance(module, nn.LayerNorm):
            # Layer norm has curvature ~ 1/var
            var = stats.input_var.mean().item()
            stats.curvature = 1.0 / (var + 1e-8)
        
        elif isinstance(module, nn.Softmax):
            # Softmax has curvature ~ exp(2 * max_logit)
            max_val = stats.input_max.item()
            stats.curvature = np.exp(2 * max_val)
        
        else:
            stats.curvature = 1.0  # Default

@dataclass
class LayerStats:
    name: str
    curvature: float = 1.0
    input_range: Tuple[float, float] = (0.0, 1.0)
    output_range: Tuple[float, float] = (0.0, 1.0)
    input_var: float = 1.0
    input_max: float = 1.0
    n_params: int = 0
    
    def update_input_stats(self, x):
        # Update running statistics
        self.input_range = (
            min(self.input_range[0], x.min().item()),
            max(self.input_range[1], x.max().item())
        )
        self.input_var = x.var().item()
        self.input_max = max(self.input_max, x.max().item())
```

### 2. Bit Width Optimization

```python
class BitWidthOptimizer:
    """Optimize per-layer bit widths given budget."""
    
    def __init__(self, layer_stats: Dict[str, LayerStats],
                 min_bits: int = 4,
                 max_bits: int = 16):
        self.layer_stats = layer_stats
        self.min_bits = min_bits
        self.max_bits = max_bits
    
    def optimize(self, total_bits: int) -> Dict[str, int]:
        """
        Find optimal bit allocation under budget constraint.
        
        Uses dynamic programming for discrete optimization.
        """
        layers = list(self.layer_stats.keys())
        n_layers = len(layers)
        
        # Get sizes and curvatures
        sizes = [self.layer_stats[l].n_params for l in layers]
        curvatures = [self.layer_stats[l].curvature for l in layers]
        
        # Normalize curvatures
        log_curv = [np.log(max(k, 1.0)) for k in curvatures]
        total_log_curv = sum(log_curv)
        
        # Initial allocation: proportional to log(curvature)
        initial = {}
        for i, layer in enumerate(layers):
            frac = log_curv[i] / total_log_curv
            bits = int(self.min_bits + frac * (self.max_bits - self.min_bits))
            initial[layer] = np.clip(bits, self.min_bits, self.max_bits)
        
        # Refine via gradient descent on relaxed problem
        bits_float = {l: float(initial[l]) for l in layers}
        
        for iteration in range(100):
            # Compute error estimate
            error = self._estimate_error(bits_float)
            
            # Gradient of error w.r.t. bits
            grad = {}
            for layer in layers:
                # Error decreases with more bits
                grad[layer] = -curvatures[layers.index(layer)] * \
                              np.log(2) * (2 ** (-bits_float[layer]))
            
            # Projected gradient step
            step_size = 0.1
            for layer in layers:
                bits_float[layer] -= step_size * grad[layer]
            
            # Project to budget constraint
            bits_float = self._project_to_budget(bits_float, total_bits, sizes)
        
        # Round to integers
        return {l: int(np.clip(round(b), self.min_bits, self.max_bits))
                for l, b in bits_float.items()}
    
    def optimize_for_accuracy(self, target_accuracy: float) -> Dict[str, int]:
        """
        Find minimum bits per layer to achieve target accuracy.
        
        Uses curvature-based precision formula.
        """
        allocations = {}
        
        for name, stats in self.layer_stats.items():
            κ = stats.curvature
            D = stats.input_range[1] - stats.input_range[0]
            
            # From precision theorem: bits needed for accuracy ε
            bits_needed = np.log2(κ * D ** 2 / target_accuracy)
            bits_needed = int(np.ceil(max(bits_needed, self.min_bits)))
            
            allocations[name] = min(bits_needed, self.max_bits)
        
        return allocations
    
    def _estimate_error(self, bits: Dict[str, float]) -> float:
        """Estimate total quantization error."""
        error = 0.0
        for layer, b in bits.items():
            κ = self.layer_stats[layer].curvature
            error += κ * (2 ** (-b))
        return error
    
    def _project_to_budget(self, bits: Dict[str, float], 
                           budget: int, sizes: List[int]) -> Dict[str, float]:
        """Project allocation to satisfy budget constraint."""
        layers = list(bits.keys())
        
        # Current total
        current = sum(bits[l] * sizes[i] for i, l in enumerate(layers))
        
        if current <= budget:
            return bits
        
        # Scale down proportionally
        scale = budget / current
        return {l: max(self.min_bits, b * scale) for l, b in bits.items()}
```

### 3. Quantization Application

```python
class PrecisionAwareQuantizer:
    """Apply quantization with per-layer bit widths."""
    
    def __init__(self, model, bit_allocations: Dict[str, int]):
        self.model = model
        self.bit_allocations = bit_allocations
    
    def quantize(self) -> nn.Module:
        """Return quantized model with per-layer precision."""
        quantized = copy.deepcopy(self.model)
        
        for name, module in quantized.named_modules():
            if name in self.bit_allocations:
                bits = self.bit_allocations[name]
                self._quantize_module(module, bits)
        
        return quantized
    
    def _quantize_module(self, module: nn.Module, bits: int):
        """Quantize a single module to given bit width."""
        if isinstance(module, nn.Linear):
            # Quantize weights
            scale = (2 ** (bits - 1) - 1) / module.weight.abs().max()
            module.weight.data = torch.round(module.weight * scale) / scale
            
            if module.bias is not None:
                scale = (2 ** (bits - 1) - 1) / module.bias.abs().max()
                module.bias.data = torch.round(module.bias * scale) / scale
        
        elif isinstance(module, nn.Conv2d):
            scale = (2 ** (bits - 1) - 1) / module.weight.abs().max()
            module.weight.data = torch.round(module.weight * scale) / scale
    
    def export_onnx(self, sample_input, path: str):
        """Export quantized model to ONNX with precision annotations."""
        quantized = self.quantize()
        
        # Add precision metadata
        metadata = {'bit_allocations': self.bit_allocations}
        
        torch.onnx.export(
            quantized, sample_input, path,
            opset_version=13,
            custom_opsets={'precision': 1},
            export_params=True
        )
        
        # Save metadata separately
        with open(path + '.precision.json', 'w') as f:
            json.dump(metadata, f)
```

---

## Implementation Plan

### Phase 1: Analysis Pipeline (Week 1-2)

**Deliverables:**
- `QuantizationAnalyzer` with hook-based statistics
- Per-layer curvature estimation
- Calibration workflow

**Validation:**
- Compare curvature estimates against manual computation
- Verify statistics collection is accurate

### Phase 2: Optimization (Week 3-4)

**Deliverables:**
- `BitWidthOptimizer` with budget constraint
- Accuracy-based optimization
- Visualization of allocations

**Validation:**
- Verify budget constraints satisfied
- Compare allocation quality vs uniform

### Phase 3: Quantization (Week 5-6)

**Deliverables:**
- `PrecisionAwareQuantizer`
- Support for common layer types
- ONNX export with precision metadata

**Validation:**
- Verify quantized models run correctly
- Measure accuracy degradation

### Phase 4: Benchmarking (Week 7-8)

**Deliverables:**
- Comparison against uniform INT8
- Comparison against other mixed-precision methods
- Performance benchmarks

**Validation:**
- Measure accuracy vs bit budget tradeoffs
- Demonstrate 20-30% bit reduction at same accuracy

---

## Algorithm Details

### Curvature-Based Bit Allocation

For layer $\ell$ with curvature $\kappa_\ell$:

**Step 1:** Compute precision requirement:
$$p_\ell = \log_2\left(\kappa_\ell \cdot D_\ell^2 / \varepsilon\right)$$

**Step 2:** Allocate bits proportionally:
$$b_\ell = \frac{p_\ell}{\sum_k p_k} \cdot B_{\text{total}}$$

**Step 3:** Clip to valid range:
$$b_\ell = \max(\min(b_\ell, 16), 4)$$

**Step 4:** Adjust for budget:
If $\sum_\ell b_\ell \cdot |\theta_\ell| > B$, reduce bits for low-curvature layers first.

### Error Model

Quantization error for $b$-bit representation:
$$\text{Error}(b) = \frac{\text{range}}{2^b - 1}$$

For a layer with curvature $\kappa$, the output error amplification:
$$\text{OutputError} \approx \kappa \cdot \text{InputError}$$

Total error after quantization:
$$\text{TotalError} \approx \sum_\ell \kappa_\ell \cdot 2^{-b_\ell}$$

---

## Example Analysis

### ResNet-18 Layer Curvatures

| Layer | Type | Curvature | Recommended Bits |
|-------|------|-----------|------------------|
| conv1 | Conv2d | 2.3 | 6 |
| bn1 | BatchNorm | 15.7 | 8 |
| layer1.0.conv1 | Conv2d | 3.1 | 6 |
| layer4.1.conv2 | Conv2d | 45.2 | 10 |
| fc | Linear | 128.5 | 12 |

**Observation:** Later layers and fully connected have higher curvature → need more bits.

### BERT Attention Layer

| Component | Curvature | Recommended Bits |
|-----------|-----------|------------------|
| Q projection | 5.2 | 8 |
| K projection | 5.1 | 8 |
| V projection | 4.8 | 8 |
| Attention scores (softmax) | 2.3e5 | 14 |
| Output projection | 6.7 | 8 |

**Observation:** Softmax has extremely high curvature → keep high precision.

---

## Validation Strategy

### Experiment 1: Bit Savings at Same Accuracy

**Setup:**
1. Take ResNet-50 on ImageNet
2. Quantize uniformly to INT8 (8 bits everywhere)
3. Quantize with our method to match accuracy
4. Measure total bits used

**Success Metric:** 20-30% fewer bits at same accuracy

### Experiment 2: Accuracy at Same Bits

**Setup:**
1. Fix bit budget = uniform INT8 budget
2. Compare accuracy: uniform vs curvature-aware

**Success Metric:** 1-2% higher accuracy

### Experiment 3: Comparison with Other Methods

**Setup:**
- Compare against:
  - HAWQ (Hessian-Aware Quantization)
  - Mixed-precision via sensitivity analysis
  - Random allocation (baseline)

**Success Metric:** Match or exceed HAWQ with simpler method

---

## API Design

```python
from hnf.quantize import QuantizationAnalyzer, BitWidthOptimizer, PrecisionAwareQuantizer

# Step 1: Analyze model
analyzer = QuantizationAnalyzer(model, calibration_loader)
layer_stats = analyzer.analyze()

# View analysis
for name, stats in layer_stats.items():
    print(f"{name}: curvature={stats.curvature:.2e}, range={stats.input_range}")

# Step 2: Optimize bit allocation
optimizer = BitWidthOptimizer(layer_stats, min_bits=4, max_bits=16)

# Option A: Given bit budget
bits = optimizer.optimize(total_bits=32_000_000)

# Option B: Given accuracy target
bits = optimizer.optimize_for_accuracy(target_accuracy=1e-4)

# Step 3: Apply quantization
quantizer = PrecisionAwareQuantizer(model, bits)
quantized_model = quantizer.quantize()

# Evaluate
print(f"Original accuracy: {evaluate(model, test_loader):.2%}")
print(f"Quantized accuracy: {evaluate(quantized_model, test_loader):.2%}")
print(f"Total bits: {sum(b * s for b, s in zip(bits.values(), sizes))}")

# Export
quantizer.export_onnx(sample_input, 'model_quantized.onnx')
```

---

## Advanced Features

### 1. Activation Quantization

Extend to quantize activations, not just weights:

```python
def quantize_activations(model, bit_allocations):
    """Insert quantization nodes for activations."""
    class QuantizedLayer(nn.Module):
        def __init__(self, original, bits):
            super().__init__()
            self.original = original
            self.bits = bits
        
        def forward(self, x):
            # Quantize input
            x_q = quantize_tensor(x, self.bits)
            # Apply original layer
            y = self.original(x_q)
            return y
    
    # Wrap each layer
    for name, bits in bit_allocations.items():
        module = get_module(model, name)
        set_module(model, name, QuantizedLayer(module, bits))
```

### 2. Hardware-Aware Optimization

Consider hardware constraints:

```python
def hardware_aware_optimize(layer_stats, hardware):
    """Optimize for specific hardware constraints."""
    # Get supported bit widths
    supported = hardware.supported_bit_widths  # e.g., [4, 8, 16]
    
    optimizer = BitWidthOptimizer(layer_stats)
    
    # Constrain to supported widths
    bits = optimizer.optimize(...)
    bits = {l: min(supported, key=lambda s: abs(s - b))
            for l, b in bits.items()}
    
    return bits
```

### 3. Fine-Tuning After Quantization

```python
def quantization_aware_finetuning(model, bit_allocations, train_loader, epochs=5):
    """Fine-tune with straight-through estimator."""
    quantized = PrecisionAwareQuantizer(model, bit_allocations).quantize()
    
    optimizer = torch.optim.Adam(quantized.parameters(), lr=1e-5)
    
    for epoch in range(epochs):
        for batch in train_loader:
            loss = compute_loss(quantized, batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
    
    return quantized
```

---

## Comparison with Existing Methods

### vs Uniform Quantization

| Aspect | Uniform | Ours |
|--------|---------|------|
| Bit allocation | Same everywhere | Curvature-proportional |
| Accuracy | Baseline | +1-2% at same bits |
| Complexity | Simple | Requires analysis |

### vs HAWQ

| Aspect | HAWQ | Ours |
|--------|------|------|
| Sensitivity metric | Hessian eigenvalues | Curvature bounds |
| Computation | Expensive (full Hessian) | Cheaper (curvature only) |
| Theoretical basis | Empirical | HNF theorem |

### vs Sensitivity Analysis

| Aspect | Sensitivity | Ours |
|--------|-------------|------|
| Method | Perturb and measure | Compute curvature |
| Cost | Many forward passes | One analysis pass |
| Guarantee | None | Precision theorem |

---

## Compute Requirements

| Task | Time | Hardware |
|------|------|----------|
| Calibration (100 batches) | ~1 min | GPU |
| Curvature analysis | ~1 min | GPU |
| Bit optimization | <1 sec | CPU |
| Quantization | <1 sec | CPU |
| Accuracy evaluation | Model-dependent | GPU |

Total pipeline: 5-10 minutes for typical model.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Curvature doesn't predict quantization error | Low | Validated by precision theorem |
| Calibration data unrepresentative | Medium | Use diverse calibration set |
| Per-layer quantization not supported by hardware | Medium | Constrain to supported widths |
| Overhead of mixed precision | Low | Most hardware supports this |

---

## Expected Impact

### For Deployment

- Smaller models with same accuracy
- Better accuracy at same size
- Principled quantization decisions

### For Edge Devices

- Better utilization of limited precision
- Fewer bits = less memory = faster inference
- Formal guarantees on accuracy

### For Research

- Validates curvature-precision relationship
- New quantization methodology
- Connection between HNF and compression

---

## Next Steps

1. Implement `QuantizationAnalyzer` with curvature estimation
2. Build `BitWidthOptimizer` with budget constraint
3. Create `PrecisionAwareQuantizer` for common layers
4. Validate on ResNet, BERT
5. Compare against uniform INT8 and HAWQ
