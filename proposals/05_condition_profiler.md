# Project 5: Condition Number Profiler for Transformer Training

## Transformer Application: Predict and Prevent Training Instability

**Use case:** Monitor numerical condition throughout transformer training to predict loss spikes, gradient explosions, and NaN values before they happen. Essential for training large models in mixed precision where instability is common.

### The Problem with Transformer Training

Large transformer training frequently encounters:
- **Loss spikes:** Sudden jumps in loss, often requiring checkpoint rollback
- **Gradient explosion:** Especially in attention layers with long sequences
- **NaN cascades:** One bad batch corrupts the entire training run
- **Silent precision loss:** Model trains but converges to suboptimal solution

These problems cost compute time and make reproducibility difficult.

### This Tool Predicts Instabilities Before They Happen

```python
profiler = TransformerConditionProfiler(model)

for step, batch in enumerate(dataloader):
    loss = model(batch)
    
    # Get per-layer condition analysis
    conditions = profiler.analyze(loss)
    
    # Warnings appear BEFORE loss spikes
    # [WARNING] Step 4532: Attention layer 12 condition number spiking (κ=1e8)
    # [WARNING] Step 4532: Predicted loss spike in 10-50 steps
    # [ACTION] Recommended: reduce LR by 2x for next 100 steps
    
    if conditions.risk_level > 0.8:
        # Automatic intervention
        optimizer.param_groups[0]['lr'] *= 0.5
        
    loss.backward()
    optimizer.step()
```

---

## Theoretical Foundation

### Training as a Path Through Curvature Landscape

Training a neural network traces a path in parameter space:
$$\theta(t): [0, T] \to \Theta$$

At each point, the network $f(\cdot; \theta(t))$ has a numerical curvature profile:
$$\kappa^{\mathrm{curv}}(t) = \{\kappa_\ell^{\mathrm{curv}}(\theta(t))\}_{\ell=1}^L$$

**Key insight:** Sharp increases in curvature *precede* training failures. The loss landscape becomes numerically treacherous before the loss actually spikes.

### Transformer-Specific Curvature Sources

| Component | Curvature Source | Typical Range |
|-----------|------------------|---------------|
| Softmax attention | $\kappa \approx e^{2 \cdot \max(QK^T)}$ | $10^2$ to $10^{10}$ |
| LayerNorm | $\kappa \approx 1/\sigma^2$ | $10^1$ to $10^4$ |
| Embedding lookup | Sparse → condition issues | $10^1$ to $10^3$ |
| FFN (GELU) | Smooth but scales with width | $10^0$ to $10^2$ |
| Cross-entropy | $\kappa \approx 1/p_{\min}$ | $10^2$ to $10^6$ |

### Correlation with Training Pathologies

| Pathology | Curvature Signature |
|-----------|---------------------|
| Loss spike | Sharp curvature increase 10-100 steps before |
| Gradient explosion | $\kappa$ grows exponentially across attention layers |
| Attention collapse | Single head curvature → $\infty$, others → 0 |
| NaN/Inf | $\kappa$ exceeds float16 precision limit ($\kappa > 10^4$) |

---

## Technical Approach

### 1. Efficient Curvature Estimation

We can't compute full Hessians for large networks. Instead, use efficient approximations:

#### Pearlmutter's Trick (Hessian-Vector Products)

Compute $\nabla^2 f \cdot v$ without forming $\nabla^2 f$:

```python
def hessian_vector_product(loss_fn, params, v):
    """
    Compute Hessian-vector product using double backward.
    
    Cost: 2 backward passes (same as computing gradient)
    """
    # First: compute gradient
    grad = torch.autograd.grad(loss_fn(params), params, create_graph=True)
    
    # Second: differentiate <grad, v> w.r.t. params
    grad_v = sum((g * vi).sum() for g, vi in zip(grad, v))
    hvp = torch.autograd.grad(grad_v, params)
    
    return hvp
```

#### Power Iteration for Spectral Norm

Estimate $\|D^2 f\|$ by finding the largest eigenvalue:

```python
def estimate_hessian_spectral_norm(loss_fn, params, n_iterations=10):
    """
    Estimate ||∇²f|| via power iteration.
    
    Returns largest eigenvalue of Hessian.
    """
    # Random initial vector
    v = [torch.randn_like(p) for p in params]
    v = normalize(v)
    
    for _ in range(n_iterations):
        hvp = hessian_vector_product(loss_fn, params, v)
        eigenvalue = dot(hvp, v)
        v = normalize(hvp)
    
    return eigenvalue
```

#### Jacobian Singular Values

For layer-wise analysis, estimate condition number of Jacobian:

```python
def layer_condition_number(layer, input_batch, n_samples=10):
    """
    Estimate condition number of layer Jacobian.
    
    Uses randomized SVD approximation.
    """
    # Compute Jacobian-vector products for random vectors
    outputs = []
    for _ in range(n_samples):
        v = torch.randn(input_batch.shape[0], layer.out_features)
        jvp = jacobian_vector_product(layer, input_batch, v)
        outputs.append(jvp)
    
    # Stack and compute approximate singular values
    J_approx = torch.stack(outputs, dim=-1)
    _, S, _ = torch.linalg.svd(J_approx, full_matrices=False)
    
    return S[0] / S[-1]  # Condition number = σ_max / σ_min
```

### 2. Hook System

```python
class CurvatureProfiler:
    """Profile per-layer curvature during training."""
    
    def __init__(self, model, sample_input):
        self.model = model
        self.sample_input = sample_input
        self.hooks = []
        self.curvature_history = defaultdict(list)
        self.gradient_history = defaultdict(list)
        
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward and backward hooks on all layers."""
        for name, module in self.model.named_modules():
            if self._is_tracked_layer(module):
                hook = module.register_forward_hook(
                    partial(self._forward_hook, name=name)
                )
                self.hooks.append(hook)
    
    def _forward_hook(self, module, input, output, name):
        """Record activations for curvature computation."""
        self.activations[name] = input[0].detach()
    
    def compute_curvature(self, loss):
        """Compute per-layer curvature after forward pass."""
        curvatures = {}
        
        for name, module in self.model.named_modules():
            if name not in self.activations:
                continue
            
            # Compute layer-specific curvature
            input_act = self.activations[name]
            κ = self._layer_curvature(module, input_act, loss)
            curvatures[name] = κ
            self.curvature_history[name].append(κ)
        
        return curvatures
    
    def _layer_curvature(self, module, input_act, loss):
        """Estimate curvature for a single layer."""
        # Estimate ||D²f||
        hess_norm = estimate_hessian_spectral_norm(
            lambda: module(input_act).sum(), 
            module.parameters()
        )
        
        # Estimate ||Df⁻¹||²
        cond = layer_condition_number(module, input_act)
        
        return hess_norm * cond ** 2
```

### 3. Real-Time Monitoring

```python
class TrainingMonitor:
    """Monitor training for numerical issues."""
    
    def __init__(self, model, warn_threshold=1e6, danger_threshold=1e9):
        self.profiler = CurvatureProfiler(model)
        self.warn_threshold = warn_threshold
        self.danger_threshold = danger_threshold
        self.step = 0
    
    def on_step(self, loss):
        """Call after each training step."""
        self.step += 1
        
        curvatures = self.profiler.compute_curvature(loss)
        
        # Check for warnings
        warnings = []
        for name, κ in curvatures.items():
            if κ > self.danger_threshold:
                warnings.append(f"DANGER: Layer {name} curvature = {κ:.2e}")
            elif κ > self.warn_threshold:
                warnings.append(f"WARNING: Layer {name} curvature = {κ:.2e}")
        
        # Check for rapid increases
        for name, history in self.profiler.curvature_history.items():
            if len(history) >= 10:
                recent = history[-10:]
                if recent[-1] / recent[0] > 10:
                    warnings.append(
                        f"RAPID INCREASE: Layer {name} curvature grew 10x in 10 steps"
                    )
        
        return warnings
    
    def predict_failure(self, horizon=100):
        """Predict if training will fail within horizon steps."""
        for name, history in self.profiler.curvature_history.items():
            if len(history) < 20:
                continue
            
            # Fit exponential to recent curvature
            recent = np.array(history[-20:])
            log_recent = np.log(recent + 1e-10)
            slope = np.polyfit(range(len(log_recent)), log_recent, 1)[0]
            
            # Extrapolate
            projected = recent[-1] * np.exp(slope * horizon)
            
            if projected > 1e15:  # Would overflow float32
                return True, name, projected
        
        return False, None, None
```

---

## Implementation Plan

### Phase 1: Curvature Estimation (Week 1-2)

**Deliverables:**
- Hessian-vector product implementation
- Power iteration for spectral norm
- Layer condition number estimation

**Validation:**
- Compare against full Hessian computation (small networks)
- Benchmark estimation accuracy and speed

### Phase 2: Hook System (Week 3-4)

**Deliverables:**
- `CurvatureProfiler` class
- Forward/backward hooks for activation capture
- Per-layer curvature computation

**Validation:**
- Profile small models (MLP, small CNN)
- Verify curvature changes during training

### Phase 3: Monitoring (Week 5-6)

**Deliverables:**
- `TrainingMonitor` class
- Warning system for high curvature
- Failure prediction

**Validation:**
- Induce training failures and check prediction
- Measure false positive/negative rates

### Phase 4: Visualization (Week 7-8)

**Deliverables:**
- Dashboard for curvature visualization
- Layer-wise heatmaps over time
- Correlation plots with loss

**Validation:**
- User study: does visualization help debug training?
- Integration with W&B/TensorBoard

---

## Visualization Design

### Curvature Heatmap

```
Layer          |Step 0    100    200    300    400    500
---------------|------------------------------------------
embed          |  ■       ■      ■      ■      ■      ■
attn.q_proj    |  ■       ■      ■      ■      ■      ▓
attn.k_proj    |  ■       ■      ■      ■      ▓      █
attn.softmax   |  ▓       ▓      █      █      █      !!!
attn.v_proj    |  ■       ■      ■      ■      ■      ▓
ffn.up_proj    |  ■       ■      ■      ▓      ▓      ▓
ffn.down_proj  |  ■       ■      ■      ■      ■      ■

Legend: ■ = low (<1e3), ▓ = medium (<1e6), █ = high (<1e9), !!! = danger (>1e9)
```

### Time Series

```python
def plot_curvature_timeseries(profiler, layers_to_plot):
    fig, axes = plt.subplots(len(layers_to_plot), 1, figsize=(12, 4*len(layers_to_plot)))
    
    for ax, layer in zip(axes, layers_to_plot):
        history = profiler.curvature_history[layer]
        ax.semilogy(history, label=layer)
        ax.axhline(1e6, color='orange', linestyle='--', label='Warning')
        ax.axhline(1e9, color='red', linestyle='--', label='Danger')
        ax.set_ylabel('Curvature κ')
        ax.legend()
    
    axes[-1].set_xlabel('Training Step')
    return fig
```

### Correlation with Loss

```python
def plot_curvature_loss_correlation(profiler, loss_history):
    """Show curvature often spikes before loss spikes."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Total curvature
    total_curv = [sum(profiler.curvature_history[l][i] 
                      for l in profiler.curvature_history)
                  for i in range(len(loss_history))]
    
    ax1.semilogy(total_curv)
    ax1.set_ylabel('Total Curvature')
    
    ax2.semilogy(loss_history)
    ax2.set_ylabel('Loss')
    ax2.set_xlabel('Step')
    
    # Mark spikes
    loss_spikes = find_spikes(loss_history)
    curv_spikes = find_spikes(total_curv)
    
    for ls in loss_spikes:
        ax2.axvline(ls, color='red', alpha=0.5)
        # Find preceding curvature spike
        preceding = [cs for cs in curv_spikes if cs < ls and ls - cs < 100]
        if preceding:
            ax1.axvline(preceding[-1], color='green', alpha=0.5)
    
    return fig
```

---

## Validation Strategy

### Experiment 1: Failure Prediction

**Setup:**
1. Train Transformers of varying depths (6, 12, 24, 48 layers)
2. Use learning rates that cause some runs to fail
3. Track curvature and loss

**Metrics:**
- Precision: What fraction of curvature warnings lead to actual failures?
- Recall: What fraction of failures are predicted by curvature spikes?
- Lead time: How many steps before failure do we detect?

**Success Criterion:** >80% recall with >50% precision, >10 step lead time

### Experiment 2: Curvature vs Gradient Norms

**Setup:**
1. Track both curvature and gradient norms
2. Compare as predictors of instability

**Hypothesis:** Curvature is a better predictor because it captures second-order effects

### Experiment 3: Intervention Study

**Setup:**
1. When curvature exceeds threshold, reduce learning rate
2. Compare against fixed learning rate and standard schedulers

**Success Criterion:** Fewer failed runs, comparable final accuracy

---

## API Design

```python
from hnf.profiler import CurvatureProfiler, TrainingMonitor

# Basic profiling
profiler = CurvatureProfiler(model)

for batch in dataloader:
    loss = model(batch)
    curvatures = profiler.compute_curvature(loss)
    
    loss.backward()
    optimizer.step()

# Visualization
profiler.plot_heatmap()
profiler.plot_timeseries(['attention.softmax', 'ffn.up'])

# Real-time monitoring
monitor = TrainingMonitor(model, warn_threshold=1e6)

for batch in dataloader:
    loss = model(batch)
    
    warnings = monitor.on_step(loss)
    if warnings:
        print('\n'.join(warnings))
        
    # Predictive intervention
    failure_predicted, layer, projected = monitor.predict_failure(horizon=100)
    if failure_predicted:
        print(f"Reducing LR: {layer} projected curvature {projected:.2e}")
        for g in optimizer.param_groups:
            g['lr'] *= 0.5
    
    loss.backward()
    optimizer.step()
```

---

## Advanced Features

### 1. Curvature-Aware Learning Rate

Adapt learning rate based on local curvature:

```python
class CurvatureAdaptiveLR:
    """Learning rate scheduler based on curvature."""
    
    def __init__(self, optimizer, profiler, base_lr, curvature_target=1e4):
        self.optimizer = optimizer
        self.profiler = profiler
        self.base_lr = base_lr
        self.curvature_target = curvature_target
    
    def step(self, loss):
        curvatures = self.profiler.compute_curvature(loss)
        max_curv = max(curvatures.values())
        
        # Scale LR inversely with curvature
        ratio = self.curvature_target / max_curv
        new_lr = self.base_lr * min(1.0, ratio)
        
        for g in self.optimizer.param_groups:
            g['lr'] = new_lr
```

### 2. Per-Layer Learning Rates

Different layers may need different learning rates:

```python
def compute_per_layer_lr(curvatures, base_lr):
    """Compute per-layer learning rates based on curvature."""
    lrs = {}
    for layer, κ in curvatures.items():
        # Layers with higher curvature need smaller LR
        lrs[layer] = base_lr / (1 + κ / 1e4)
    return lrs
```

### 3. Curvature Regularization

Add curvature as a regularization term:

```python
def curvature_regularized_loss(model, input, target, loss_fn, profiler, λ=1e-6):
    """Loss with curvature regularization."""
    output = model(input)
    base_loss = loss_fn(output, target)
    
    # Compute curvature (expensive, do periodically)
    curvatures = profiler.compute_curvature(base_loss)
    curv_penalty = sum(curvatures.values())
    
    return base_loss + λ * curv_penalty
```

---

## Compute Requirements

| Task | Time per Step | Overhead |
|------|---------------|----------|
| Forward pass | Baseline | 1x |
| Curvature estimation (10 layers) | ~2x forward | 2x |
| Full profiling | ~3x forward | 3x |

For reduced overhead:
- Profile every 10-100 steps instead of every step
- Sample subset of layers
- Use cheaper approximations

---

## Theoretical Contributions

### 1. Curvature as Training Health Metric

This project validates the theoretical claim that numerical curvature predicts training dynamics. If confirmed, it:
- Provides new diagnostic tool for practitioners
- Suggests curvature-aware optimization algorithms
- Connects HNF theory to practical training

### 2. Predictive Intervention

Using curvature for predictive intervention (reducing LR before failure) is novel. It's like a "check engine light" for training.

### 3. Per-Layer Analysis

Fine-grained per-layer curvature profiles reveal which components cause instability. This guides architecture improvements.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Curvature estimation too expensive | Medium | Periodic sampling, cheaper approximations |
| Curvature doesn't predict failures | Medium | Validate on diverse architectures first |
| Too many false positives | Medium | Tune thresholds, use rate of change |
| Not actionable | Low | Provide specific interventions (reduce LR, prune layer) |

---

## Expected Impact

### For Practitioners

- Early warning system for training instabilities
- Reduced time debugging failed training runs
- Actionable diagnostics (which layer, what to do)

### For Researchers

- Empirical validation of HNF curvature theory
- New metric for comparing architectures
- Insights into training dynamics

### For Tools

- Integration with W&B, TensorBoard, MLflow
- New profiling paradigm (numerical stability, not just speed)
- Foundation for curvature-aware optimization

---

## Next Steps

1. Implement Hessian-vector product efficiently
2. Build power iteration for spectral norm estimation
3. Create `CurvatureProfiler` with hook system
4. Validate on small models with induced failures
5. Build visualization dashboard
