# Project 7: Curvature-Adaptive Learning Rate for Transformers

## Transformer Application: Smarter Learning Rate Scheduling Based on Numerical Geometry

**Use case:** Replace heuristic learning rate schedules (linear warmup, cosine decay) with principled curvature-based adaptation. When the transformer enters high-curvature regions (attention instability, LayerNorm variance collapse), automatically reduce step size. Get faster convergence with fewer hyperparameters.

### The Problem with Transformer LR Scheduling

Transformer training uses complex LR schedules with many hyperparameters:
- **Warmup steps:** Usually 1-10% of training. Why? "It works"
- **Peak LR:** Heavily tuned per model size. No principled selection
- **Decay schedule:** Cosine, linear, inverse sqrt—all arbitrary choices
- **Batch size scaling:** Rules like "sqrt scaling" are empirical approximations

These schedules are designed for the *average* loss landscape, not *your* model's specific geometry.

### This Tool Adapts LR to Your Model's Curvature

```python
from homotopy_lr import CurvatureAdaptiveLR

# Standard transformer training setup
model = TransformerLM(config)
optimizer = AdamW(model.parameters())

# Replace arbitrary schedules with curvature-based adaptation
scheduler = CurvatureAdaptiveLR(
    optimizer,
    model,
    base_lr=1e-4,
    curvature_target=1e3,  # Target condition number
    adaptation_rate=0.1    # How fast to adapt
)

for batch in dataloader:
    loss = model(batch)
    loss.backward()
    
    # Scheduler estimates curvature and adapts LR
    scheduler.step(loss)
    
    # Visualization shows:
    # - When LR drops: high curvature in attention layers
    # - When LR increases: flat region, can take bigger steps
    # - Warmup emerges naturally from initial high curvature!
    
    optimizer.step()
```

**Result:** Warmup happens automatically (high initial curvature → low LR), and the schedule adapts to your specific model rather than using generic heuristics.

---

## Theoretical Foundation

### Training as Path Lifting

In the HNF framework, training traces a path in two spaces simultaneously:

**Loss space:** $\gamma: [0, T] \to \mathcal{L}$ — the loss decreases over time

**Parameter space:** $\tilde{\gamma}: [0, T] \to \Theta$ — the parameters evolve

The relationship is governed by gradient flow:
$$\frac{d\tilde{\gamma}}{dt} = -\eta \cdot \nabla_\theta L(\tilde{\gamma}(t))$$

### Curvature and Step Size in Transformers

At each point $\theta$ in parameter space, the transformer has local curvature:
$$\kappa^{\mathrm{curv}}(\theta) = \|\nabla^2 L(\theta)\| \cdot \|\nabla L(\theta)\|^{-2}$$

For transformers, curvature comes from:
- **Attention softmax:** $\kappa_{\text{attn}} \propto e^{2 \cdot \max(QK^T)}$
- **LayerNorm:** $\kappa_{\text{LN}} \propto 1/\sigma^2$
- **Cross-entropy:** $\kappa_{\text{CE}} \propto 1/p_{\min}$

**Key insight:** The learning rate should adapt inversely to curvature:
$$\eta(t) \propto \frac{1}{\kappa^{\mathrm{curv}}(\tilde{\gamma}(t))}$$

### Why This Explains Warmup

At initialization:
- Random attention patterns → high softmax curvature
- Untrained predictions → low confidence → high cross-entropy curvature
- Result: $\kappa$ is high, so optimal LR is low

After warmup:
- Attention patterns stabilize → lower curvature
- Predictions improve → lower cross-entropy curvature
- Result: $\kappa$ decreases, so optimal LR can increase

**Curvature-based LR naturally produces warmup without explicit scheduling!**

---

## Technical Approach

### 1. Curvature Estimation During Training

```python
class CurvatureEstimator:
    """Estimate local curvature of loss landscape."""
    
    def __init__(self, model, loss_fn, method='hutchinson'):
        self.model = model
        self.loss_fn = loss_fn
        self.method = method
        self.curvature_history = []
    
    def estimate(self, inputs, targets) -> float:
        """
        Estimate curvature at current parameter point.
        
        Returns:
            κ = ||∇²L|| / ||∇L||²
        """
        # Compute gradient
        loss = self.loss_fn(self.model(inputs), targets)
        grads = torch.autograd.grad(loss, self.model.parameters(), 
                                     create_graph=True)
        grad_norm = sum(g.norm() ** 2 for g in grads).sqrt()
        
        # Estimate Hessian norm
        if self.method == 'hutchinson':
            hess_norm = self._hutchinson_hessian_norm(loss, grads)
        elif self.method == 'power':
            hess_norm = self._power_iteration_hessian_norm(loss)
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        κ = hess_norm / (grad_norm ** 2 + 1e-8)
        self.curvature_history.append(κ.item())
        return κ.item()
    
    def _hutchinson_hessian_norm(self, loss, grads, n_samples=10):
        """
        Estimate ||∇²L|| using Hutchinson's trace estimator.
        """
        params = list(self.model.parameters())
        estimates = []
        
        for _ in range(n_samples):
            # Random vector
            v = [torch.randn_like(p) for p in params]
            
            # Hessian-vector product
            Hv = torch.autograd.grad(
                sum((g * vi).sum() for g, vi in zip(grads, v)),
                params,
                retain_graph=True
            )
            
            # <v, Hv> estimates trace(H) when v is Rademacher
            estimate = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv))
            estimates.append(estimate)
        
        # Spectral norm ≤ sqrt(n) * max|eigenvalue| ≈ trace
        return torch.tensor(estimates).abs().max()
    
    def _power_iteration_hessian_norm(self, loss, n_iterations=5):
        """
        Estimate largest eigenvalue of Hessian via power iteration.
        """
        params = list(self.model.parameters())
        
        # Random initial vector
        v = [torch.randn_like(p) for p in params]
        v = self._normalize(v)
        
        for _ in range(n_iterations):
            # Compute ∇²L @ v
            grad = torch.autograd.grad(loss, params, create_graph=True)
            grad_v = sum((g * vi).sum() for g, vi in zip(grad, v))
            Hv = torch.autograd.grad(grad_v, params, retain_graph=True)
            
            # Update
            eigenvalue = sum((vi * hvi).sum() for vi, hvi in zip(v, Hv))
            v = self._normalize(Hv)
        
        return eigenvalue.abs()
    
    def _normalize(self, tensors):
        norm = sum(t.norm() ** 2 for t in tensors).sqrt()
        return [t / norm for t in tensors]
```

### 2. Homotopy Learning Rate Scheduler

```python
class HomotopyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler based on loss landscape curvature.
    
    η(t) = η_base / (1 + α * κ(t) / κ_target)
    
    When curvature exceeds target, LR decreases.
    When curvature is below target, LR approaches base.
    """
    
    def __init__(self, optimizer, model, loss_fn, 
                 base_lr=0.01,
                 curvature_target=1e4,
                 alpha=1.0,
                 estimation_freq=10,
                 smoothing=0.9):
        
        self.base_lr = base_lr
        self.curvature_target = curvature_target
        self.alpha = alpha
        self.estimation_freq = estimation_freq
        self.smoothing = smoothing
        
        self.estimator = CurvatureEstimator(model, loss_fn)
        self.current_curvature = curvature_target
        self.step_count = 0
        
        super().__init__(optimizer)
    
    def update_curvature(self, inputs, targets):
        """Call each step to update curvature estimate."""
        self.step_count += 1
        
        if self.step_count % self.estimation_freq == 0:
            new_curv = self.estimator.estimate(inputs, targets)
            # Exponential smoothing
            self.current_curvature = (
                self.smoothing * self.current_curvature +
                (1 - self.smoothing) * new_curv
            )
    
    def get_lr(self):
        """Compute learning rate based on curvature."""
        ratio = self.current_curvature / self.curvature_target
        scale = 1.0 / (1.0 + self.alpha * max(0, ratio - 1))
        return [self.base_lr * scale for _ in self.base_lrs]
```

### 3. Adaptive Variants

```python
class AdaptiveHomotopyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Adaptive version that learns curvature target from history.
    """
    
    def __init__(self, optimizer, model, loss_fn,
                 base_lr=0.01,
                 warmup_steps=1000,
                 adaptation_rate=0.01):
        
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.adaptation_rate = adaptation_rate
        
        self.estimator = CurvatureEstimator(model, loss_fn)
        self.curvature_history = []
        self.curvature_target = None
        
        super().__init__(optimizer)
    
    def update_curvature(self, inputs, targets):
        κ = self.estimator.estimate(inputs, targets)
        self.curvature_history.append(κ)
        
        # After warmup, set target as percentile
        if len(self.curvature_history) == self.warmup_steps:
            self.curvature_target = np.percentile(self.curvature_history, 75)
        
        # Adapt target slowly
        if self.curvature_target is not None:
            self.curvature_target = (
                (1 - self.adaptation_rate) * self.curvature_target +
                self.adaptation_rate * np.percentile(self.curvature_history[-100:], 75)
            )
    
    def get_lr(self):
        if self.curvature_target is None:
            # During warmup, use linear warmup
            progress = len(self.curvature_history) / self.warmup_steps
            return [self.base_lr * progress for _ in self.base_lrs]
        
        κ = self.curvature_history[-1] if self.curvature_history else 1.0
        ratio = κ / self.curvature_target
        scale = 1.0 / (1.0 + max(0, ratio - 1))
        return [self.base_lr * scale for _ in self.base_lrs]
```

### 4. Per-Layer Learning Rates

```python
class PerLayerHomotopyLR:
    """
    Different learning rates for each layer based on per-layer curvature.
    """
    
    def __init__(self, model, loss_fn, base_lr=0.01):
        self.model = model
        self.loss_fn = loss_fn
        self.base_lr = base_lr
        
        # Per-layer estimators
        self.layer_estimators = {}
        for name, module in model.named_modules():
            if list(module.parameters()):
                self.layer_estimators[name] = LayerCurvatureEstimator(module, loss_fn)
        
        self.layer_curvatures = {name: 1.0 for name in self.layer_estimators}
    
    def get_per_layer_lr(self, inputs, targets):
        """Compute per-layer learning rates."""
        # Estimate curvature for each layer
        for name, estimator in self.layer_estimators.items():
            self.layer_curvatures[name] = estimator.estimate(inputs, targets)
        
        # Normalize so median curvature gets base_lr
        median_curv = np.median(list(self.layer_curvatures.values()))
        
        lrs = {}
        for name, κ in self.layer_curvatures.items():
            ratio = κ / median_curv
            lrs[name] = self.base_lr / (1 + ratio)
        
        return lrs
```

---

## Implementation Plan

### Phase 1: Curvature Estimation (Week 1-2)

**Deliverables:**
- `CurvatureEstimator` with Hutchinson and power iteration methods
- Integration with training loop
- Benchmarking estimation overhead

**Validation:**
- Compare estimates against full Hessian (small models)
- Measure accuracy vs cost tradeoff

### Phase 2: Basic Scheduler (Week 3-4)

**Deliverables:**
- `HomotopyLR` scheduler
- Integration with PyTorch optimizers
- Logging and visualization

**Validation:**
- Compare against constant LR on simple problems
- Verify LR decreases in high-curvature regions

### Phase 3: Adaptive Variants (Week 5-6)

**Deliverables:**
- `AdaptiveHomotopyLR` with learned target
- Per-layer LR scheduler
- Warmup strategies

**Validation:**
- Compare against cosine annealing, step decay
- Test on variety of architectures

### Phase 4: Benchmarking (Week 7-8)

**Deliverables:**
- Comprehensive benchmarks on CIFAR, ImageNet subset
- Comparison with standard schedulers
- Documentation and examples

---

## Algorithm Analysis

### Convergence Guarantee

Under standard assumptions (L-smooth, μ-strongly convex):

With constant LR $\eta = 1/L$, convergence rate is $(1 - \mu/L)^t$.

With homotopy LR $\eta(t) = 1/\kappa(t)$ where $\kappa(t)$ is local curvature:
- In flat regions: $\eta$ large, fast progress
- In curved regions: $\eta$ small, stable but slow
- Overall: adaptive to local geometry

**Theorem (informal):** Homotopy LR achieves convergence rate at least as good as constant LR, and strictly better when curvature varies significantly across the landscape.

### Overhead Analysis

| Operation | Cost |
|-----------|------|
| Forward pass | 1 forward |
| Backward pass | 1 backward |
| Curvature estimation | 1 Hessian-vector product × n_samples |
| Total per step | ~1.2x if estimating every 10 steps |

With estimation every 10 steps and 5 samples:
$$\text{overhead} = 0.1 \times 5 \times 0.5 \times \text{backward} = 0.25 \times \text{backward}$$

About 12% total overhead.

---

## Validation Strategy

### Experiment 1: Synthetic Convex

**Setup:**
- Quadratic loss with known curvature
- Compare HomotopyLR vs constant LR vs Adam

**Success Metric:** HomotopyLR matches or beats Adam

### Experiment 2: CIFAR-10

**Setup:**
- ResNet-18 on CIFAR-10
- Compare:
  - Constant LR
  - Step decay
  - Cosine annealing
  - HomotopyLR

**Metrics:**
- Final accuracy
- Training stability (variance of loss)
- Convergence speed (steps to 90% accuracy)

**Success Metric:** HomotopyLR achieves comparable accuracy with fewer epochs or better stability

### Experiment 3: Transformer Training

**Setup:**
- Small Transformer (6 layers) on WikiText-2
- Known to be sensitive to LR scheduling

**Hypothesis:** HomotopyLR prevents loss spikes by reducing LR in high-curvature regions

**Success Metric:** Fewer loss spikes, comparable final perplexity

### Experiment 4: Varying Architecture Depth

**Setup:**
- Train models of depth 6, 12, 24, 48
- Deeper models have more unstable curvature

**Hypothesis:** HomotopyLR benefit increases with depth

---

## API Design

```python
from hnf.schedulers import HomotopyLR, AdaptiveHomotopyLR

# Basic usage
scheduler = HomotopyLR(
    optimizer,
    model,
    loss_fn=nn.CrossEntropyLoss(),
    base_lr=0.1,
    curvature_target=1e4
)

for epoch in range(100):
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        optimizer.zero_grad()
        loss = loss_fn(model(inputs), targets)
        loss.backward()
        
        # Update curvature estimate (before step)
        scheduler.update_curvature(inputs, targets)
        
        optimizer.step()
        scheduler.step()
        
        if batch_idx % 100 == 0:
            print(f"LR: {scheduler.get_lr()[0]:.6f}, "
                  f"Curvature: {scheduler.current_curvature:.2e}")

# With logging
scheduler.plot_curvature_history()
scheduler.plot_lr_history()
```

---

## Advanced Features

### 1. Curvature-Aware Warmup

```python
class CurvatureWarmup:
    """
    Warmup that adapts to initial curvature landscape.
    """
    
    def __init__(self, optimizer, model, loss_fn, 
                 target_lr, warmup_steps=1000):
        self.estimator = CurvatureEstimator(model, loss_fn)
        self.target_lr = target_lr
        self.warmup_steps = warmup_steps
        
        # Start with very small LR
        self.current_lr = target_lr / 100
        
    def step(self, inputs, targets):
        κ = self.estimator.estimate(inputs, targets)
        
        # Increase LR only if curvature is manageable
        if κ < 1e6:
            self.current_lr = min(
                self.current_lr * 1.01,
                self.target_lr
            )
        else:
            # Don't increase LR in high-curvature regions
            pass
        
        return self.current_lr
```

### 2. Curvature-Aware Gradient Clipping

```python
def curvature_aware_clip(grads, curvature, max_norm=1.0):
    """
    Clip gradients based on local curvature.
    
    In high-curvature regions, clip more aggressively.
    """
    effective_max = max_norm / (1 + curvature / 1e4)
    
    grad_norm = sum(g.norm() ** 2 for g in grads).sqrt()
    if grad_norm > effective_max:
        scale = effective_max / grad_norm
        return [g * scale for g in grads]
    return grads
```

### 3. Integration with Adam

```python
class HomotopyAdam(torch.optim.Adam):
    """
    Adam with curvature-adaptive learning rate.
    """
    
    def __init__(self, params, lr=0.001, curvature_estimator=None, **kwargs):
        super().__init__(params, lr=lr, **kwargs)
        self.curvature_estimator = curvature_estimator
        self.base_lr = lr
    
    def step(self, closure=None, inputs=None, targets=None):
        if self.curvature_estimator and inputs is not None:
            κ = self.curvature_estimator.estimate(inputs, targets)
            
            # Adapt LR
            for group in self.param_groups:
                group['lr'] = self.base_lr / (1 + κ / 1e4)
        
        super().step(closure)
```

---

## Theoretical Connections

### 1. Natural Gradient

The natural gradient uses Fisher information to adapt step size:
$$\theta_{t+1} = \theta_t - \eta F^{-1} \nabla L$$

Our approach is related but uses Hessian curvature instead of Fisher. For MSE loss, they're equivalent.

### 2. Riemannian Optimization

On a Riemannian manifold with metric $g$, the geodesic distance relates to curvature. Our homotopy LR can be seen as adapting step size based on manifold curvature.

### 3. Stochastic Differential Equations

Training as SDE:
$$d\theta_t = -\nabla L(\theta_t) dt + \sigma dW_t$$

Curvature affects the drift term's stability. Our scheduler corresponds to using a curvature-dependent time step.

---

## Compute Requirements

| Task | Time | Hardware |
|------|------|----------|
| Curvature estimation (5 samples) | ~0.5 backward | GPU |
| Per-step overhead | ~2-5% if every 10 steps | GPU |
| Training with scheduler | ~1.05x baseline | GPU |

Mac laptop sufficient for small experiments; GPU recommended for full-scale validation.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Curvature estimation noisy | Medium | Smoothing, more samples |
| Overhead too high | Low | Sparse estimation (every 50 steps) |
| Doesn't help in practice | Medium | Validate on known hard cases first |
| Hyperparameter sensitivity | Medium | Adaptive variant, default presets |

---

## Expected Impact

### For Practitioners

- Automatic LR adaptation without tuning schedule
- More stable training on difficult architectures
- Reduced need for LR search

### For Research

- Connection between geometry and optimization
- New tool for understanding training dynamics
- Foundation for curvature-aware algorithms

### For Theory

- Validates HNF prediction that curvature guides optimization
- Provides empirical data on loss landscape geometry
- Opens new research direction

---

## Next Steps

1. Implement Hutchinson's estimator for Hessian norm
2. Build basic `HomotopyLR` scheduler
3. Validate on convex problems
4. Test on CIFAR-10 with ResNet
5. Compare against standard schedulers
