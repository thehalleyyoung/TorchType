# Project 3: Attention Stability Analysis Tool

## Transformer Application: Analyze and Fix Unstable Attention Patterns

**The Problem:** Attention mechanisms are the core of transformers, but they're numerically fragile. Large attention logits cause softmax overflow, small attention scores cause vanishing gradients, and attention entropy collapse can kill training. Current tools don't predict these issues.

**Our Solution:** Geometric analysis of attention patterns using HNF curvature theory. We characterize the stability of attention configurations and predict when attention will become numerically problematic.

**Concrete Use Cases:**
1. **Training monitoring:** "Head 7 in layer 12 is approaching entropy collapse—attention distribution converging to delta"
2. **Architecture design:** "Your 128-dim attention with 64 heads has unstable curvature—try 32 heads or 256-dim"
3. **Debugging attention:** "NaN in layer 15 traced to QK^T logits reaching 89.3 (softmax overflow)"

---

## Executive Summary

Build a tool that analyzes transformer attention patterns for numerical stability using HNF curvature bounds. The tool detects attention entropy collapse, softmax overflow risk, and gradient vanishing—and suggests fixes (temperature scaling, attention clamping, head pruning).

---

## Theoretical Foundation

### Attention Curvature

For attention $A = \text{softmax}(QK^T / \sqrt{d})$:

The curvature bound is:
$$\kappa_{\text{attn}}^{\text{curv}} = O\left(\frac{\|Q\|\|K\|}{d} \cdot e^{2\|QK^T\|_\infty / \sqrt{d}}\right)$$

This predicts:
- **Overflow:** Large $\|QK^T\|_\infty$ → exponential curvature → needs float32
- **Underflow:** Very negative logits → softmax outputs near 0 → gradient vanishing
- **Collapse:** Attention concentrating on few tokens → high curvature in those directions

### Attention Entropy

Define attention entropy for head $h$:
$$H_h = -\sum_j A_{ij} \log A_{ij}$$

Low entropy (< 0.5 nats) indicates collapse—attention focuses on very few tokens, making gradients sparse and training unstable.

### Stability Regions

For each attention configuration, we can compute a "stability region" in (temperature, dimension, sequence length) space where curvature stays below threshold.

---

## Technical Approach

### 1. Attention Statistics Collector

```python
class AttentionAnalyzer:
    """Analyze attention patterns for stability."""
    
    def __init__(self, model):
        self.model = model
        self.hooks = []
        self.attention_stats = defaultdict(list)
        self._register_hooks()
    
    def _register_hooks(self):
        """Hook into all attention layers."""
        for name, module in self.model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                hook = module.register_forward_hook(
                    partial(self._attention_hook, name=name)
                )
                self.hooks.append(hook)
    
    def _attention_hook(self, module, input, output, name):
        """Capture attention weights and compute statistics."""
        # Extract attention weights (implementation varies by model)
        attn_weights = self._extract_attention_weights(module, output)
        
        if attn_weights is not None:
            stats = self._compute_stats(attn_weights)
            self.attention_stats[name].append(stats)
    
    def _compute_stats(self, attn: torch.Tensor) -> dict:
        """Compute stability statistics for attention weights."""
        # Shape: [batch, heads, seq, seq]
        
        # Entropy per head
        entropy = -(attn * torch.log(attn + 1e-10)).sum(-1).mean(0).mean(-1)
        
        # Max attention weight (spikiness)
        max_attn = attn.max(-1)[0].mean()
        
        # Curvature estimate
        logits = torch.log(attn + 1e-10)  # Approximate pre-softmax
        logit_range = logits.max(-1)[0] - logits.min(-1)[0]
        curvature = torch.exp(2 * logit_range).mean()
        
        return {
            'entropy': entropy.cpu().numpy(),
            'max_attention': max_attn.item(),
            'curvature': curvature.item(),
            'logit_range': logit_range.mean().item()
        }
    
    def diagnose(self) -> 'AttentionDiagnosis':
        """Analyze collected statistics and produce diagnosis."""
        issues = []
        
        for layer_name, stats_list in self.attention_stats.items():
            recent = stats_list[-100:] if len(stats_list) > 100 else stats_list
            
            # Check for entropy collapse
            avg_entropy = np.mean([s['entropy'] for s in recent], axis=0)
            for head_idx, h in enumerate(avg_entropy):
                if h < 0.5:
                    issues.append(StabilityIssue(
                        layer=layer_name,
                        head=head_idx,
                        issue_type='entropy_collapse',
                        severity='warning',
                        value=h,
                        suggestion='Consider attention dropout or entropy regularization'
                    ))
            
            # Check for overflow risk
            avg_curvature = np.mean([s['curvature'] for s in recent])
            if avg_curvature > 1e6:
                issues.append(StabilityIssue(
                    layer=layer_name,
                    head=None,
                    issue_type='overflow_risk',
                    severity='error',
                    value=avg_curvature,
                    suggestion='Use float32 for softmax or clamp attention logits'
                ))
            
            # Check for attention spikiness
            avg_max = np.mean([s['max_attention'] for s in recent])
            if avg_max > 0.95:
                issues.append(StabilityIssue(
                    layer=layer_name,
                    head=None,
                    issue_type='attention_spike',
                    severity='warning',
                    value=avg_max,
                    suggestion='Attention nearly one-hot; gradients may vanish'
                ))
        
        return AttentionDiagnosis(issues=issues, stats=self.attention_stats)
```

### 2. Pre-Training Stability Check

```python
def check_attention_stability(model, sample_input, config):
    """Check attention stability before training."""
    
    # Compute theoretical bounds
    d_model = config.hidden_size
    d_head = config.hidden_size // config.num_attention_heads
    n_heads = config.num_attention_heads
    seq_len = sample_input.shape[1]
    
    # Estimate attention logit range
    # For random init: QK^T ~ N(0, d_head) element-wise
    expected_logit_max = np.sqrt(2 * np.log(seq_len)) * np.sqrt(d_head) / np.sqrt(d_head)
    
    # Curvature bound
    curvature_bound = np.exp(2 * expected_logit_max)
    
    # Precision requirement
    precision_bits = np.log2(curvature_bound * seq_len**2 / 1e-6)
    
    report = {
        'expected_logit_max': expected_logit_max,
        'curvature_bound': curvature_bound,
        'precision_bits_needed': precision_bits,
        'float16_safe': precision_bits <= 11,
        'recommendations': []
    }
    
    if not report['float16_safe']:
        report['recommendations'].append(
            f"Attention softmax needs float32 (requires {precision_bits:.1f} bits, float16 has 11)"
        )
    
    if expected_logit_max > 20:
        report['recommendations'].append(
            f"Consider reducing sequence length or using ALiBi/RoPE instead of learned positions"
        )
    
    return report
```

### 3. Training-Time Monitoring

```python
class AttentionStabilityMonitor:
    """Monitor attention stability during training."""
    
    def __init__(self, model, log_every=100):
        self.analyzer = AttentionAnalyzer(model)
        self.log_every = log_every
        self.step = 0
        self.warnings_issued = set()
    
    def on_step(self):
        """Call after each training step."""
        self.step += 1
        
        if self.step % self.log_every == 0:
            diagnosis = self.analyzer.diagnose()
            
            for issue in diagnosis.issues:
                issue_key = (issue.layer, issue.head, issue.issue_type)
                
                if issue_key not in self.warnings_issued:
                    self._log_issue(issue)
                    self.warnings_issued.add(issue_key)
            
            return diagnosis
        
        return None
    
    def get_intervention(self, diagnosis) -> Optional[dict]:
        """Suggest intervention based on diagnosis."""
        
        for issue in diagnosis.issues:
            if issue.severity == 'error':
                if issue.issue_type == 'overflow_risk':
                    return {
                        'action': 'reduce_lr',
                        'factor': 0.5,
                        'reason': f'Overflow risk in {issue.layer}'
                    }
                elif issue.issue_type == 'entropy_collapse':
                    return {
                        'action': 'increase_dropout',
                        'layer': issue.layer,
                        'reason': f'Entropy collapse in head {issue.head}'
                    }
        
        return None
```

---

## Implementation Plan

### Phase 1: Basic Analysis (Week 1-2)

- Attention weight extraction for common architectures (GPT, BERT, LLaMA)
- Basic statistics: entropy, max attention, logit range
- Curvature estimation

### Phase 2: Diagnosis (Week 3-4)

- Issue detection: collapse, overflow, spikes
- Pre-training stability checker
- Recommendations engine

### Phase 3: Monitoring (Week 5-6)

- Training-time integration
- Intervention suggestions
- Integration with W&B/TensorBoard

### Phase 4: Visualization (Week 7-8)

- Attention stability heatmaps
- Per-head entropy over training
- Curvature evolution plots

---

## Validation Strategy

### Experiment 1: Predict Attention Instabilities

**Setup:**
1. Train GPT-2 variants with different attention temperatures
2. Predict which will become unstable
3. Validate against actual training failures

**Success Metric:** >80% of instabilities predicted

### Experiment 2: Fix Instabilities

**Setup:**
1. Take models with known attention issues
2. Apply suggested fixes
3. Measure stability improvement

**Success Metric:** Suggested fixes resolve issues in >70% of cases

### Experiment 3: Real Models

**Setup:**
1. Analyze attention in LLaMA, Mistral, GPT-2
2. Compare theoretical predictions to observed behavior
3. Identify patterns

---

## API Design

```python
from hnf.attention import AttentionAnalyzer, check_attention_stability

# Pre-training check
config = AutoConfig.from_pretrained("gpt2-large")
report = check_attention_stability(model, sample_input, config)
print(report['recommendations'])

# Training-time monitoring
monitor = AttentionStabilityMonitor(model)

for step, batch in enumerate(dataloader):
    loss = train_step(model, batch)
    
    diagnosis = monitor.on_step()
    if diagnosis and diagnosis.has_issues():
        intervention = monitor.get_intervention(diagnosis)
        if intervention:
            apply_intervention(optimizer, intervention)

# Post-hoc analysis
analyzer = AttentionAnalyzer(model)
with torch.no_grad():
    for batch in eval_dataloader:
        model(batch)

diagnosis = analyzer.diagnose()
diagnosis.plot_entropy_heatmap()
diagnosis.plot_curvature_evolution()
```

---

## Expected Impact

### For LLM Training

- Predict attention instabilities before they cause NaNs
- Automated suggestions for fixing attention issues
- Reduce debugging time from days to minutes

### For Model Design

- Understand stability tradeoffs in attention design
- Guide temperature, dimension, head count choices
- Compare stability across architectures

### For Research

- Geometric understanding of attention dynamics
- Connect curvature theory to attention mechanisms
- New tools for studying transformer behavior
