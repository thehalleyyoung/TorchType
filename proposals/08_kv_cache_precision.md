# Project 8: KV-Cache Precision Analyzer

## Transformer Application: Optimize Memory vs. Quality for Long-Context Inference

**Use case:** KV-cache is the dominant memory cost during transformer inference. This tool analyzes which layers and positions can use lower precision for their cached keys/values without quality degradation, enabling 2-4x longer context windows with the same memory.

### The Problem with KV-Cache

During autoregressive generation:
- Every previous token's keys and values are cached
- Memory grows linearly with sequence length: $O(\text{layers} \times \text{seq\_len} \times d_{model})$
- For GPT-4 scale models with 128K context, KV-cache can exceed 100GB
- Most systems use FP16 uniformly, wasting memory on positions that could use less

### Current Solutions Are Suboptimal

- **Uniform quantization:** INT8 everywhere loses quality on critical positions
- **Sliding window:** Drops old context entirely, breaking long-range tasks
- **Grouped-query attention:** Reduces memory but changes architecture
- **No precision analysis:** Nobody knows which positions actually need high precision

### This Tool Tells You Exactly What Precision Each Position Needs

```python
from kv_cache_precision import KVCacheAnalyzer

analyzer = KVCacheAnalyzer(model)

# Analyze on representative prompts
analysis = analyzer.analyze(
    calibration_prompts=[
        "Long document summarization...",
        "Multi-turn conversation...",
        "Code completion context..."
    ],
    quality_threshold=0.99  # 99% output quality preservation
)

# Output:
# ╔════════════════════════════════════════════════════════════════╗
# ║ KV-CACHE PRECISION MAP                                          ║
# ╠════════════════════════════════════════════════════════════════╣
# ║ Layer │ Positions 0-128  │ Positions 128-1K │ Positions 1K+   ║
# ╠═══════╪══════════════════╪══════════════════╪═════════════════╣
# ║   0   │     FP16         │      INT8        │     INT4        ║
# ║   1   │     FP16         │      INT8        │     INT4        ║
# ║  ...  │      ...         │       ...        │      ...        ║
# ║  11   │     FP16         │      FP16        │     INT8        ║  ← Critical!
# ╠════════════════════════════════════════════════════════════════╣
# ║ Memory Savings: 2.7x (from 16GB to 5.9GB for 8K context)       ║
# ║ Quality Preserved: 99.3%                                        ║
# ║                                                                 ║
# ║ ⚠️  Layer 11 is precision-critical for long-range attention    ║
# ║    Recommend: Keep FP16 for positions with attention > 0.1     ║
# ╚════════════════════════════════════════════════════════════════╝
```

---

## Theoretical Foundation

### KV-Cache Curvature Analysis

The key insight is that different positions contribute differently to the output. Let:
- $K_t, V_t$ = cached key/value at position $t$
- $q$ = current query
- $\alpha_t = \text{softmax}(q \cdot K_t / \sqrt{d})$ = attention weight

**Curvature contribution from position $t$:**
$$\kappa_t^{KV} = \alpha_t \cdot \left\| \frac{\partial \text{output}}{\partial K_t} \right\|$$

Positions with low $\kappa_t^{KV}$:
- Have low attention weight
- Are far from current position
- Contribute to easily-recoverable patterns

These positions can use lower precision.

### Precision Requirement Per Position

From Theorem 5.7:
$$p_t \geq \log_2\left(\frac{\kappa_t^{KV} \cdot \|K_t\|}{\varepsilon}\right)$$

This gives a **per-position precision map**: different bits for different cache entries.

### Attention Locality Patterns

Transformers exhibit predictable attention patterns:
1. **Recency bias:** Recent positions get more attention
2. **Positional anchors:** First few tokens (BOS, system prompt) often critical
3. **Semantic clustering:** Related tokens attend to each other regardless of distance
4. **Layer variation:** Early layers → local, late layers → global

We can exploit these patterns for precision allocation.

---

## Technical Approach

### 1. Curvature-Based Position Scoring

```python
class KVCacheAnalyzer:
    """Analyze KV-cache precision requirements."""
    
    def __init__(self, model):
        self.model = model
        self.n_layers = model.config.n_layer
        self.attention_patterns = {}
        self.curvature_scores = {}
    
    def analyze(self, calibration_prompts, quality_threshold=0.99):
        """
        Analyze which cache positions need high precision.
        
        Returns position-wise precision map.
        """
        # Step 1: Collect attention patterns
        self._collect_attention_patterns(calibration_prompts)
        
        # Step 2: Compute curvature contribution per position
        self._compute_position_curvatures()
        
        # Step 3: Determine precision requirements
        precision_map = self._compute_precision_map(quality_threshold)
        
        return precision_map
    
    def _collect_attention_patterns(self, prompts):
        """Hook into attention layers to collect patterns."""
        hooks = []
        
        for layer_idx in range(self.n_layers):
            hook = self.model.transformer.h[layer_idx].attn.register_forward_hook(
                lambda m, i, o, idx=layer_idx: self._attention_hook(idx, o)
            )
            hooks.append(hook)
        
        for prompt in prompts:
            self.model.generate(prompt, max_length=1024)
        
        for hook in hooks:
            hook.remove()
    
    def _attention_hook(self, layer_idx, output):
        """Record attention weights."""
        attn_weights = output[1]  # (batch, heads, seq, seq)
        
        if layer_idx not in self.attention_patterns:
            self.attention_patterns[layer_idx] = []
        
        self.attention_patterns[layer_idx].append(attn_weights.detach())
    
    def _compute_position_curvatures(self):
        """Compute curvature contribution for each cached position."""
        for layer_idx in range(self.n_layers):
            patterns = torch.cat(self.attention_patterns[layer_idx], dim=0)
            
            # Average attention to each position across all queries
            avg_attention = patterns.mean(dim=(0, 1, 2))  # (seq_len,)
            
            # Curvature scales with attention weight and position's norm
            # High attention = high curvature = needs more precision
            self.curvature_scores[layer_idx] = avg_attention
    
    def _compute_precision_map(self, quality_threshold):
        """Determine per-position precision requirements."""
        precision_map = {}
        
        for layer_idx in range(self.n_layers):
            scores = self.curvature_scores[layer_idx]
            
            # Positions with curvature > threshold get FP16
            # Others can use INT8 or INT4
            precision_map[layer_idx] = self._scores_to_precision(
                scores, quality_threshold
            )
        
        return precision_map
    
    def _scores_to_precision(self, scores, threshold):
        """Convert curvature scores to precision requirements."""
        # Normalize scores
        max_score = scores.max()
        normalized = scores / max_score
        
        precisions = []
        for score in normalized:
            if score > 0.5:
                precisions.append(16)  # FP16
            elif score > 0.1:
                precisions.append(8)   # INT8
            else:
                precisions.append(4)   # INT4
        
        return precisions
```

### 2. Adaptive KV-Cache Implementation

```python
class AdaptivePrecisionKVCache:
    """KV-Cache with per-position precision control."""
    
    def __init__(self, precision_map, max_length, n_layers, n_heads, head_dim):
        self.precision_map = precision_map
        self.max_length = max_length
        self.n_layers = n_layers
        
        # Allocate cache with mixed precision storage
        self.caches = {}
        for layer_idx in range(n_layers):
            self.caches[layer_idx] = {
                'keys': MixedPrecisionBuffer(max_length, n_heads * head_dim),
                'values': MixedPrecisionBuffer(max_length, n_heads * head_dim)
            }
    
    def update(self, layer_idx, position, key, value):
        """Add new key/value to cache with appropriate precision."""
        precision = self._get_precision(layer_idx, position)
        
        self.caches[layer_idx]['keys'].write(
            position, key, precision=precision
        )
        self.caches[layer_idx]['values'].write(
            position, value, precision=precision
        )
    
    def get(self, layer_idx, positions):
        """Retrieve cached keys/values, dequantizing as needed."""
        keys = self.caches[layer_idx]['keys'].read(positions)
        values = self.caches[layer_idx]['values'].read(positions)
        return keys, values
    
    def memory_usage(self):
        """Report actual memory usage (less than uniform FP16)."""
        total_bits = 0
        for layer_idx in range(self.n_layers):
            for pos, prec in enumerate(self.precision_map[layer_idx]):
                total_bits += 2 * prec * self.head_dim  # K and V
        return total_bits / 8  # bytes


class MixedPrecisionBuffer:
    """Buffer that stores different positions at different precisions."""
    
    def __init__(self, max_length, dim):
        self.max_length = max_length
        self.dim = dim
        
        # Separate storage for each precision level
        self.fp16_data = {}  # position -> tensor
        self.int8_data = {}
        self.int8_scales = {}
        self.int4_data = {}
        self.int4_scales = {}
    
    def write(self, position, tensor, precision):
        """Store tensor at given precision."""
        if precision == 16:
            self.fp16_data[position] = tensor.half()
        elif precision == 8:
            # INT8 quantization
            scale = tensor.abs().max() / 127
            quantized = (tensor / scale).round().to(torch.int8)
            self.int8_data[position] = quantized
            self.int8_scales[position] = scale
        elif precision == 4:
            # INT4 quantization (packed)
            scale = tensor.abs().max() / 7
            quantized = (tensor / scale).round().clamp(-8, 7).to(torch.int8)
            self.int4_data[position] = quantized
            self.int4_scales[position] = scale
    
    def read(self, positions):
        """Read and dequantize tensors."""
        results = []
        for pos in positions:
            if pos in self.fp16_data:
                results.append(self.fp16_data[pos].float())
            elif pos in self.int8_data:
                dequant = self.int8_data[pos].float() * self.int8_scales[pos]
                results.append(dequant)
            elif pos in self.int4_data:
                dequant = self.int4_data[pos].float() * self.int4_scales[pos]
                results.append(dequant)
        return torch.stack(results)
```

### 3. Dynamic Precision Adjustment

```python
class DynamicKVPrecision:
    """Adjust precision on-the-fly based on attention patterns."""
    
    def __init__(self, model, initial_precision=8):
        self.model = model
        self.initial_precision = initial_precision
        self.position_importance = {}
    
    def update_importance(self, layer_idx, attention_weights):
        """
        Update position importance based on current attention.
        
        Positions that receive high attention get upgraded to higher precision.
        Positions that haven't been attended to get downgraded.
        """
        # Get attention to each cached position
        current_attention = attention_weights[:, :, -1, :]  # Current query's attention
        avg_attention = current_attention.mean(dim=(0, 1))  # Average over batch and heads
        
        if layer_idx not in self.position_importance:
            self.position_importance[layer_idx] = avg_attention
        else:
            # Exponential moving average
            alpha = 0.1
            self.position_importance[layer_idx] = (
                alpha * avg_attention + 
                (1 - alpha) * self.position_importance[layer_idx]
            )
    
    def get_precision(self, layer_idx, position):
        """Get current precision for a position based on importance."""
        if layer_idx not in self.position_importance:
            return self.initial_precision
        
        importance = self.position_importance[layer_idx][position]
        
        if importance > 0.1:
            return 16  # Critical position
        elif importance > 0.01:
            return 8   # Moderately important
        else:
            return 4   # Can compress heavily
```

---

## Validation Experiments

### Experiment 1: Memory Savings on LLaMA-7B

```python
def measure_memory_savings():
    model = load_llama_7b()
    analyzer = KVCacheAnalyzer(model)
    
    # Analyze on diverse prompts
    precision_map = analyzer.analyze(
        calibration_prompts=load_diverse_prompts(),
        quality_threshold=0.99
    )
    
    # Compare memory usage
    uniform_fp16 = 32 * 4096 * 4096 * 2  # layers * seq * dim * 2 bytes
    adaptive = AdaptivePrecisionKVCache(precision_map, ...).memory_usage()
    
    print(f"Uniform FP16: {uniform_fp16 / 1e9:.1f} GB")
    print(f"Adaptive: {adaptive / 1e9:.1f} GB")
    print(f"Savings: {uniform_fp16 / adaptive:.1f}x")
```

**Expected results:**
- 2-4x memory reduction for typical workloads
- Larger savings for longer contexts (distant positions less important)
- Quality preserved within 1% on standard benchmarks

### Experiment 2: Quality vs. Compression Trade-off

```python
def quality_compression_curve():
    thresholds = [0.95, 0.99, 0.999, 0.9999]
    
    for thresh in thresholds:
        precision_map = analyzer.analyze(quality_threshold=thresh)
        memory = compute_memory(precision_map)
        quality = evaluate_on_benchmark(model_with_cache(precision_map))
        
        print(f"Threshold {thresh}: {memory:.1f}GB, Quality {quality:.3f}")
```

### Experiment 3: Long-Context Tasks

Show that adaptive precision enables longer context than uniform precision:

```python
def max_context_comparison():
    memory_budget = 16  # GB
    
    # Uniform FP16: limited context
    uniform_max_len = memory_budget / (32 * 4096 * 2 / 1e9)  # ~4K tokens
    
    # Adaptive: longer context with same memory
    # Analysis shows 60% of positions can use INT4
    adaptive_max_len = 4 * uniform_max_len  # ~16K tokens
```

---

## API Design

```python
# Simple usage
from kv_cache_precision import optimize_kv_cache

model = load_model()
optimized_model = optimize_kv_cache(
    model,
    calibration_data=your_prompts,
    memory_budget="8GB",  # or quality_threshold=0.99
)

# Generate with optimized cache
output = optimized_model.generate(long_prompt, max_length=32768)
```

---

## Connection to HNF Paper

This project applies Theorem 5.7 to the specific case of transformer KV-cache:

1. **Curvature analysis** identifies which cached positions are precision-critical
2. **Precision bounds** determine minimum bits per position
3. **Sheaf perspective**: The precision requirement varies over the cache, forming a sheaf over position indices

The result: theoretically-grounded compression that preserves quality by allocating bits where they matter.

---

## Implementation Milestones

1. **Week 1-2:** Attention pattern collection and curvature scoring
2. **Week 3-4:** Precision map computation and validation
3. **Week 5-6:** Mixed-precision cache implementation
4. **Week 7-8:** Dynamic precision adjustment
5. **Week 9-10:** Integration with inference engines (vLLM, TensorRT-LLM)
6. **Week 11-12:** Benchmarking and paper writing
