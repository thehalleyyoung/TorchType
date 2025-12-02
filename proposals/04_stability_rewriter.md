# Project 4: Stability-Preserving Graph Rewriter

## Transformer Application: Automatic Fusion and Stability Optimization

**Use case:** Automatically rewrite transformer computation graphs to fuse operations and improve numerical stability. Discovers FlashAttention-like optimizations, fuses LayerNorm with linear projections, and stabilizes attention patterns—all without manual optimization.

### The Problem with Custom Transformers

When you modify transformer architectures for research or fine-tuning:
- Standard attention implementation is numerically unstable for long sequences
- Naive cross-entropy with softmax loses precision in float16
- Custom attention variants (linear attention, sliding window) often have hidden numerical issues
- Layer fusion opportunities are missed, wasting memory bandwidth

### This Tool Automatically Finds Stable Rewrites

```python
# Input: Your naive transformer code
def naive_attention(Q, K, V):
    scores = Q @ K.T / sqrt(d_k)
    weights = exp(scores) / exp(scores).sum(-1)  # Unstable!
    return weights @ V

# Output: Automatically rewritten stable version
def stable_attention(Q, K, V):
    scores = Q @ K.T / sqrt(d_k)
    scores_stable = scores - scores.max(-1, keepdim=True)  # Inserted
    weights = softmax(scores_stable)  # Fused
    return weights @ V

# Tool reports:
# [REWRITE] Fused exp(x)/sum(exp(x)) -> softmax (κ: 1e6 -> 1e2)
# [REWRITE] Inserted max-subtraction for numerical stability
# [SAVED] 2x memory, 30% faster, stable in float16
```

---

## Theoretical Foundation

### The Rewriting Problem

Two programs are **mathematically equivalent** if they compute the same function in exact arithmetic:
$$f_1 \equiv f_2 \iff \forall x \in \mathbb{R}^n: f_1(x) = f_2(x)$$

But they may have very different **numerical properties**:
- Naive softmax: $\frac{e^{x_i}}{\sum_j e^{x_j}}$ — overflows when $x_i > 89$ in float32
- Stable softmax: $\frac{e^{x_i - x_{\max}}}{\sum_j e^{x_j - x_{\max}}}$ — stable for all inputs

### Stability Metric

For a computation graph $G$, define total curvature:
$$\kappa_G^{\mathrm{total}} = \sum_{v \in G} \kappa_v^{\mathrm{curv}}$$

Lower total curvature = better numerical stability.

### Transformer-Specific Rewrite Rules

**Attention stabilization:**
$$\frac{e^{QK^T/\sqrt{d}}}{\sum e^{QK^T/\sqrt{d}}} \leftrightarrow \mathrm{stable\_softmax}(QK^T/\sqrt{d})$$

**Cross-entropy fusion:**
$$-\log(\mathrm{softmax}(x)_y) \leftrightarrow \mathrm{log\_softmax}(x)_y$$

**LayerNorm fusion:**
$$\mathrm{LayerNorm}(Wx + b) \leftrightarrow W' \cdot \mathrm{LayerNorm}(x) + b'$$

**Attention fusion (FlashAttention-style):**
$$\mathrm{softmax}(QK^T) \cdot V \leftrightarrow \mathrm{fused\_attention}(Q, K, V)$$

---

## Technical Approach

### 1. Computation Graph IR

```python
from dataclasses import dataclass
from typing import List, Optional, Union
from enum import Enum

class OpType(Enum):
    ADD = "add"
    MUL = "mul"
    DIV = "div"
    EXP = "exp"
    LOG = "log"
    SQRT = "sqrt"
    NEG = "neg"
    MATMUL = "matmul"
    SUM = "sum"
    MAX = "max"
    RELU = "relu"
    
@dataclass
class Node:
    id: str
    op: OpType
    inputs: List[str]
    attrs: dict = None  # axis for sum, etc.
    
@dataclass
class Graph:
    nodes: dict[str, Node]
    inputs: List[str]
    outputs: List[str]
    
    def topological_order(self) -> List[str]:
        """Return nodes in topological order."""
        ...
    
    def subgraph(self, node_ids: set) -> 'Graph':
        """Extract subgraph containing given nodes."""
        ...
    
    def replace(self, old_ids: set, new_graph: 'Graph') -> 'Graph':
        """Replace subgraph with equivalent new graph."""
        ...
```

### 2. Pattern Matching

```python
@dataclass
class Pattern:
    """Pattern for matching subgraphs."""
    nodes: List[Node]  # Pattern nodes with wildcard inputs
    root: str  # Root node ID
    
    def match(self, graph: Graph, start: str) -> Optional[dict]:
        """
        Try to match pattern starting at given node.
        
        Returns:
            Mapping from pattern node IDs to graph node IDs,
            or None if no match.
        """
        ...

# Example: log(exp(x)) pattern
LOG_EXP_PATTERN = Pattern(
    nodes=[
        Node("log", OpType.LOG, ["exp_node"]),
        Node("exp_node", OpType.EXP, ["x"]),
    ],
    root="log"
)

# Example: naive logsumexp pattern
NAIVE_LOGSUMEXP = Pattern(
    nodes=[
        Node("log", OpType.LOG, ["sum"]),
        Node("sum", OpType.SUM, ["exp"]),
        Node("exp", OpType.EXP, ["x"]),
    ],
    root="log"
)
```

### 3. Rewrite Rules

```python
@dataclass
class RewriteRule:
    name: str
    pattern: Pattern  # What to match
    replacement: Graph  # What to replace with
    condition: Optional[Callable] = None  # Additional condition
    
    def apply(self, graph: Graph) -> Optional[Graph]:
        """Apply rule once if pattern matches anywhere."""
        for node_id in graph.nodes:
            match = self.pattern.match(graph, node_id)
            if match and (self.condition is None or self.condition(graph, match)):
                return graph.replace(set(match.values()), self.replacement)
        return None

# Rule: log(exp(x)) -> x
LOG_EXP_CANCEL = RewriteRule(
    name="log_exp_cancel",
    pattern=LOG_EXP_PATTERN,
    replacement=Graph(nodes={}, inputs=["x"], outputs=["x"])  # Identity
)

# Rule: log(sum(exp(x))) -> logsumexp(x)
LOGSUMEXP_RULE = RewriteRule(
    name="logsumexp",
    pattern=NAIVE_LOGSUMEXP,
    replacement=Graph(
        nodes={"lse": Node("lse", OpType.LOGSUMEXP, ["x"])},
        inputs=["x"],
        outputs=["lse"]
    )
)
```

### 4. Curvature Computation

```python
def compute_curvature(node: Node, input_stats: dict) -> float:
    """
    Compute curvature bound for a node given input statistics.
    
    Args:
        node: The computation node
        input_stats: Dict with 'range', 'mean', 'std' for each input
    
    Returns:
        Curvature bound κ^curv
    """
    if node.op == OpType.EXP:
        x_max = input_stats[node.inputs[0]]['range'][1]
        return np.exp(2 * x_max)  # κ_exp = e^{2x_max}
    
    elif node.op == OpType.LOG:
        x_min = input_stats[node.inputs[0]]['range'][0]
        return 1.0 / (x_min ** 2)  # κ_log = 1/x_min^2
    
    elif node.op == OpType.DIV:
        denom_min = abs(input_stats[node.inputs[1]]['range'][0])
        return 1.0 / (denom_min ** 3)  # κ_div = 1/|d|^3
    
    elif node.op == OpType.MATMUL:
        # Need condition numbers of inputs
        A_cond = input_stats[node.inputs[0]].get('condition', 1.0)
        B_cond = input_stats[node.inputs[1]].get('condition', 1.0)
        return A_cond * B_cond
    
    elif node.op in [OpType.ADD, OpType.MUL, OpType.RELU]:
        return 0.0  # Linear operations have zero curvature
    
    else:
        return 1.0  # Default

def total_curvature(graph: Graph, input_stats: dict) -> float:
    """Compute total curvature of a graph."""
    # Forward pass to compute intermediate statistics
    stats = propagate_stats(graph, input_stats)
    
    return sum(compute_curvature(node, stats) for node in graph.nodes.values())
```

### 5. Rewrite Search

```python
def stability_rewrite(graph: Graph, 
                      input_stats: dict,
                      rules: List[RewriteRule],
                      max_iterations: int = 100) -> Graph:
    """
    Search for equivalent graph with minimum curvature.
    
    Uses beam search over rewrite sequences.
    """
    initial_curvature = total_curvature(graph, input_stats)
    
    # Beam search
    beam = [(graph, initial_curvature)]
    best = (graph, initial_curvature)
    
    for iteration in range(max_iterations):
        candidates = []
        
        for current_graph, current_curv in beam:
            # Try each rule
            for rule in rules:
                new_graph = rule.apply(current_graph)
                if new_graph is not None:
                    new_curv = total_curvature(new_graph, input_stats)
                    candidates.append((new_graph, new_curv))
        
        if not candidates:
            break
        
        # Keep best candidates
        candidates.sort(key=lambda x: x[1])
        beam = candidates[:10]  # Beam width = 10
        
        if beam[0][1] < best[1]:
            best = beam[0]
    
    return best[0]
```

---

## Implementation Plan

### Phase 1: Graph IR and Parsing (Week 1-2)

**Deliverables:**
- `Graph`, `Node` classes
- Parser from PyTorch FX to our IR
- Graph manipulation utilities (subgraph, replace)

**Validation:**
- Round-trip: FX → IR → FX preserves semantics
- Visualize graphs with graphviz

### Phase 2: Pattern Matching (Week 3-4)

**Deliverables:**
- `Pattern` class with matching algorithm
- Library of 10-15 common patterns
- Pattern visualization

**Validation:**
- Match patterns in real models
- Performance benchmarks

### Phase 3: Rewrite Rules (Week 5-6)

**Deliverables:**
- `RewriteRule` class
- Library of 20+ stability-preserving rules
- Curvature computation for all ops

**Validation:**
- Verify rewrites preserve semantics
- Measure curvature reduction

### Phase 4: Search and Optimization (Week 7-8)

**Deliverables:**
- Beam search over rewrites
- Integration with PyTorch
- End-to-end pipeline

**Validation:**
- Benchmark on real models
- Measure numerical error reduction

---

## Rule Library

### Category 1: Cancellation Rules

```python
# log(exp(x)) → x
LOG_EXP_CANCEL = RewriteRule(...)

# exp(log(x)) → x  (for x > 0)
EXP_LOG_CANCEL = RewriteRule(...)

# sqrt(x^2) → |x|
SQRT_SQUARE = RewriteRule(...)

# x / x → 1  (for x ≠ 0)
DIV_SELF = RewriteRule(...)
```

### Category 2: Fusion Rules

```python
# log(sum(exp(x))) → logsumexp(x)
LOGSUMEXP_FUSION = RewriteRule(...)

# exp(x) / sum(exp(x)) → softmax(x)
SOFTMAX_FUSION = RewriteRule(...)

# (x - mean(x)) / std(x) → layer_norm(x)
LAYERNORM_FUSION = RewriteRule(...)

# x @ A @ B → x @ (A @ B)  (precompute when A, B constant)
MATMUL_CHAIN_FUSION = RewriteRule(...)
```

### Category 3: Stabilization Rules

```python
# a - b where a ≈ b → use compensated subtraction
COMPENSATED_SUB = RewriteRule(...)

# x / (1 + exp(-x)) → sigmoid(x)  (numerically stable sigmoid)
STABLE_SIGMOID = RewriteRule(...)

# log(1 + x) → log1p(x)  (for small x)
LOG1P_RULE = RewriteRule(...)

# exp(x) - 1 → expm1(x)  (for small x)
EXPM1_RULE = RewriteRule(...)
```

### Category 4: Reassociation Rules

```python
# (a + b) + c → a + (b + c)  (when stability improves)
REASSOCIATE_ADD = RewriteRule(
    name="reassociate_add",
    pattern=...,
    replacement=...,
    condition=lambda g, m: curvature_improves(g, m)
)

# (a * b) * c → a * (b * c)  (when stability improves)
REASSOCIATE_MUL = RewriteRule(...)
```

---

## Example: Optimizing Softmax

### Input Graph (Naive)
```
exp_logits = exp(logits)
sum_exp = sum(exp_logits, axis=-1)
probs = exp_logits / sum_exp
```

Curvature: $\kappa^{\mathrm{curv}} \approx e^{2 \|logits\|_\infty}$ — **exponentially bad!**

### After Rewriting
```
max_logit = max(logits, axis=-1)
shifted = logits - max_logit
exp_shifted = exp(shifted)
sum_exp = sum(exp_shifted, axis=-1)
probs = exp_shifted / sum_exp
```

Curvature: $\kappa^{\mathrm{curv}} \approx O(1)$ — **stable!**

### The Rewrite Sequence

1. **Introduce shift:** $\exp(x_i) / \sum_j \exp(x_j) = \exp(x_i - c) / \sum_j \exp(x_j - c)$ for any $c$
2. **Choose $c = \max_j x_j$:** Makes all exponents ≤ 0
3. **Curvature drops:** From $e^{2x_{\max}}$ to $O(1)$

Our rewriter discovers this automatically by trying the SHIFT_INVARIANCE rule and measuring curvature.

---

## Example: Optimizing Log-Sum-Exp

### Input Graph (Naive)
```
exp_x = exp(x)
sum_exp = sum(exp_x)
result = log(sum_exp)
```

Curvature: $\kappa \approx e^{2\|x\|_\infty}$

### After Rewriting
```
max_x = max(x)
shifted = x - max_x  
exp_shifted = exp(shifted)
sum_exp = sum(exp_shifted)
log_sum = log(sum_exp)
result = max_x + log_sum
```

Curvature: $\kappa \approx O(1)$

---

## Validation Strategy

### Experiment 1: Known Stability Patterns

**Setup:**
1. Collect 10 known numerically unstable patterns (naive softmax, naive variance, etc.)
2. Run rewriter on each
3. Verify it finds the stable version

**Success Metric:** Correctly fixes ≥8/10 patterns

### Experiment 2: Real Model Improvement

**Setup:**
1. Take PyTorch implementations of attention, layer norm, batch norm
2. Run rewriter
3. Measure numerical error on random inputs

**Success Metric:** Reduce maximum relative error by 10-100x

### Experiment 3: Curvature Correlation

**Setup:**
1. Generate random computation graphs
2. Compute curvature before/after rewriting
3. Measure actual numerical error before/after

**Success Metric:** Curvature reduction correlates with error reduction (r > 0.8)

---

## API Design

```python
from hnf.rewriter import GraphRewriter, load_rules

# Load standard rules
rules = load_rules('stability')  # or 'fusion', 'all'

# Create rewriter
rewriter = GraphRewriter(rules)

# Rewrite a PyTorch module
optimized_model = rewriter.optimize(model, sample_input)

# See what changed
diff = rewriter.diff(model, optimized_model)
print(diff)
# [Rewrite: naive_softmax → stable_softmax at layer attention.softmax]
# [Rewrite: log(sum(exp(x))) → logsumexp at layer output_proj]

# Measure improvement
original_curv = rewriter.curvature(model, sample_input)
optimized_curv = rewriter.curvature(optimized_model, sample_input)
print(f"Curvature: {original_curv:.2e} → {optimized_curv:.2e}")
```

---

## Advanced Features

### 1. Equality Saturation

Instead of greedy rewriting, use equality saturation (e-graphs) to explore all equivalent programs:

```python
from hnf.egraph import EGraph

def equality_saturation_rewrite(graph, rules, input_stats):
    """Find optimal equivalent program using e-graphs."""
    egraph = EGraph()
    egraph.add(graph)
    
    # Saturate with all rules
    for _ in range(100):
        for rule in rules:
            egraph.apply(rule)
        if egraph.saturated():
            break
    
    # Extract minimum-curvature program
    return egraph.extract(lambda g: total_curvature(g, input_stats))
```

### 2. Domain-Specific Rules

Add rules for specific domains:

```python
# Image processing
rules_image = [
    # Separable convolutions
    CONV2D_SEPARATE,
    # Winograd for small kernels
    CONV_WINOGRAD,
]

# NLP
rules_nlp = [
    # Flash attention pattern
    ATTENTION_FLASH,
    # Fused layer norm
    LAYERNORM_FUSED,
]
```

### 3. Hardware-Aware Rewriting

Consider hardware constraints:

```python
def hardware_aware_rewrite(graph, rules, input_stats, hardware):
    """Rewrite considering hardware capabilities."""
    # Filter rules by hardware support
    supported_rules = [r for r in rules if hardware.supports(r.replacement)]
    
    # Also consider hardware-specific cost
    def cost(g):
        curv = total_curvature(g, input_stats)
        flops = hardware.estimate_flops(g)
        return curv + 0.001 * flops  # Trade off stability vs speed
    
    return search_with_cost(graph, supported_rules, cost)
```

---

## Compute Requirements

| Task | Time | Hardware |
|------|------|----------|
| Parse model to IR | <1 sec | Laptop |
| Pattern matching | <1 sec per pattern | Laptop |
| Curvature computation | <1 sec per graph | Laptop |
| Beam search (100 iterations) | <1 min | Laptop |
| Equality saturation | <10 min | Laptop |

All development feasible on Mac laptop.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Rules don't cover real patterns | Medium | Extensible rule library, learn rules from data |
| Search space too large | Medium | Beam search, equality saturation, heuristics |
| Curvature doesn't predict error | Low | Validated by Project 1 first |
| PyTorch internals change | Low | Abstract IR layer |

---

## Expected Impact

### For Practitioners

- Automatic discovery of numerical stabilization tricks
- Less manual effort tuning implementations
- Fewer NaN/Inf issues in training

### For Compilers

- New optimization pass for numerical stability
- Principled approach to floating-point optimization
- Integration with XLA, TorchScript, etc.

### For Research

- Framework for studying numerical program transformations
- Connection between rewriting and stability theory
- Foundation for verified numerical compilation

---

## Next Steps

1. Implement `Graph` IR with FX parsing
2. Build pattern matching for 5 simple patterns
3. Implement curvature computation for core ops
4. Create 10 stability-preserving rules
5. Build beam search and test on softmax/logsumexp
