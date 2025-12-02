# Project 2: Mixed-Precision Optimizer via Sheaf Cohomology

## Transformer Application: Optimal Mixed-Precision for Transformer Training

**The Problem:** Large language model training uses mixed precision (float16/bfloat16 + float32) to reduce memory and speed up training. Current approaches (AMP) use heuristics that either waste memory (too much float32) or cause instabilities (too little float32).

**Our Solution:** Use sheaf cohomology to find the mathematically optimal precision assignment for each transformer component. When no global assignment exists (H^0 = ∅), the obstruction cocycle tells us exactly which layers force mixed precision.

**Concrete Use Cases:**
1. **Pre-training LLMs:** "For GPT-3 scale, attention softmax needs float32, but 78% of FFN params can use bfloat16—saves 15GB VRAM"
2. **Memory-constrained fine-tuning:** "Given 24GB VRAM budget, here's the precision assignment that fits your model"
3. **Debugging AMP failures:** "AMP failed because layers 23-27 have cohomological obstruction—they can't all use the same precision"

---

## Executive Summary

Implement the precision sheaf $\mathcal{P}_G^\varepsilon$ for real computation graphs and use cohomological obstructions to generate optimal mixed-precision configurations. When $H^0 \neq \emptyset$, a consistent global precision assignment exists; when $H^1 \neq 0$, the obstruction cocycle tells us exactly where precision must increase.

---

## Theoretical Foundation

### The Precision Sheaf

Given a computation graph $G = (V, E)$ representing a neural network, we define a presheaf of precision assignments.

**Definition.** For an open set $U \subseteq G$ (a subgraph), define:
$$\mathcal{P}_G^\varepsilon(U) = \{p: U \to \mathbb{N} \mid \text{precision } p(v) \text{ achieves } \varepsilon\text{-accuracy on } U\}$$

The restriction maps are natural: if $V \subseteq U$, then $\rho_{U,V}: \mathcal{P}_G^\varepsilon(U) \to \mathcal{P}_G^\varepsilon(V)$ just restricts the domain.

### Global Sections and Obstructions

**Global sections:** $H^0(G, \mathcal{P}_G^\varepsilon) = \mathcal{P}_G^\varepsilon(G)$ is the set of globally consistent precision assignments.

**Key insight:** $H^0 = \emptyset$ means no uniform precision assignment achieves $\varepsilon$-accuracy. Mixed precision is *required*.

**The obstruction:** When $H^0 = \emptyset$, we can compute $H^1(G, \mathcal{P}_G^\varepsilon)$ using Čech cohomology. The obstruction cocycle $[\omega] \in H^1$ lives on edges and measures "precision jumps" needed between adjacent operations.

### From Cohomology to Mixed Precision

The algorithm:
1. Try to find $\sigma \in H^0$ at minimal precision (e.g., float16 everywhere)
2. If $H^0 = \emptyset$, compute the obstruction cocycle $\omega \in C^1$
3. The cocycle $\omega(e)$ for edge $e: v \to w$ tells us the precision gap needed
4. Increase precision at vertices where $\omega$ is nonzero
5. Iterate until $H^0 \neq \emptyset$

---

## Technical Approach

### 1. Graph Representation

Parse PyTorch models into explicit DAGs:

```python
@dataclass
class ComputationNode:
    name: str
    op: Callable
    curvature: float  # κ^curv for this operation
    inputs: List[str]
    outputs: List[str]

@dataclass  
class ComputationGraph:
    nodes: Dict[str, ComputationNode]
    edges: List[Tuple[str, str]]  # (source, target)
    
    @classmethod
    def from_fx(cls, fx_graph: torch.fx.Graph) -> 'ComputationGraph':
        """Convert PyTorch FX graph to our representation."""
        ...
```

### 2. Čech Complex Construction

For a computation graph, build the Čech complex:

```python
def build_cech_complex(graph: ComputationGraph, cover: List[Set[str]]):
    """
    Build Čech complex for the graph with given cover.
    
    Args:
        graph: The computation graph
        cover: Open cover {U_i} of the graph (e.g., one set per node + neighbors)
    
    Returns:
        C0: Dict mapping nodes to precision assignments
        C1: Dict mapping edges to precision compatibility constraints
        d0: Boundary map C0 -> C1
    """
    # C^0: sections over each open set
    C0 = {U: precision_assignments_for(graph.subgraph(U)) for U in cover}
    
    # C^1: sections over intersections
    C1 = {}
    for i, U_i in enumerate(cover):
        for j, U_j in enumerate(cover):
            if i < j:
                intersection = U_i & U_j
                if intersection:
                    C1[(i,j)] = precision_assignments_for(graph.subgraph(intersection))
    
    return C0, C1
```

### 3. Cohomology Computation

Compute $H^0$ and $H^1$ using linear algebra over $\mathbb{Z}$:

```python
def compute_H0(C0, C1, d0):
    """
    Compute H^0 = ker(d^0).
    
    Returns:
        List of global sections (consistent precision assignments),
        or empty list if none exist.
    """
    # A global section is a choice σ_i ∈ C0[U_i] for each U_i
    # such that σ_i|_{U_i ∩ U_j} = σ_j|_{U_i ∩ U_j}
    
    # Build constraint matrix and solve
    ...

def compute_H1(C0, C1, d0, d1):
    """
    Compute H^1 = ker(d^1) / im(d^0).
    
    When H^0 = ∅, this tells us the obstruction to global sections.
    
    Returns:
        Representative cocycles for H^1
    """
    # A 1-cocycle ω assigns precision to each edge
    # such that on triple intersections: ω_ij + ω_jk + ω_ki = 0
    
    # Compute kernel and quotient
    ...
```

### 4. Optimization Loop

```python
def optimize_precision(graph: ComputationGraph, 
                       target_accuracy: float,
                       min_precision: int = 16) -> Dict[str, int]:
    """
    Find minimal mixed-precision assignment achieving target accuracy.
    
    Returns:
        Dict mapping node names to precision (in bits)
    """
    # Start with minimum precision everywhere
    precision = {node: min_precision for node in graph.nodes}
    
    while True:
        # Check if current assignment achieves accuracy
        sheaf = build_precision_sheaf(graph, precision, target_accuracy)
        H0 = compute_H0(sheaf)
        
        if H0:
            # Found valid assignment
            return precision
        
        # Compute obstruction
        H1 = compute_H1(sheaf)
        obstruction = get_representative_cocycle(H1)
        
        # Increase precision where obstruction is nonzero
        for edge, gap in obstruction.items():
            if gap > 0:
                source, target = edge
                # Increase precision at the more sensitive node
                sensitive = max(source, target, 
                               key=lambda n: graph.nodes[n].curvature)
                precision[sensitive] += gap
        
        # Check for convergence
        if max(precision.values()) > 64:
            raise ValueError("No feasible precision assignment found")
    
    return precision
```

---

## Implementation Plan

### Phase 1: Graph Infrastructure (Week 1-2)

**Deliverables:**
- `ComputationGraph` class with FX integration
- Subgraph extraction and cover generation
- Basic precision constraint representation

**Validation:**
- Parse ResNet-18 and visualize as DAG
- Verify subgraph operations are correct

### Phase 2: Čech Complex (Week 3-4)

**Deliverables:**
- Čech complex construction for computation graphs
- Boundary map computation
- Basic cohomology (H^0) computation

**Validation:**
- Verify on toy graphs with known cohomology
- Test that H^0 = ∅ when expected

### Phase 3: H^1 and Obstruction (Week 5-6)

**Deliverables:**
- Full H^1 computation
- Obstruction cocycle extraction
- Interpretation as precision requirements

**Validation:**
- Create graphs where mixed precision is provably necessary
- Verify obstruction correctly identifies the problem

### Phase 4: Optimization and AMP Integration (Week 7-8)

**Deliverables:**
- Full optimization loop
- Export to PyTorch AMP config format
- Comparison benchmarks against baseline AMP

**Validation:**
- Benchmark on standard models
- Measure memory reduction and accuracy preservation

---

## Detailed Algorithm

### Step 1: Build the Cover

For a computation graph $G$, we use the **star cover**: for each node $v$, the open set $U_v$ contains $v$ and all its immediate neighbors.

```python
def star_cover(graph: ComputationGraph) -> List[Set[str]]:
    """Build star cover: each node plus its neighbors."""
    cover = []
    for node in graph.nodes:
        neighbors = graph.get_neighbors(node)
        cover.append({node} | neighbors)
    return cover
```

### Step 2: Local Precision Constraints

For each open set $U_v$ (star of $v$), we need precision $p_v$ satisfying:

$$p_v \geq \log_2\left(\frac{\kappa_v^{\mathrm{curv}} \cdot D_v^2}{\varepsilon}\right)$$

where $D_v$ is the diameter of inputs flowing through $v$.

### Step 3: Compatibility on Intersections

For adjacent nodes $v, w$, the intersection $U_v \cap U_w$ contains both. Compatibility requires:

$$|p_v - p_w| \leq \delta_{vw}$$

where $\delta_{vw}$ is a tolerance based on the edge's precision propagation.

If this cannot be satisfied, we have a **nonzero obstruction** on edge $(v, w)$.

### Step 4: Resolve Obstructions

When obstruction $\omega(v, w) \neq 0$:
- If $\omega(v, w) > 0$: need to increase precision at $v$ or $w$
- Choose the node with higher curvature (more sensitive)
- Increase by $\omega(v, w)$ bits

---

## Example: Attention Layer

Consider a single attention head:

```
Q ──────┐
        ├──→ QK^T ──→ scale ──→ softmax ──→ ×V ──→ output
K ──────┘                                   ↑
                                            │
V ──────────────────────────────────────────┘
```

**Local precision requirements:**
- $\mathrm{QK^T}$: $p_1 \geq \log_2(\kappa(Q) \kappa(K) D^2 / \varepsilon)$
- $\mathrm{softmax}$: $p_2 \geq \log_2(e^{2\|QK^T\|} / \varepsilon)$ — very high!
- $\mathrm{×V}$: $p_3 \geq \log_2(\kappa(V) / \varepsilon)$

**Obstruction analysis:**
- Edge $(\mathrm{scale}, \mathrm{softmax})$: $\omega = p_2 - p_1$ often large
- The softmax node needs much higher precision than its neighbors

**Resolution:**
- Compute softmax in float32, keep Q, K, V in float16
- This is exactly what Flash Attention does, but we derived it systematically!

---

## Validation Strategy

### Experiment 1: Comparison with AMP

**Setup:**
1. Take standard models: ResNet-50, BERT-base, GPT-2
2. Run our optimizer to get precision assignment
3. Compare against PyTorch AMP defaults

**Metrics:**
- Memory usage
- Training stability (no loss spikes)
- Final accuracy

**Success Criterion:** Same accuracy as AMP with 10-20% less memory

### Experiment 2: Pathological Cases

**Setup:**
1. Construct models where mixed precision is provably necessary
2. Verify our algorithm finds valid assignments
3. Verify uniform precision fails

**Example pathological model:**
```python
class PathologicalNet(nn.Module):
    def forward(self, x):
        # Low precision OK
        x = self.linear1(x)
        x = F.relu(x)
        
        # MUST be high precision
        x = torch.exp(torch.exp(x))  # κ ~ e^(e^x)
        
        # Low precision OK again
        x = self.linear2(x)
        return x
```

### Experiment 3: Cohomology Correctness

**Setup:**
1. Create toy graphs with known H^0, H^1
2. Verify our computation matches
3. Test edge cases (disconnected graphs, cycles, etc.)

---

## API Design

```python
from hnf.sheaf import PrecisionOptimizer

# Create optimizer
optimizer = PrecisionOptimizer(target_accuracy=1e-5)

# Analyze model
result = optimizer.analyze(model, sample_input)

# View cohomology
print(f"H^0 dimension: {result.h0_dim}")  # 0 means mixed precision required
print(f"H^1 obstruction: {result.h1_cocycle}")

# Get optimal assignment
precision_map = result.get_precision_assignment()
# {'attention.softmax': 32, 'attention.qk': 16, 'ffn': 16, ...}

# Export for PyTorch
amp_config = result.to_amp_config()

# Visualize obstruction
result.visualize_obstruction()  # Shows graph with precision requirements
```

---

## Theoretical Contributions

This project makes several novel theoretical contributions:

### 1. First Application of Sheaf Cohomology to Mixed Precision

While sheaves have been used in sensor networks and distributed systems, this is the first application to numerical precision analysis.

### 2. Obstruction-Theoretic View of Precision

The insight that H^1 ≠ 0 means "mixed precision is topologically required" is new. It explains *why* certain architectures need mixed precision, not just *that* they do.

### 3. Optimal Precision from Cohomological Minimization

The algorithm to minimize total precision subject to H^0 ≠ ∅ gives provably minimal mixed-precision assignments.

---

## Compute Requirements

| Task | Time | Hardware |
|------|------|----------|
| Graph construction | O(n) | Laptop |
| Čech complex | O(n²) | Laptop |
| Cohomology computation | O(n³) | Laptop (up to ~1000 nodes) |
| Full optimization | O(n³ × iterations) | Laptop |

For very large models (>1000 layers), we can use hierarchical decomposition:
1. Decompose graph into blocks (e.g., transformer layers)
2. Compute cohomology within blocks
3. Stitch together using relative cohomology

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Cohomology too expensive | Medium | Hierarchical decomposition, sparse methods |
| Cover choice affects results | Low | Use canonical star cover, prove invariance |
| Obstruction hard to interpret | Medium | Clear visualization, case studies |
| AMP already good enough | Medium | Focus on edge cases where AMP fails |

---

## Expected Impact

### For Practitioners

- Principled mixed-precision configs instead of trial and error
- Automatic detection of precision-critical subgraphs
- Reduced memory usage with guaranteed accuracy

### For Researchers

- New connection between algebraic topology and numerical computing
- Framework for understanding precision requirements geometrically
- Foundation for future precision analysis tools

---

## Next Steps

1. Implement `ComputationGraph` with FX integration
2. Build Čech complex for simple graphs
3. Implement H^0 computation and verify on toy examples
4. Add H^1 and obstruction extraction
5. Build optimization loop and compare against AMP
