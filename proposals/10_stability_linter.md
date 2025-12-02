# Project 10: Numerical Stability Linter for Transformer Code

## Transformer Application: Catch Numerical Bugs in Attention and Training Code Before They Crash

**Use case:** Static analysis tool that parses your transformer implementation and flags numerical issues before you run anything. Like ESLint but for numerical stability—finds unstable attention implementations, missing epsilon in LayerNorm, and cross-entropy overflow risks.

### The Problem: Numerical Bugs Are Hard to Find

Transformer numerical bugs are insidious:
- **Compile fine, crash at runtime** (NaN in attention at step 50,000)
- **Silent quality degradation** (loss converges but 5% worse than it should be)
- **Input-dependent** (works on short sequences, fails on long ones)
- **Precision-dependent** (works in FP32, fails in FP16)

These bugs waste days of debugging and compute.

### Current Debugging Workflow

```python
# 1. Write transformer code
# 2. Start training
# 3. Wait 10 hours
# 4. See NaN in loss
# 5. Add print statements everywhere
# 6. Repeat
```

### This Tool Catches Issues Before Training

```python
from stability_linter import TransformerLinter

# Analyze your transformer code
linter = TransformerLinter()
issues = linter.analyze_file("my_transformer.py")

# Output:
# my_transformer.py:45:12 [ERROR] Unstable softmax implementation
#   Found: exp(scores) / exp(scores).sum(-1)
#   Issue: Overflows when max(scores) > 88 (float32) or > 11 (float16)
#   Fix: Use torch.softmax(scores, dim=-1) or subtract max before exp
#
# my_transformer.py:67:8 [WARNING] LayerNorm without epsilon
#   Found: (x - x.mean()) / x.std()
#   Issue: Division by zero when input is constant
#   Fix: Use (x - x.mean()) / (x.std() + 1e-5) or torch.nn.LayerNorm
#
# my_transformer.py:89:4 [WARNING] Cross-entropy precision loss
#   Found: -log(softmax(logits)[target])
#   Issue: log of small probabilities loses precision in float16
#   Fix: Use F.cross_entropy(logits, target) which fuses log-softmax
#
# my_transformer.py:112:16 [INFO] High curvature operation
#   Found: attention_weights ** temperature (temperature < 1)
#   Issue: Curvature κ ≈ 1/temperature², high precision needed
#   Recommendation: Ensure FP32 or add gradient scaling
```

---

## Theoretical Foundation

### Static Numerical Analysis

Many numerical bugs are **structural**—they arise from the pattern of operations, not specific weight values. These can be detected without running the model.

**Examples:**
- `exp(x)` where x might exceed 88 (float32 overflow)
- `x / y` where y might be zero (LayerNorm without epsilon)  
- `log(softmax(x))` computed separately instead of fused
- Attention scores without max-subtraction trick

### Curvature as a Lint Signal

High curvature operations are **precision hazards**. We compute curvature bounds statically:
$$\kappa_{\text{exp}} = e^{2x}, \quad \kappa_{\text{log}} = 1/x^2, \quad \kappa_{\text{softmax}} = e^{2 \cdot \text{range}(x)}$$

If we can bound input ranges, we can flag high-curvature patterns.

### Transformer-Specific Pattern Library

| Pattern | Issue | Curvature | Fix |
|---------|-------|-----------|-----|
| Naive softmax | Overflow | $e^{2 \max}$ | Use `torch.softmax` |
| Separate log-softmax | Precision loss | $e^{\max}$ | Fuse with `F.log_softmax` |
| LayerNorm without eps | Div by zero | $1/\sigma^2$ | Add epsilon |
| Attention without scaling | Explosion | $d_k \cdot e^{2QK}$ | Divide by $\sqrt{d_k}$ |
| Temperature < 1 | Amplification | $1/T^2$ | Document precision needs |
| Large embedding dim | Variance issues | $\sqrt{d}$ | Use scaled initialization |

---

## Technical Approach

### 1. Graph Parsing

```python
class ComputationGraph:
    """Parsed computation graph for analysis."""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Tuple[str, str]] = []
    
    @classmethod
    def from_pytorch_fx(cls, model: nn.Module, 
                        sample_input: torch.Tensor) -> 'ComputationGraph':
        """Parse PyTorch model via FX tracing."""
        traced = torch.fx.symbolic_trace(model)
        
        graph = cls()
        for node in traced.graph.nodes:
            graph.add_node(Node(
                id=node.name,
                op=node.op,
                target=node.target,
                args=[str(a) for a in node.args],
                kwargs=dict(node.kwargs)
            ))
            
            for arg in node.args:
                if hasattr(arg, 'name'):
                    graph.add_edge(arg.name, node.name)
        
        return graph
    
    @classmethod
    def from_jax_jaxpr(cls, fn, sample_input) -> 'ComputationGraph':
        """Parse JAX function via jaxpr."""
        import jax
        
        jaxpr = jax.make_jaxpr(fn)(sample_input)
        
        graph = cls()
        for eqn in jaxpr.jaxpr.eqns:
            graph.add_node(Node(
                id=str(eqn.outvars[0]),
                op='call_primitive',
                target=str(eqn.primitive),
                args=[str(v) for v in eqn.invars]
            ))
        
        return graph

@dataclass
class Node:
    id: str
    op: str
    target: Any
    args: List[str]
    kwargs: dict = None
    
    def get_op_name(self) -> str:
        """Get human-readable operation name."""
        if self.op == 'call_function':
            return self.target.__name__
        elif self.op == 'call_method':
            return self.target
        elif self.op == 'call_module':
            return self.target
        return self.op
```

### 2. Pattern Matching

```python
@dataclass
class LintPattern:
    """Pattern to match in computation graph."""
    
    name: str
    description: str
    severity: str  # 'error', 'warning', 'info'
    ops: List[str]  # Sequence of operations to match
    condition: Optional[Callable] = None
    suggestion: str = ""
    
    def matches(self, graph: ComputationGraph, start_node: str) -> Optional['Match']:
        """Check if pattern matches starting at node."""
        current = start_node
        matched_nodes = []
        
        for expected_op in self.ops:
            node = graph.nodes.get(current)
            if node is None:
                return None
            
            if not self._op_matches(node, expected_op):
                return None
            
            matched_nodes.append(current)
            
            # Move to next node (first output edge)
            outputs = [e[1] for e in graph.edges if e[0] == current]
            current = outputs[0] if outputs else None
        
        if self.condition and not self.condition(graph, matched_nodes):
            return None
        
        return Match(pattern=self, nodes=matched_nodes)
    
    def _op_matches(self, node: Node, expected: str) -> bool:
        """Check if node operation matches expected."""
        op_name = node.get_op_name()
        return op_name == expected or expected == '*'

# Define common anti-patterns
NAIVE_SOFTMAX = LintPattern(
    name='naive-softmax',
    description='Softmax without max-subtraction trick',
    severity='warning',
    ops=['exp', 'sum', 'div'],
    suggestion='Use torch.softmax() or subtract max before exp'
)

NAIVE_LOGSOFTMAX = LintPattern(
    name='naive-logsoftmax',
    description='log(softmax(x)) is numerically unstable',
    severity='error',
    ops=['softmax', 'log'],
    suggestion='Use F.log_softmax() instead'
)

UNPROTECTED_DIV = LintPattern(
    name='unprotected-division',
    description='Division without epsilon protection',
    severity='warning',
    ops=['div'],
    condition=lambda g, nodes: not has_epsilon_protection(g, nodes),
    suggestion='Add small epsilon to denominator: x / (y + 1e-8)'
)

UNPROTECTED_LOG = LintPattern(
    name='unprotected-log',
    description='Logarithm without clamping',
    severity='warning',
    ops=['log'],
    condition=lambda g, nodes: not has_clamp_protection(g, nodes),
    suggestion='Clamp input: log(x.clamp(min=1e-8))'
)

DOUBLE_EXP = LintPattern(
    name='double-exponential',
    description='exp(exp(x)) has extremely high curvature',
    severity='error',
    ops=['exp', 'exp'],
    suggestion='Reconsider computation; this is numerically unstable for x > 3'
)
```

### 3. Curvature Analysis

```python
class CurvatureLinter:
    """Lint based on curvature bounds."""
    
    def __init__(self, curvature_threshold: float = 1e6):
        self.threshold = curvature_threshold
    
    def analyze(self, graph: ComputationGraph, 
                input_range: Tuple[float, float]) -> List['LintResult']:
        """Find high-curvature operations."""
        results = []
        
        # Propagate ranges through graph
        ranges = self._propagate_ranges(graph, input_range)
        
        for node_id, node in graph.nodes.items():
            if node_id in ranges:
                r = ranges[node_id]
                κ = self._estimate_curvature(node, r)
                
                if κ > self.threshold:
                    results.append(LintResult(
                        severity='warning',
                        node=node_id,
                        message=f'High curvature ({κ:.2e}) may cause precision issues',
                        suggestion=self._suggest_fix(node, κ)
                    ))
        
        return results
    
    def _estimate_curvature(self, node: Node, 
                            input_range: Tuple[float, float]) -> float:
        """Estimate curvature of operation given input range."""
        op = node.get_op_name()
        lo, hi = input_range
        
        curvature_estimates = {
            'exp': lambda: np.exp(2 * hi),
            'log': lambda: 1 / (max(lo, 1e-10) ** 2),
            'div': lambda: 1 / (max(abs(lo), 1e-10) ** 3),
            'softmax': lambda: np.exp(2 * (hi - lo)),
            'sqrt': lambda: 1 / (4 * max(lo, 1e-10) ** 1.5),
            'tanh': lambda: 1.0,
            'sigmoid': lambda: 0.25,
            'relu': lambda: 0.0,
            'add': lambda: 0.0,
            'mul': lambda: 0.0,
        }
        
        return curvature_estimates.get(op, lambda: 1.0)()
    
    def _propagate_ranges(self, graph, input_range):
        """Forward propagation of value ranges."""
        # Topological sort
        order = graph.topological_sort()
        
        ranges = {}
        for node_id in order:
            node = graph.nodes[node_id]
            
            if node.op == 'placeholder':
                ranges[node_id] = input_range
            else:
                # Combine input ranges
                input_ranges = [ranges.get(arg, input_range) for arg in node.args]
                ranges[node_id] = self._apply_op_range(node, input_ranges)
        
        return ranges
    
    def _apply_op_range(self, node, input_ranges):
        """Compute output range from input ranges."""
        # Simplified interval arithmetic
        op = node.get_op_name()
        
        if not input_ranges:
            return (-1e10, 1e10)
        
        lo, hi = input_ranges[0]
        
        range_rules = {
            'exp': (np.exp(lo), np.exp(hi)),
            'log': (np.log(max(lo, 1e-10)), np.log(max(hi, 1e-10))),
            'relu': (max(0, lo), max(0, hi)),
            'sigmoid': (0.0, 1.0),
            'tanh': (-1.0, 1.0),
            'softmax': (0.0, 1.0),
        }
        
        return range_rules.get(op, (lo, hi))
```

### 4. Linting Engine

```python
class NumericalLinter:
    """Main linting engine."""
    
    def __init__(self):
        self.pattern_library = [
            NAIVE_SOFTMAX,
            NAIVE_LOGSOFTMAX,
            UNPROTECTED_DIV,
            UNPROTECTED_LOG,
            DOUBLE_EXP,
        ]
        self.curvature_linter = CurvatureLinter()
    
    def lint(self, model: nn.Module, 
             sample_input: torch.Tensor,
             input_range: Tuple[float, float] = (-10, 10)) -> 'LintReport':
        """
        Lint a PyTorch model for numerical issues.
        
        Args:
            model: The model to analyze
            sample_input: Sample input for tracing
            input_range: Expected range of input values
        
        Returns:
            LintReport with all findings
        """
        # Parse model to graph
        graph = ComputationGraph.from_pytorch_fx(model, sample_input)
        
        results = []
        
        # Pattern matching
        for pattern in self.pattern_library:
            for node_id in graph.nodes:
                match = pattern.matches(graph, node_id)
                if match:
                    results.append(LintResult(
                        severity=pattern.severity,
                        node=node_id,
                        pattern=pattern.name,
                        message=pattern.description,
                        suggestion=pattern.suggestion
                    ))
        
        # Curvature analysis
        results.extend(self.curvature_linter.analyze(graph, input_range))
        
        return LintReport(results=results, graph=graph)
    
    def lint_file(self, filepath: str) -> 'LintReport':
        """Lint a Python file containing model definition."""
        # Parse file and extract model
        # This is more complex - would need AST parsing
        raise NotImplementedError("File linting requires AST analysis")

@dataclass
class LintResult:
    severity: str
    node: str
    message: str
    suggestion: str
    pattern: Optional[str] = None
    line_number: Optional[int] = None

@dataclass  
class LintReport:
    results: List[LintResult]
    graph: ComputationGraph
    
    def __str__(self):
        lines = []
        for r in sorted(self.results, key=lambda x: x.severity):
            icon = {'error': '❌', 'warning': '⚠️', 'info': 'ℹ️'}[r.severity]
            lines.append(f"{icon} [{r.severity.upper()}] at {r.node}")
            lines.append(f"   {r.message}")
            if r.suggestion:
                lines.append(f"   Suggestion: {r.suggestion}")
            lines.append("")
        return '\n'.join(lines)
    
    @property
    def n_errors(self):
        return sum(1 for r in self.results if r.severity == 'error')
    
    @property
    def n_warnings(self):
        return sum(1 for r in self.results if r.severity == 'warning')
    
    def to_json(self):
        return json.dumps([asdict(r) for r in self.results])
```

---

## Implementation Plan

### Phase 1: Graph Parsing (Week 1-2)

**Deliverables:**
- `ComputationGraph` from PyTorch FX
- Basic node/edge representation
- Topological sort

**Validation:**
- Parse common models (ResNet, Transformer)
- Verify graph structure is correct

### Phase 2: Pattern Library (Week 3-4)

**Deliverables:**
- `LintPattern` matching infrastructure
- 10-15 common anti-patterns
- Pattern tests

**Validation:**
- Create test cases for each pattern
- Verify no false positives on clean code

### Phase 3: Curvature Analysis (Week 5-6)

**Deliverables:**
- Range propagation
- Curvature estimation per operation
- High-curvature warnings

**Validation:**
- Compare curvature estimates to actual
- Tune threshold for useful warnings

### Phase 4: Integration (Week 7-8)

**Deliverables:**
- CLI tool
- VS Code extension (optional)
- Documentation

**Validation:**
- Run on Hugging Face top-100 models
- Collect feedback on usefulness

---

## Pattern Library

### Category 1: Stability Patterns

```python
patterns = [
    LintPattern(
        name='naive-softmax',
        description='Softmax without numerical stabilization',
        severity='warning',
        ops=['exp', '*', 'div'],
        suggestion='Use F.softmax() or subtract max first'
    ),
    
    LintPattern(
        name='naive-cross-entropy',
        description='log(softmax(x)) chain is unstable',
        severity='error',
        ops=['softmax', 'log'],
        suggestion='Use F.log_softmax() or F.cross_entropy()'
    ),
    
    LintPattern(
        name='naive-log1p',
        description='log(1 + x) loses precision for small x',
        severity='info',
        ops=['add', 'log'],
        condition=lambda g, n: is_adding_one(g, n),
        suggestion='Use torch.log1p(x) for small x'
    ),
    
    LintPattern(
        name='naive-expm1',
        description='exp(x) - 1 loses precision for small x',
        severity='info',
        ops=['exp', 'sub'],
        condition=lambda g, n: is_subtracting_one(g, n),
        suggestion='Use torch.expm1(x) for small x'
    ),
]
```

### Category 2: Protection Patterns

```python
patterns += [
    LintPattern(
        name='unprotected-division',
        description='Division without epsilon in denominator',
        severity='warning',
        ops=['div'],
        condition=lambda g, n: not has_epsilon(g, n[0]),
        suggestion='Add epsilon: x / (y + 1e-8)'
    ),
    
    LintPattern(
        name='unprotected-log',
        description='Log of potentially non-positive value',
        severity='warning',
        ops=['log'],
        condition=lambda g, n: not has_clamp(g, n[0]),
        suggestion='Use torch.log(x.clamp(min=1e-8))'
    ),
    
    LintPattern(
        name='unprotected-sqrt',
        description='Sqrt of potentially negative value',
        severity='warning',
        ops=['sqrt'],
        condition=lambda g, n: not has_relu_or_clamp(g, n[0]),
        suggestion='Use torch.sqrt(x.clamp(min=0))'
    ),
]
```

### Category 3: Overflow Patterns

```python
patterns += [
    LintPattern(
        name='exp-overflow',
        description='Exponential of potentially large value',
        severity='warning',
        ops=['exp'],
        condition=lambda g, n: max_input_estimate(g, n[0]) > 80,
        suggestion='Clamp input: exp(x.clamp(max=80))'
    ),
    
    LintPattern(
        name='double-exp',
        description='exp(exp(x)) overflows for x > ~4',
        severity='error',
        ops=['exp', 'exp'],
        suggestion='Reconsider computation structure'
    ),
    
    LintPattern(
        name='large-power',
        description='Large exponent may overflow',
        severity='warning',
        ops=['pow'],
        condition=lambda g, n: exponent_value(g, n) > 10,
        suggestion='Consider log-space computation'
    ),
]
```

### Category 4: Cancellation Patterns

```python
patterns += [
    LintPattern(
        name='catastrophic-cancellation',
        description='Subtraction of similar values loses precision',
        severity='info',
        ops=['sub'],
        condition=lambda g, n: operands_similar_magnitude(g, n),
        suggestion='Reformulate to avoid subtraction'
    ),
    
    LintPattern(
        name='variance-cancellation',
        description='Variance computed as E[X²] - E[X]² loses precision',
        severity='warning',
        ops=['mean', 'pow', 'sub'],  # Simplified
        suggestion='Use Welford\'s algorithm or torch.var()'
    ),
]
```

---

## Validation Strategy

### Experiment 1: Known Bug Detection

**Setup:**
1. Collect known numerical bugs from GitHub issues
2. Create test cases from bug reports
3. Run linter and measure detection

**Success Metric:** Detect 80%+ of known bugs

### Experiment 2: False Positive Rate

**Setup:**
1. Run on clean, well-written code (PyTorch internals, Hugging Face)
2. Count false positives
3. Tune thresholds

**Success Metric:** <20% false positive rate

### Experiment 3: User Study

**Setup:**
1. Give linter to 10 practitioners
2. Collect feedback on usefulness
3. Measure: bugs found, false positives, time saved

**Success Metric:** Positive net value reported by majority

---

## API Design

### CLI

```bash
# Lint a model file
$ hnf-lint model.py
❌ [ERROR] at layer_12.attention.softmax
   log(softmax(x)) chain is unstable
   Suggestion: Use F.log_softmax() or F.cross_entropy()

⚠️ [WARNING] at layer_5.norm.div
   Division without epsilon in denominator
   Suggestion: Add epsilon: x / (y + 1e-8)

Found 1 error, 1 warning.

# Lint with options
$ hnf-lint model.py --severity warning --format json
$ hnf-lint model.py --ignore naive-softmax
$ hnf-lint model.py --input-range -100 100
```

### Python API

```python
from hnf.lint import NumericalLinter

# Basic usage
linter = NumericalLinter()
report = linter.lint(model, sample_input)

print(report)
print(f"Errors: {report.n_errors}, Warnings: {report.n_warnings}")

# With options
linter = NumericalLinter(
    patterns=['naive-softmax', 'unprotected-division'],  # Specific patterns
    curvature_threshold=1e8,  # Adjust sensitivity
    severity_filter='error'  # Only errors
)

# Check in CI
if report.n_errors > 0:
    sys.exit(1)
```

### VS Code Integration (Optional)

```json
// .vscode/settings.json
{
    "hnf.lint.enable": true,
    "hnf.lint.severity": "warning",
    "hnf.lint.inputRange": [-10, 10]
}
```

```python
# Language server protocol integration
class HNFLanguageServer:
    def on_did_change(self, document):
        model = extract_model(document)
        if model:
            report = self.linter.lint(model)
            diagnostics = self.to_diagnostics(report)
            self.publish_diagnostics(document.uri, diagnostics)
```

---

## Advanced Features

### 1. Auto-Fix

Generate fixed code for simple patterns:

```python
class AutoFixer:
    fixes = {
        'naive-softmax': 'F.softmax({input}, dim={dim})',
        'unprotected-log': 'torch.log({input}.clamp(min=1e-8))',
        'unprotected-div': '{numer} / ({denom} + 1e-8)',
    }
    
    def fix(self, result: LintResult, graph: ComputationGraph) -> str:
        """Generate fixed code."""
        template = self.fixes.get(result.pattern)
        if template:
            args = self.extract_args(graph, result.node)
            return template.format(**args)
        return None
```

### 2. Custom Patterns

Allow users to define their own patterns:

```python
# custom_patterns.yaml
patterns:
  - name: my-pattern
    description: Custom pattern description
    severity: warning
    ops: [matmul, relu, matmul]
    suggestion: Consider using a different approach

# Usage
linter = NumericalLinter()
linter.load_patterns('custom_patterns.yaml')
```

### 3. Model Comparison

Compare two implementations:

```python
def compare_stability(model1, model2, sample_input):
    """Compare numerical stability of two implementations."""
    report1 = linter.lint(model1, sample_input)
    report2 = linter.lint(model2, sample_input)
    
    return {
        'model1_issues': len(report1.results),
        'model2_issues': len(report2.results),
        'recommendation': 'model1' if len(report1.results) < len(report2.results) else 'model2'
    }
```

---

## Compute Requirements

| Task | Time | Hardware |
|------|------|----------|
| FX tracing | <1 sec | CPU |
| Pattern matching | <1 sec | CPU |
| Curvature analysis | <1 sec | CPU |
| Full lint | <2 sec | CPU |

Entirely symbolic—runs instantly on any hardware.

---

## Risks and Mitigations

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| Too many false positives | Medium | Tune thresholds, allow ignores |
| Misses real issues | Medium | Expand pattern library over time |
| FX tracing fails | Low | Fall back to AST analysis |
| Users don't adopt | Medium | Integrate with popular tools |

---

## Expected Impact

### For Practitioners

- Catch numerical bugs before training
- Learn numerical stability best practices
- Faster debugging of NaN/Inf issues

### For Teams

- Enforce numerical stability standards
- CI integration to prevent regressions
- Documentation of numerical assumptions

### For the Community

- Shared pattern library grows over time
- Education about numerical issues
- Reduced time wasted on stability bugs

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Bug detection rate | >80% of known bugs |
| False positive rate | <20% |
| User satisfaction | >70% positive |
| Adoption | 100+ GitHub stars in 6 months |

---

## Next Steps

1. Implement FX graph parsing
2. Build pattern matching infrastructure
3. Create initial pattern library (10 patterns)
4. Add curvature-based linting
5. Create CLI and test on real models
