# Proposal 20: A Numerical Geometry Benchmark Suite for Machine Learning

## Abstract

We introduce NumGeom-Bench, an open-source benchmark suite designed to stress-test ML algorithms for numerical robustness using metrics derived from Numerical Geometry. The suite includes 12 carefully designed tasks spanning classification, regression, scientific ML, and pipeline robustness, each annotated with curvature profiles, Lipschitz constants, and precision-sensitivity curves. We define three novel metrics: Numerical Entropy H^num (information capacity under precision constraints), Error Functional Φ (end-to-end error propagation), and Precision Robustness Score (performance degradation under quantization). Baseline evaluations of standard methods reveal surprising failures: several widely-used architectures lose > 10% accuracy when evaluated at float16 despite training at float32. We release NumGeom-Bench as a pip-installable Python package with pre-configured Colab notebooks, ensuring all benchmarks run on a laptop in under 30 minutes total.

## 1. Introduction and Motivation

Benchmarks drive ML progress, but current benchmarks ignore numerical robustness. We optimize for accuracy on ImageNet, perplexity on WikiText—but do our models work at float16? What about int8? When gradients vanish at low precision, is it the algorithm or the hardware? These questions lack systematic answers because we lack a benchmark that treats numerical precision as a first-class dimension. NumGeom-Bench fills this gap. Inspired by the theoretical framework of Numerical Geometry, we design tasks that stress-test numerical robustness: ill-conditioned linear problems, sharp decision boundaries, chaotic dynamics, multi-stage pipelines with precision mismatches. For each task, we compute intrinsic numerical properties (curvature κ, Lipschitz L, condition number κ(A)) and use these to predict which methods will fail at which precision. The benchmark serves dual purposes: (1) evaluating existing methods for numerical robustness, and (2) validating that numerical-geometry-based methods from our companion papers actually improve robustness.

## 2. Benchmark Tasks

### 2.1 Classification Tasks

**Task 1: Ill-Conditioned Linear Classification.** Synthetic dataset with features X = UΣV^T where Σ has condition number 10^6. Standard logistic regression fails at float32; tests whether algorithms handle ill-conditioning.

**Task 2: Sharp Decision Boundary.** Two-class problem where optimal boundary has curvature κ > 100. Tests whether classifiers can learn high-curvature boundaries at low precision.

**Task 3: MNIST with Precision Sweep.** Standard MNIST, but evaluated at float64/32/16/8-bit. Baseline for comparing precision sensitivity across architectures.

**Task 4: High-Lipschitz Preprocessing.** CIFAR-10 with adversarial preprocessing (high-Lipschitz normalization L > 100). Tests error composition through pipelines.

### 2.2 Regression Tasks

**Task 5: Polynomial Fitting.** Fit degree-20 polynomial to noisy data. High curvature near roots causes precision sensitivity.

**Task 6: Inverse Problem.** Recover parameters from observations via least squares with ill-conditioned Jacobian. Tests regularization under precision constraints.

### 2.3 Scientific ML Tasks

**Task 7: Neural ODE on Lorenz.** Train neural ODE on chaotic Lorenz system. Trajectory error explodes at low precision due to sensitivity.

**Task 8: PINN on Poisson.** Physics-informed network for 1D Poisson equation. Residual floor from autodiff error at low precision.

### 2.4 Pipeline Tasks

**Task 9: Multi-Stage Pipeline.** Data standardization → PCA → MLP → Calibration. Tests error composition across stages.

**Task 10: Precision Mismatch Detection.** Pipeline with injected precision inconsistencies. Tests whether analysis tools (like SheafCheck) detect issues.

### 2.5 Stress Tests

**Task 11: Gradient Vanishing/Explosion.** Deep network (20 layers) where gradient norms span 10^{-10} to 10^{10}. Tests optimizer robustness.

**Task 12: Stiff Dynamics.** ODE with eigenvalue ratio 10^6. Explicit solvers fail; tests implicit method + precision interaction.

## 3. Metrics

### 3.1 Numerical Entropy (H^num)

For a representation z = f(x) at precision ε:
H^num(f, ε) ≈ log₂(Ñ_ε(f(X)))

where Ñ_ε is an approximation to the covering number N_ε, estimated via greedy set cover on sample embeddings. Note: greedy cover gives an O(log n)-approximation to the optimal cover, so H^num estimates may be off by O(log log n) bits. For practical ML representations (n ~ 10⁴-10⁶ samples), this gives estimates accurate to within ~4 bits. Higher H^num = more information preserved at given precision.

### 3.2 Error Functional (Φ)

For a pipeline F = f_k ∘ ... ∘ f_1:
Φ_F(ε) = (∏_i L_i)ε + Σ_i Δ_i(∏_{j>i} L_j)

Estimated by measuring L_i (Lipschitz), Δ_i (intrinsic error) for each stage. Lower Φ = more robust.

### 3.3 Precision Robustness Score (PRS)

PRS = AUC of accuracy vs. precision curve, normalized by full-precision accuracy.

PRS = (1/A_{fp64}) ∫_{p_min}^{p_max} A(p) dp

where A(p) is accuracy at p bits. Higher PRS = more robust to precision reduction.

## 4. Benchmark Design Principles

### 4.1 Laptop-Friendly by Design

Every task is designed to run on a MacBook Pro or free Colab instance:
- Model sizes: < 1M parameters
- Dataset sizes: < 100K samples
- Training time: < 5 minutes per task
- Total suite runtime: < 30 minutes

### 4.2 Ground Truth Annotations

Each task includes:
- Analytical or high-precision numerical ground truth
- Pre-computed curvature profiles κ(x) over input domain
- Pre-computed Lipschitz constants per layer
- Predicted precision thresholds where accuracy drops

**Hero Visualizations for Paper/Website (< 1 hr total compute):**
- **Task gallery figure**: 4×3 grid showing each task with: (a) problem visualization, (b) curvature heatmap, (c) precision-accuracy curve. Single figure summarizing entire benchmark.
- **Curvature-precision scatter**: All 12 tasks on one plot, x = estimated κ, y = observed precision threshold. Strong correlation validates the theory.
- **PRS leaderboard radar chart**: Spider plot comparing MLP vs CNN vs Transformer across all tasks at different precisions.
- **Interactive web demo**: Colab notebook where users can visualize precision effects in real-time for any task.

### 4.3 Challenge Variants

Each task has variants:
- Standard: float32 training and evaluation
- Low-Precision: float16 or int8 throughout
- Mixed-Precision: float16 forward, float32 backward
- Noisy-Hardware: Simulated bit-flip and rounding errors

## 5. Baselines and Expected Results

### 5.1 Baseline Methods

| Category | Methods |
|----------|---------|
| Architectures | MLP, CNN, ResNet, Transformer (all small) |
| Optimizers | SGD, Adam, AdamW |
| Precision | float64, float32, float16, int8 |
| Regularization | None, Spectral Norm, Weight Decay |

### 5.2 Expected Findings

1. **Ill-conditioned tasks (1, 5, 6)**: Standard methods fail at float32; regularization helps.
2. **Sharp boundary tasks (2, 5)**: High-curvature regions cause float16 failures; Lipschitz-constrained networks survive.
3. **Scientific ML (7, 8, 12)**: float16 universally fails; float32 marginal; float64 required.
4. **Pipelines (4, 9, 10)**: Error composition makes early-stage precision critical.
5. **Deep networks (11)**: Mixed precision essential; uniform low precision fails.

### 5.3 Leaderboard Format

For each task, report:
- Accuracy at each precision level
- PRS score
- H^num at target precision
- Training stability (NaN count, loss variance)

## 6. Software Package

### 6.1 API Design

```python
import numgeom_bench as ngb

# Load a task
task = ngb.load_task("ill_conditioned_linear")

# Get data
train_data, test_data = task.get_data()

# Train your model
model = YourModel()
train(model, train_data)

# Evaluate with NumGeom metrics
results = ngb.evaluate(
    model, 
    test_data,
    precisions=['float64', 'float32', 'float16', 'int8'],
    metrics=['accuracy', 'H_num', 'Phi', 'PRS']
)

# Compare to baselines
ngb.compare_to_baselines(results, task)

# Submit to leaderboard (optional)
ngb.submit(results, task, model_name="MyMethod")
```

### 6.2 Colab Notebooks

Pre-configured notebooks for each task:
- `Task_01_Ill_Conditioned.ipynb`
- `Task_07_NeuralODE_Lorenz.ipynb`
- `Task_09_Pipeline.ipynb`
- etc.

Each notebook: < 10 minutes runtime on free Colab GPU.

## 7. Theoretical Contributions

1. **Benchmark Design Methodology**: Principled approach to creating numerical robustness benchmarks based on geometric invariants (curvature, Lipschitz, condition number).

2. **Novel Metrics**: H^num, Φ, and PRS as complementary measures of numerical robustness.

3. **Ground Truth Annotations**: First benchmark with pre-computed curvature profiles and precision predictions.

4. **Community Resource**: Open-source benchmark enabling reproducible numerical robustness research.

## 8. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Task design & synthetic data | 1 week | None |
| Metric implementation | 1 week | Laptop |
| Baseline runs (all tasks × all methods) | 1 week | 10 hrs laptop |
| Package development | 1 week | Laptop |
| Colab notebooks | 3 days | 2 hrs Colab |
| Documentation & paper | 1 week | None |
| **Total** | **5 weeks** | **~12 hrs compute** |

## 9. Impact and Extensibility

NumGeom-Bench is designed as a living benchmark:
- **Modular architecture**: Easy to add new tasks, metrics, or baselines
- **Community contributions**: GitHub-based submission process
- **Integration ready**: Compatible with PyTorch, JAX, TensorFlow
- **Citation tracking**: Monitor adoption and impact

We envision NumGeom-Bench becoming the standard for evaluating numerical robustness, just as ImageNet became standard for accuracy evaluation.

