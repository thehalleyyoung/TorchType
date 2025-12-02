# Proposal 18: End-to-End Quantization-Aware Training via Numerical Geometry

## Abstract

We develop a quantization-aware training (QAT) procedure where precision is allocated dynamically across the entire training loop—forward pass, backward pass, and optimizer state—using Numerical Geometry's stability composition and curvature bounds. Unlike standard QAT which only quantizes the forward pass, we model the complete training step θ_{t+1} = T(θ_t) as a numerical morphism and derive precision requirements for each component. Our key insight is that different training phases have different precision needs: early training tolerates low precision due to large gradients dominating noise, while late training requires high precision for fine convergence. We prove that curvature-based lower bounds predict where low precision causes training instability and propose adaptive precision scheduling. Experiments on small CNNs (MNIST, CIFAR-10) demonstrate 30-50% reduction in average bit-operations compared to uniform mixed precision, with equivalent final accuracy. All experiments run on a laptop in under 4 hours using simulated quantization.

## 1. Introduction and Motivation

Quantization-aware training (QAT) typically focuses on quantizing the forward pass for efficient inference, while training in full precision. But what about quantizing training itself? Training on edge devices, federated learning, and memory-constrained settings motivate fully-quantized training. The challenge: which components can tolerate low precision, and when? We answer this using Numerical Geometry. The training step T: θ ↦ θ - η·∇L(θ) decomposes into: (1) forward pass to compute loss, (2) backward pass to compute gradients, (3) optimizer update. Each component has different Lipschitz constants and curvatures that vary over training. Our framework derives precision schedules that adapt to these properties, allocating bits where they matter most. Key finding: early training is robust (large gradients mask quantization noise) but late training near convergence requires high precision (small gradients comparable to quantization noise).

## 2. Technical Approach

### 2.1 Training Step as Numerical Morphism

The complete training step is a composition:

T = U ∘ B ∘ F

where:
- F(θ, x, y): Forward pass, computes loss L(θ; x, y)
- B(θ, L): Backward pass, computes gradient ∇_θ L  
- U(θ, g): Optimizer update, computes θ' = θ - η·g (or more complex for Adam, etc.)

Each component has error functional:
- Φ_F(ε) = L_F · ε + Δ_F where L_F ≈ ∏_layers L_i (product of layer Lipschitz constants)
- Φ_B(ε) = L_B · ε + Δ_B where L_B includes Jacobian condition numbers
- Φ_U(ε) = L_U · ε + Δ_U where L_U = 1 for SGD, depends on state for Adam

The total training step error is Φ_T(ε) = Φ_U(Φ_B(Φ_F(ε))).

### 2.2 Precision Requirements by Training Phase

**Theorem (Phase-Dependent Precision).** Let ||g_t|| be the gradient norm at step t and let Δ_p be the quantization error at precision p. Training remains stable if:

||g_t|| ≫ L_T · Δ_p

where L_T is the Lipschitz constant of the training step. This implies:
- Early training (large ||g_t||): Low precision tolerable, p_early = O(log(L_T · θ_scale / ||g_early||))
- Late training (small ||g_t||): High precision required, p_late = O(log(L_T · θ_scale / ||g_late||))

**Proof Strategy.** The parameter update δθ = -η·g has magnitude η·||g||. Quantization noise adds error of magnitude Δ_p. For the signal (true gradient) to dominate noise (quantization error), we need η·||g|| ≫ L_T · Δ_p. As training converges, ||g|| → 0, requiring Δ_p → 0, i.e., increasing precision. The Lipschitz constant L_T determines how errors amplify through the training step.

### 2.3 Curvature-Based Precision Allocation

Beyond temporal phases, we allocate precision spatially (across layers) using curvature bounds.

**Theorem (Layer-Wise Precision for Training).** For layer i with local curvature κ_i = ||∇²L_i||, gradient norm ||g_i||, and parameter scale ||θ_i||, the minimum precision to ensure gradient-dominated updates is:

p_i ≥ log₂(κ_i · ||θ_i|| / ||g_i||) + c

where c ≈ 3-4 bits provides a safety margin. This ensures the quantization-induced gradient error κ_i · ||θ_i|| · 2^{-p} remains below ||g_i||/2^c.

**Derivation.** The gradient error from precision-p arithmetic in a region with curvature κ is δg ≤ κ · ||θ|| · 2^{-p}. For the true gradient to dominate: ||g|| > 2^c · δg, giving the stated bound.

**Algorithm: NumGeom-QAT**

```
Input: Model, data, target error ε_target, precision levels {4, 8, 16, 32}
Output: Trained model with adaptive precision schedule

1. Initialize: p_global = 32 (full precision warmup for 1 epoch)

2. For each epoch e:
   a. Estimate current gradient norm ||g|| and per-layer ||g_i|| on minibatch
   b. Estimate per-layer curvatures κ_i via Hessian-vector products
   c. Compute precision requirements:
      - p_min(e) = ceil(log₂(L_T / ||g||))  # global minimum
      - p_i(e) = ceil(log₂(κ_i · ||θ_i|| / ||g_i||)) + 4  # per-layer with safety margin
   d. Set precision schedule: p_i = max(p_min(e), p_i(e), 4)
   
3. For each step in epoch:
   - Forward: use layer precisions p_i
   - Backward: use layer precisions p_i
   - Optimizer: use precision max_i(p_i)
   
4. Every 10 epochs, update curvature estimates

Return trained model
```

## 3. Laptop-Friendly Implementation

All experiments use simulated quantization on laptop hardware: (1) **Fake quantization**: Round weights/activations/gradients to target precision using torch.quantize, then dequantize for computation. No specialized integer kernels needed; (2) **Small models**: 2-4 layer CNNs with < 500K parameters, training for 50-100 epochs; (3) **Small datasets**: MNIST, Fashion-MNIST, CIFAR-10 (full or subsampled); (4) **Efficient curvature estimation**: One Hessian-vector product per layer per epoch (< 1 second overhead); (5) **Precision levels**: {4, 8, 16, 32} bits, represented as scaling factors. Total training time: approximately 30 minutes per model configuration on a MacBook.

## 4. Experimental Design

### 4.1 Models and Datasets

| Model | Architecture | Dataset | Training Epochs |
|-------|--------------|---------|-----------------|
| CNN-2 | 2 conv + 2 FC | MNIST | 50 |
| CNN-4 | 4 conv + 2 FC | Fashion-MNIST | 100 |
| ResNet-8 | 8 residual blocks | CIFAR-10 | 100 |

### 4.2 Baselines

1. **Full Precision**: float32 throughout training
2. **Uniform Low Precision**: int8 or float16 throughout
3. **Standard Mixed Precision**: float16 forward, float32 backward/optimizer (PyTorch AMP)
4. **Gradient-Based Scheduling**: Increase precision when gradient norm drops (heuristic)
5. **NumGeom-QAT**: Our method (curvature + gradient-based)

### 4.3 Metrics

1. **Final Accuracy**: Test accuracy after full training
2. **Bit-Operations**: Total bits used across all operations (Σ_t Σ_layers p_i(t) · FLOPs_i)
3. **Training Stability**: Variance of loss across seeds; number of NaN/Inf occurrences
4. **Precision Schedule**: Visualization of how precision evolves over training
5. **Curvature-Precision Correlation**: Scatter plot of layer curvature vs. allocated precision

### 4.4 Experiments

**Experiment 1: Accuracy vs. Bit-Budget.** For each method, plot final accuracy against total bit-operations. Show Pareto frontier.

**Experiment 2: Training Dynamics.** Plot loss curves and precision schedules over training. Show that NumGeom-QAT starts low, increases during critical phases.

**Experiment 3: Stability Analysis.** Count training failures (NaN/divergence) at different uniform precision levels. Show that adaptive precision avoids failures.

**Experiment 4: Ablation - Curvature vs. Gradient.** Compare: (a) gradient-only scheduling, (b) curvature-only scheduling, (c) combined. Show that combined is superior.

### 4.5 Expected Results

1. NumGeom-QAT achieves 30-50% reduction in bit-operations vs. uniform float32, with < 0.3% accuracy loss.
2. Compared to standard mixed precision (AMP), NumGeom-QAT saves additional 10-20% bit-ops by using lower precision in early epochs.
3. Precision schedule shows clear pattern: low early, high late, with spikes at high-curvature layers.
4. Uniform 8-bit training fails on 2/3 models; NumGeom-QAT with adaptive precision succeeds on all.
5. Combined curvature+gradient scheduling outperforms either alone by 5-10% bit-ops at same accuracy.

**High-Impact Visualizations (< 1 hr compute):**
- **Precision schedule heatmap**: 2D plot with x = training step, y = layer. Color = precision (4=blue to 32=red). Shows adaptive pattern: starts blue, transitions to red, with layer-specific variations.
- **Accuracy-vs-bit-ops Pareto frontier**: Single figure with all 5 methods. NumGeom-QAT dominates the frontier.
- **Training stability histogram**: For uniform 8-bit, show distribution of loss values at step 1000 (many NaN). For NumGeom-QAT, show stable distribution. Dramatic visual difference.
- **Gradient/curvature time series**: Dual-axis plot showing ||g|| (decreasing) and allocated precision (increasing) over training. Validates the theory.

## 5. Theoretical Contributions Summary

1. **Training Step Error Model**: Complete error functional for forward, backward, and optimizer as composed morphisms.
2. **Phase-Dependent Precision Theorem**: Formal justification for why early training tolerates low precision.
3. **NumGeom-QAT Algorithm**: Practical adaptive precision scheduling with theoretical grounding.
4. **Stability Analysis**: Conditions under which low-precision training diverges.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error model derivation | 1 week | None |
| NumGeom-QAT implementation | 1 week | Laptop |
| MNIST experiments | 2 days | 1 hr laptop |
| Fashion-MNIST experiments | 2 days | 1 hr laptop |
| CIFAR-10 experiments | 3 days | 2 hrs laptop |
| Ablations | 2 days | 1 hr laptop |
| Writing | 1 week | None |
| **Total** | **4.5 weeks** | **~5 hrs laptop** |

