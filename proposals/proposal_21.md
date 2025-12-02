# Proposal 21: Numerical Geometry of Reinforcement Learning: Curvature of the Bellman Operator

## Abstract

We apply Numerical Geometry to reinforcement learning, modeling the Bellman operator as a numerical morphism with explicit Lipschitz constant and curvature bounds. We prove that value iteration and Q-learning accumulate numerical errors according to the Stability Composition Theorem, deriving precision lower bounds for stable policy evaluation. Our key result: when precision falls below a critical threshold p*(γ, κ), numerical noise dominates the contraction, causing the effective discount factor to exceed 1 and value iteration to diverge. We provide concrete precision requirements as functions of discount factor γ, reward scale, and value function curvature. Experiments on small gridworlds, bandits, and CartPole with tiny function approximators verify that low-precision training fails exactly when predicted and succeeds in theoretically safe regimes. All experiments run on a laptop in under 2 hours.

## 1. Introduction and Motivation

Reinforcement learning on edge devices—robots, embedded systems, mobile phones—demands low-precision arithmetic for efficiency. But RL algorithms like value iteration and Q-learning are iterative: they apply the Bellman operator T repeatedly, accumulating numerical errors at each step. When does this accumulation break the algorithm? Current practice offers no principled answer. We provide one using Numerical Geometry. The Bellman operator T is a contraction with Lipschitz constant γ (the discount factor), but finite-precision arithmetic adds error Δ_T at each step. The Stability Composition Theorem lets us track error accumulation: after k iterations, the total error is bounded by Φ_{T^k}(ε). Our key insight is that there's a critical precision p* below which the per-step numerical error Δ_T exceeds the per-step contraction (1-γ), causing divergence. We derive p* explicitly in terms of γ, reward scale, and curvature of the value function.

## 2. Technical Approach

### 2.1 Bellman Operator as Numerical Morphism

The Bellman operator T: V → V acts on value functions:

(TV)(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]

We model V as living in a numerical space (V, d_∞, R_p) where d_∞ is the sup-norm and R_p is the set of p-bit representable value functions.

**Lipschitz constant**: By standard RL theory, ||TV - TV'||_∞ ≤ γ ||V - V'||_∞, so L_T = γ.

**Intrinsic error**: Each Bellman update involves:
1. Reward lookup: error O(ε_mach · |R_max|)
2. Expectation over transitions: error O(ε_mach · |S| · ||V||_∞)
3. Max over actions: exact (discrete)
4. Final rounding: error O(ε_mach · ||TV||_∞)

Combining: Δ_T = O(ε_mach · (|R_max| + |S| · ||V||_∞))

**Curvature of value functions**: In continuous state spaces with function approximation, the value function V_θ has curvature κ_V depending on the approximator. High curvature means small state perturbations cause large value changes, requiring higher precision.

### 2.2 Error Accumulation in Value Iteration

**Theorem (Value Iteration Error Bound).** After k Bellman iterations at precision p with machine epsilon ε_p = 2^{-p}, the numerical value function Ṽ_k satisfies:

||Ṽ_k - V*||_∞ ≤ γ^k ||V_0 - V*||_∞ + (1 - γ^k)/(1 - γ) · Δ_T

where the first term is the standard contraction and the second is accumulated numerical error.

**Proof.** By the Stability Composition Theorem:
Φ_{T^k}(ε) = γ^k · ε + Σ_{i=0}^{k-1} γ^i · Δ_T = γ^k · ε + Δ_T · (1 - γ^k)/(1 - γ)

In the limit k → ∞, numerical error saturates at Δ_T / (1 - γ).

### 2.3 Precision Lower Bound for Stable RL

**Theorem (RL Precision Lower Bound).** For value iteration to converge to within ε of V*, we need precision:

p ≥ log₂((|R_max| + |S| · V_max) / ((1 - γ) · ε))

where V_max = R_max / (1 - γ) is the maximum value scale.

**Proof.** The steady-state numerical error is Δ_T / (1 - γ). Setting this ≤ ε and solving for ε_p = 2^{-p} gives the bound.

**Critical Regime**: When Δ_T > (1 - γ) · ||V||_∞, numerical noise exceeds contraction strength. The effective operator T̃ = T + noise behaves as if γ_eff > 1, causing divergence.

### 2.4 Extension to Q-Learning and TD(0)

For stochastic algorithms, we add a term for gradient/update noise:

**Q-Learning Error Model**:
Q_{t+1}(s,a) = Q_t(s,a) + α [r + γ max_{a'} Q_t(s',a') - Q_t(s,a)] + noise_t

The numerical error in the TD target adds to the standard stochastic noise. At low precision:
- Target computation error: O(ε_mach · (|r| + γ · Q_max))
- Update rounding: O(ε_mach · α · |δ_t|)

**Proposition.** Q-learning converges if α_t → 0, Σ α_t = ∞, Σ α_t² < ∞, AND the numerical precision satisfies:

p ≥ log₂((|R_max| + γ · Q_max) / ((1 - γ) · α_min))

where α_min is the minimum learning rate used. This ensures numerical noise doesn't overwhelm the TD update.

## 3. Laptop-Friendly Implementation

All experiments use small, tabular, or tiny-function-approximator RL:

1. **Tabular environments**: 4×4 and 8×8 gridworlds (16-64 states), FrozenLake
2. **Simple bandits**: 10-arm bandit with known reward distributions
3. **Tiny function approximation**: CartPole with 2-layer MLP (< 1K parameters)
4. **Precision simulation**: Use float64 with explicit rounding to simulate {4, 8, 16, 32} bit precision
5. **Fast iteration**: Value iteration converges in < 100 iterations for small MDPs

Total compute: approximately 2 hours on a laptop for all experiments.

## 4. Experimental Design

### 4.1 Environments and Algorithms

| Environment | States | Actions | Algorithm | Precision Levels |
|-------------|--------|---------|-----------|------------------|
| 4×4 Gridworld | 16 | 4 | Value Iteration | 4, 8, 16, 32, 64 bit |
| 8×8 Gridworld | 64 | 4 | Value Iteration | 4, 8, 16, 32, 64 bit |
| FrozenLake | 16 | 4 | Q-Learning | 8, 16, 32 bit |
| 10-Arm Bandit | 1 | 10 | UCB, ε-greedy | 8, 16, 32 bit |
| CartPole-Tiny | ∞ (approx) | 2 | DQN (tiny) | 16, 32 bit |

### 4.2 Experiments

**Experiment 1: Precision Threshold Detection.** For each (environment, γ) pair, run value iteration at decreasing precision. Identify the threshold p* where convergence fails. Compare to theoretical prediction.

**Experiment 2: Error Accumulation Tracking.** Track ||Ṽ_k - V*|| over iterations at each precision level. Verify the error follows the predicted trajectory Φ_{T^k}.

**Experiment 3: Q-Learning Stability.** Run Q-learning with fixed learning rate schedule at different precisions. Measure: (a) convergence rate, (b) final policy quality, (c) training instability (variance of Q-values).

**Experiment 4: Discount Factor Sensitivity.** Vary γ from 0.5 to 0.99 and measure how precision requirements scale. Verify the 1/(1-γ) dependence.

**Experiment 5: Function Approximation.** On CartPole with tiny DQN, compare float32 vs float16 training. Show that float16 fails for γ > 0.95 but succeeds for γ ≤ 0.9.

### 4.3 Expected Results

1. Observed precision threshold p* matches theoretical prediction within ±2 bits.
2. Error accumulation curves match Φ_{T^k} predictions within 5x.
3. Q-learning at 8-bit fails for γ > 0.9, succeeds for γ ≤ 0.8.
4. Precision requirement scales as log(1/(1-γ)), matching theory.
5. Float16 DQN on CartPole requires γ ≤ 0.9 for stable training.

**High-Impact Visualizations (< 20 min compute):**
- **Precision-discount phase diagram** (5 min): 2D plot with x = precision bits, y = γ. Color = converged (green) vs diverged (red). Clear boundary matching theoretical curve p* = log₂(1/(1-γ)). Single most compelling figure.
- **Value function heatmaps** (3 min): 4×4 gridworld showing V(s) at 64/32/16/8 bits. Low precision "blurs" optimal policy—visually obvious degradation.
- **Error saturation curves** (5 min): ||Ṽ_k - V*|| vs iteration k for each precision. Shows characteristic saturation at Δ_T/(1-γ). Include theoretical prediction as dashed line.
- **Q-value stability sparklines** (5 min): Small multiples of Q(s,a) trajectories during training at different precisions. Stable vs chaotic at a glance.

## 5. Theoretical Contributions Summary

1. **Bellman Operator as Numerical Morphism**: First rigorous numerical error model for the Bellman operator with explicit Lipschitz and error bounds.
2. **Precision Lower Bound for RL**: Theorem relating discount factor, reward scale, and minimum bit-depth for stable value iteration.
3. **Stochastic Extension**: Error model for Q-learning and TD(0) incorporating both stochastic and numerical noise.
4. **Practical Guidelines**: Concrete rules for precision selection in embedded RL applications.

## 5.1 Usable Artifacts

1. **PrecisionChecker for RL**: Python library function `check_rl_precision(gamma, R_max, V_max, target_error)` that returns minimum required bit-depth for stable value iteration. Works for any MDP.
2. **StableRL Wrapper**: Wrapper class that monitors numerical error during Q-learning and raises warnings when approaching instability thresholds.
3. **Precision-Discount Lookup Table**: Precomputed table of safe (precision, γ) pairs for common reward scales, usable as a deployment guide.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Theory development | 1 week | None |
| Tabular experiments | 3 days | 30 min laptop |
| Q-learning experiments | 3 days | 30 min laptop |
| Function approx experiments | 3 days | 1 hr laptop |
| Visualization | 2 days | 20 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~2 hrs laptop** |
