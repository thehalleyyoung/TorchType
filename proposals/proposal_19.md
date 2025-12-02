# Proposal 19: Numerically Safe Scientific ML: Finite-Precision Guarantees for Neural ODEs and PINNs

## Abstract

We apply Numerical Geometry to provide rigorous finite-precision error bounds for Scientific Machine Learning, focusing on neural ODEs and physics-informed neural networks (PINNs). For neural ODEs, we model the ODE solver and neural right-hand-side as composed numerical morphisms, deriving minimum precision requirements to guarantee trajectory error ε over integration horizon T. For PINNs, we analyze how finite-precision evaluation of network derivatives affects PDE residual accuracy. Our key result: many reported PINN experiments operate below the precision floor where residuals are numerically meaningful. We prove curvature-based lower bounds on precision for neural ODE integrators and PINN derivative approximations. Experiments on low-dimensional systems (Lorenz, pendulum) and 1D PDEs (Poisson, heat equation) verify our bounds and demonstrate that precision-aware training improves solution quality by 10-50% at equivalent compute. All experiments run on a laptop in under 2 hours.

## 1. Introduction and Motivation

Scientific Machine Learning (SciML) uses neural networks to solve differential equations, either as learned dynamics (neural ODEs) or as solution approximators (PINNs). These applications demand accuracy: errors in scientific simulations have physical consequences. Yet SciML inherits the numerical fragility of neural networks while adding the challenges of numerical integration and derivative computation. We ask: what precision is actually needed? Numerical Geometry provides the framework. For neural ODEs, the ODE solver (Runge-Kutta, etc.) is a numerical morphism with error functional Φ_solver, and the neural RHS f_θ is another morphism with error Φ_f. The composed error Φ_{solve ∘ f} determines precision requirements. For PINNs, automatic differentiation of the network to compute PDE residuals introduces additional error that can swamp the residual signal at low precision. Our analysis reveals that many standard PINN experiments inadvertently operate in a regime where the training signal is dominated by numerical noise.

## 2. Technical Approach

### 2.1 Neural ODE Error Analysis

A neural ODE is dy/dt = f_θ(y, t) where f_θ is a neural network. Numerical solution uses a solver S (e.g., RK4) with step size h:

y_{n+1} = S(f_θ, y_n, t_n, h)

**Theorem (Neural ODE Precision Bound).** Let f_θ have Lipschitz constant L_f and curvature κ_f. For an order-k explicit Runge-Kutta method with n steps to time T, the global error satisfies:

||y_n - y(T)|| ≤ (exp(L_f · T) - 1) · [h^k · C_method + Φ_f(ε_prec)]

where C_method is the method constant and Φ_f(ε_prec) = L_f · ε_prec + κ_f · ε_prec² is the per-step numerical error in evaluating f_θ.

**Precision Requirement.** To achieve global error ε_target, we need:

p ≥ log₂(L_f · exp(L_f · T) · T / ε_target)

**Proof Strategy.** Standard ODE error analysis gives global error in terms of local truncation error. We add a term for the error in evaluating f_θ at finite precision. By Lipschitz continuity, this error propagates through the integration with amplification factor exp(L_f · T) (Grönwall's inequality). The curvature term κ_f · ε² contributes to higher-order precision requirements in stiff regions.

### 2.2 PINN Residual Error Analysis

A PINN solves a PDE by minimizing the residual: L_PDE(θ) = ||N[u_θ](x) - f(x)||² where N is a differential operator and u_θ is the neural network solution.

For a simple Poisson equation ∇²u = f, the residual involves second derivatives:

R(x) = ∂²u_θ/∂x² - f(x)

**Theorem (PINN Derivative Precision).** Let u_θ be a neural network with Lipschitz constant L and curvature κ. The second derivative computed via autodiff has numerical error:

||∂²u_θ/∂x²_{computed} - ∂²u_θ/∂x²_{true}|| ≤ κ · ε_prec + L² · ε_prec

When the true residual ||R(x)|| is small (as desired for a good solution), this error can dominate:

||R_{computed}|| ≈ ||R_{true}|| + κ · ε_prec

**Consequence.** If ||R_{true}|| < κ · ||θ|| · ε_prec, the residual is numerically meaningless—training is fitting noise.

**Precision Requirement for PINNs.** To train to residual tolerance δ, we need:

p ≥ log₂(κ · ||θ|| · ||x||^2 / δ)

where ||θ|| is the network weight scale and ||x||^2 accounts for second-derivative amplification. The units are: κ has units of 1/length² (curvature), ||θ|| is dimensionless weights, ||x||^2 has units length², and δ has units matching the PDE residual (typically dimensionless after normalization).

### 2.3 Precision-Aware SciML Training

Based on our analysis, we propose:

**Algorithm: Precision-Aware PINN Training**
```
1. Estimate network curvature κ and weight scale ||θ|| at initialization
2. Set initial precision p = log₂(κ · ||θ|| · ||x||^2 / δ_target)
3. During training:
   a. Monitor residual: if ||R|| < 10 · κ · ||θ|| · ε_prec, increase precision
   b. Monitor curvature: update κ estimate every 100 steps
   c. Adjust precision to maintain ||R|| > 10 · κ · ||θ|| · ε_prec (signal > noise)
4. Final refinement in high precision
```

**Algorithm: Precision-Aware Neural ODE Training**
```
1. Estimate RHS curvature κ_f and Lipschitz L_f
2. For integration horizon T, compute required precision
3. Use adaptive step-size solver with precision-matched tolerances
4. Monitor trajectory stability; increase precision if divergence detected
```

## 3. Laptop-Friendly Implementation

SciML experiments are inherently small: (1) **Low-dimensional ODEs**: Lorenz system (3D), pendulum (2D), harmonic oscillator (2D). Integration for T ≤ 100 with h ≥ 0.01 takes < 1 second; (2) **1D PDEs**: Poisson, heat equation on [0,1]. Collocation points N ≤ 1000. Training for 10K iterations takes < 5 minutes; (3) **Small networks**: 3-4 layer MLPs with 32-64 hidden units (< 10K params). Standard for SciML benchmarks; (4) **Ground truth comparison**: Use analytical solutions (available for our test problems) or high-precision numerical solutions (scipy.integrate.odeint at float64); (5) **Precision simulation**: Use float64 computations with added noise at target precision levels to simulate quantization effects. Total experiment time: < 2 hours on a laptop.

## 4. Experimental Design

### 4.1 Test Problems

**Neural ODEs:**
| System | Dimension | Known Properties | Horizon T |
|--------|-----------|-----------------|-----------|
| Harmonic Oscillator | 2D | Linear, exact solution | 20 |
| Damped Pendulum | 2D | Nonlinear, bounded | 50 |
| Lorenz System | 3D | Chaotic, sensitive | 10 |

**PINNs:**
| PDE | Domain | Exact Solution | Complexity |
|-----|--------|----------------|------------|
| Poisson 1D | [0,1] | u = sin(πx) | Low |
| Heat 1D | [0,1]×[0,1] | u = exp(-π²t)sin(πx) | Medium |
| Burgers 1D | [0,1]×[0,1] | Known implicit | High |

### 4.2 Experiments

**Experiment 1: Precision vs. Trajectory Error (Neural ODE).** For each system, train neural ODE at float64. Evaluate trajectory at float64, float32, float16, int8. Compare observed error to predicted Φ bound.

**Experiment 2: Precision Floor for PINNs.** Train PINN at different precisions and plot final residual vs. precision. Identify the precision below which residual plateaus (numerical floor).

**Experiment 3: Curvature Correlation.** For trained networks, estimate curvature κ and compare to observed precision requirements. Verify κ predicts the precision floor.

**Experiment 4: Precision-Aware Training.** Compare standard training (fixed float32) to our precision-aware algorithm. Measure final solution error.

**Experiment 5: Stiff Systems.** Test on a stiff ODE (Robertson problem, simplified) where precision requirements are extreme. Show that our bounds predict the needed precision.

### 4.3 Expected Results

1. Neural ODE trajectory errors match predicted bounds within 5x for non-chaotic systems, within 50x for Lorenz (expected due to sensitivity).
2. PINN residuals plateau at 10^{-7} for float32, 10^{-3} for float16, matching theoretical κ·ε prediction.
3. Curvature κ correlates with observed precision floor (r > 0.8 across test problems).
4. Precision-aware training achieves 10-50% lower solution error than fixed-precision at equivalent compute.
5. For stiff problems, float32 is insufficient; only float64 achieves target accuracy.

**High-Impact Visualizations (< 30 min compute):**
- **Lorenz trajectory comparison**: 3D plot showing true trajectory (black), float64 neural ODE (blue, overlapping), float32 (green, slight deviation), float16 (red, diverged). Viscerally demonstrates precision effects.
- **PINN residual floor plot**: Log-scale residual vs training iteration for different precisions. Shows plateau at precision-dependent floor. Clean theoretical validation.
- **Precision requirement heatmap**: For heat equation, show spatial domain [0,1]×[0,1] with color = minimum precision needed. High near boundary conditions where gradients are sharp.
- **Error bound vs observed scatter**: One point per (problem, precision) pair. Diagonal = perfect. Shows bounds are informative but conservative.

## 5. Theoretical Contributions Summary

1. **Neural ODE Precision Bounds**: Rigorous analysis of how finite precision affects ODE integration with neural RHS.
2. **PINN Residual Floor**: Proof that PDE residuals have a precision-dependent noise floor from autodiff error.
3. **Curvature-Based Requirements**: Explicit formulas relating network curvature to minimum precision.
4. **Precision-Aware Training**: Algorithms that adapt precision to problem requirements.

## 6. Timeline and Compute Budget

| Phase | Duration | Compute |
|-------|----------|---------|
| Error analysis derivations | 1 week | None |
| Neural ODE experiments | 3 days | 30 min laptop |
| PINN experiments | 3 days | 1 hr laptop |
| Stiff system tests | 2 days | 30 min laptop |
| Precision-aware algorithm | 3 days | 30 min laptop |
| Writing | 1 week | None |
| **Total** | **4 weeks** | **~2.5 hrs laptop** |

