"""
Numerical Geometry of Reinforcement Learning

This module implements the Bellman operator as a numerical morphism with explicit
Lipschitz constants and error bounds, following the Stability Composition Theorem
from Numerical Geometry.

Key concepts:
- Bellman operator T: V → V as a numerical morphism with L_T = γ (discount factor)
- Intrinsic error Δ_T from finite-precision arithmetic
- Error accumulation via Stability Composition Theorem
- Critical precision threshold p* below which value iteration diverges
"""

import torch
import numpy as np
from typing import Tuple, Optional, Callable, Dict, Any
from dataclasses import dataclass
import warnings


@dataclass
class NumericalMorphism:
    """
    Represents a numerical morphism with Lipschitz constant and intrinsic error.
    
    For a function f: X → Y in finite precision:
    - L: Lipschitz constant (||f(x) - f(y)|| ≤ L ||x - y||)
    - Δ: Intrinsic roundoff error (||f̃(x) - f(x)|| ≤ Δ)
    
    Error functional: Φ(ε) = L·ε + Δ
    """
    L: float  # Lipschitz constant
    Delta: float  # Intrinsic error
    name: str = "morphism"
    
    def error_functional(self, eps: float) -> float:
        """Compute error functional Φ(ε) = L·ε + Δ"""
        return self.L * eps + self.Delta
    
    def compose(self, other: 'NumericalMorphism') -> 'NumericalMorphism':
        """
        Compose two numerical morphisms via Stability Composition Theorem.
        
        For f: X → Y and g: Y → Z:
        (g ∘ f) has:
        - Lipschitz constant: L_{g∘f} = L_g · L_f
        - Intrinsic error: Δ_{g∘f} = L_g · Δ_f + Δ_g
        """
        return NumericalMorphism(
            L=self.L * other.L,
            Delta=self.L * other.Delta + self.Delta,
            name=f"{self.name}∘{other.name}"
        )
    
    def iterate(self, n: int) -> 'NumericalMorphism':
        """
        Compute n-fold composition f^n via stability composition.
        
        For Bellman operator T^k:
        - L_{T^k} = γ^k
        - Δ_{T^k} = Δ_T · (1 - γ^k) / (1 - γ)  [geometric series]
        """
        if n == 0:
            return NumericalMorphism(L=1.0, Delta=0.0, name="id")
        elif n == 1:
            return self
        else:
            # Use geometric series formula for efficiency
            L_n = self.L ** n
            if abs(self.L - 1.0) < 1e-10:
                Delta_n = n * self.Delta
            else:
                Delta_n = self.Delta * (1 - L_n) / (1 - self.L)
            return NumericalMorphism(
                L=L_n,
                Delta=Delta_n,
                name=f"{self.name}^{n}"
            )


class BellmanOperator:
    """
    Bellman operator as a numerical morphism.
    
    The Bellman operator T: V → V is defined as:
    (TV)(s) = max_a [R(s,a) + γ Σ_{s'} P(s'|s,a) V(s')]
    
    Numerical properties:
    - Lipschitz constant: L_T = γ (standard RL theory)
    - Intrinsic error: Δ_T = O(ε_mach · (R_max + |S| · V_max))
      arising from: reward lookup, expectation, max, rounding
    """
    
    def __init__(
        self,
        rewards: torch.Tensor,  # Shape: (n_states, n_actions)
        transitions: torch.Tensor,  # Shape: (n_states, n_actions, n_states)
        gamma: float,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ):
        self.n_states = rewards.shape[0]
        self.n_actions = rewards.shape[1]
        self.gamma = gamma
        self.dtype = dtype
        self.device = device
        
        # Convert to specified precision
        self.rewards = rewards.to(dtype=dtype, device=device)
        self.transitions = transitions.to(dtype=dtype, device=device)
        
        # Compute numerical properties
        self.R_max = torch.max(torch.abs(self.rewards)).item()
        self.V_max = self.R_max / (1 - gamma) if gamma < 1 else float('inf')
        
        # Machine epsilon for current dtype
        self.eps_mach = torch.finfo(dtype).eps
        
        # Intrinsic error estimate
        self.Delta_T = self._compute_intrinsic_error()
        
        # Create numerical morphism
        self.morphism = NumericalMorphism(
            L=gamma,
            Delta=self.Delta_T,
            name="Bellman"
        )
    
    def _compute_intrinsic_error(self) -> float:
        """
        Compute intrinsic error Δ_T of Bellman operator.
        
        Error sources:
        1. Reward lookup: O(ε_mach · R_max)
        2. Expectation over transitions: O(ε_mach · sqrt(|S|) · V_max) [by Cauchy-Schwarz]
        3. Max over actions: exact (discrete)
        4. Final rounding: O(ε_mach · ||TV||_∞) ≈ O(ε_mach · V_max)
        
        Tighter estimate based on probabilistic error analysis:
        Δ_T = ε_mach · (R_max + (sqrt(|S|) + 1)·V_max)
        
        The sqrt(|S|) comes from standard error of sum of |S| products,
        where transition probabilities sum to 1.
        """
        eps = float(self.eps_mach)
        # Use sqrt(n_states) instead of n_states for more realistic bound
        Delta = eps * (self.R_max + (np.sqrt(self.n_states) + 1) * self.V_max)
        return Delta
    
    def apply(self, V: torch.Tensor) -> torch.Tensor:
        """
        Apply Bellman operator: (TV)(s) = max_a [R(s,a) + γ E[V(s')|s,a]]
        
        Args:
            V: Value function, shape (n_states,)
        
        Returns:
            TV: Updated value function, shape (n_states,)
        """
        V = V.to(dtype=self.dtype, device=self.device)
        
        # Compute expected next values: E[V(s')|s,a] for each (s,a)
        # transitions[s,a,:] is P(·|s,a), so dot product gives expectation
        expected_next = torch.matmul(self.transitions, V)  # (n_states, n_actions)
        
        # Q-values: Q(s,a) = R(s,a) + γ E[V(s')|s,a]
        Q = self.rewards + self.gamma * expected_next
        
        # Value function: V(s) = max_a Q(s,a)
        TV = torch.max(Q, dim=1)[0]
        
        return TV
    
    def value_iteration(
        self,
        V_init: Optional[torch.Tensor] = None,
        max_iters: int = 1000,
        tol: float = 1e-6,
        track_error: bool = True,
        V_star: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        Run value iteration with error tracking.
        
        Returns dict with:
        - V_final: Final value function
        - V_history: Value functions over iterations (if track_error)
        - error_history: ||V_k - V*|| over iterations (if V_star provided)
        - iterations: Number of iterations
        - converged: Whether converged within tol
        - theoretical_error: Predicted error from stability theorem
        """
        if V_init is None:
            V = torch.zeros(self.n_states, dtype=self.dtype, device=self.device)
        else:
            V = V_init.to(dtype=self.dtype, device=self.device)
        
        V_history = [V.clone()] if track_error else []
        error_history = []
        
        for k in range(max_iters):
            V_new = self.apply(V)
            
            if track_error:
                V_history.append(V_new.clone())
            
            if V_star is not None:
                error = torch.max(torch.abs(V_new - V_star)).item()
                error_history.append(error)
            
            # Check convergence
            delta = torch.max(torch.abs(V_new - V)).item()
            V = V_new
            
            if delta < tol:
                break
        
        # Compute theoretical error bound
        morphism_k = self.morphism.iterate(k + 1)
        if V_star is not None:
            init_error = torch.max(torch.abs(V_init - V_star)).item() if V_init is not None else self.V_max
            theoretical_error = morphism_k.error_functional(init_error)
        else:
            theoretical_error = morphism_k.Delta  # Just accumulated roundoff
        
        result = {
            'V_final': V,
            'iterations': k + 1,
            'converged': delta < tol,
            'theoretical_error': theoretical_error,
            'final_delta': delta
        }
        
        if track_error:
            result['V_history'] = torch.stack(V_history)
        if V_star is not None:
            result['error_history'] = np.array(error_history)
        
        return result
    
    def precision_lower_bound(self, target_error: float) -> float:
        """
        Compute minimum precision (in bits) for value iteration to converge
        to within target_error of V*.
        
        From theorem: p ≥ log₂((R_max + |S|·V_max) / ((1-γ)·ε))
        
        Args:
            target_error: Desired accuracy ε
        
        Returns:
            Minimum number of bits required
        """
        numerator = self.R_max + self.n_states * self.V_max
        denominator = (1 - self.gamma) * target_error
        
        if denominator <= 0:
            return float('inf')
        
        p_min = np.log2(numerator / denominator)
        return p_min
    
    def critical_precision_regime(self) -> Dict[str, float]:
        """
        Determine if current precision is in the critical regime where
        numerical noise Δ_T exceeds contraction strength (1-γ).
        
        Critical condition: Δ_T > (1-γ)·V_max
        When this holds, effective discount γ_eff > 1, causing divergence.
        
        Returns dict with:
        - Delta_T: Intrinsic error
        - contraction_strength: (1-γ)·V_max
        - gamma_eff: Effective discount factor
        - is_critical: Whether in critical regime
        """
        contraction = (1 - self.gamma) * self.V_max
        is_critical = self.Delta_T > contraction
        
        # Effective discount: γ_eff ≈ γ + Δ_T/V_max
        gamma_eff = self.gamma + self.Delta_T / self.V_max if self.V_max > 0 else self.gamma
        
        return {
            'Delta_T': self.Delta_T,
            'contraction_strength': contraction,
            'gamma_eff': gamma_eff,
            'is_critical': is_critical,
            'precision_bits': -np.log2(float(self.eps_mach))
        }


def simulate_precision(value: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Simulate n-bit fixed-point precision by quantization.
    
    Strategy:
    1. Find dynamic range [v_min, v_max]
    2. Quantize to 2^n_bits levels
    3. Dequantize back to float
    
    This simulates the effect of limited precision without actually using
    custom dtypes.
    """
    if n_bits >= 32:
        return value.to(torch.float32)
    
    v_min = torch.min(value).item()
    v_max = torch.max(value).item()
    
    # Avoid division by zero
    if abs(v_max - v_min) < 1e-10:
        return value
    
    # Number of quantization levels
    n_levels = 2 ** n_bits
    
    # Quantize: map [v_min, v_max] to [0, n_levels-1]
    normalized = (value - v_min) / (v_max - v_min)
    quantized = torch.round(normalized * (n_levels - 1))
    
    # Dequantize back
    dequantized = quantized / (n_levels - 1) * (v_max - v_min) + v_min
    
    return dequantized


class LowPrecisionBellman(BellmanOperator):
    """
    Bellman operator with simulated low precision.
    
    Applies precision simulation at each step to study error accumulation.
    """
    
    def __init__(
        self,
        rewards: torch.Tensor,
        transitions: torch.Tensor,
        gamma: float,
        precision_bits: int,
        device: str = 'cpu'
    ):
        # Map bits to dtype for intrinsic error calculation
        if precision_bits >= 32:
            dtype = torch.float32
        elif precision_bits >= 16:
            dtype = torch.float16
        else:
            dtype = torch.float32  # Simulate with quantization
        
        super().__init__(rewards, transitions, gamma, dtype, device)
        
        self.precision_bits = precision_bits
        
        # Override intrinsic error for simulated precision
        if precision_bits < 32:
            # Simulated machine epsilon
            self.eps_mach = 2.0 ** (-precision_bits)
            # Use sqrt(n_states) for realistic error bound
            self.Delta_T = self.eps_mach * (self.R_max + (np.sqrt(self.n_states) + 1) * self.V_max)
            self.morphism.Delta = self.Delta_T
    
    def apply(self, V: torch.Tensor) -> torch.Tensor:
        """Apply Bellman operator with precision simulation."""
        # Standard Bellman update
        TV = super().apply(V)
        
        # Simulate precision
        if self.precision_bits < 32:
            TV = simulate_precision(TV, self.precision_bits)
        
        return TV


class QLearningAgent:
    """
    Q-learning with numerical error tracking.
    
    Updates: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]
    
    Numerical error sources:
    1. TD target: r + γ max_a' Q(s',a')
    2. TD error: δ = target - Q(s,a)
    3. Update: α·δ
    4. Addition: Q(s,a) + α·δ
    """
    
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        gamma: float,
        alpha: float,
        epsilon: float = 0.1,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu'
    ):
        self.n_states = n_states
        self.n_actions = n_actions
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.dtype = dtype
        self.device = device
        
        # Initialize Q-table
        self.Q = torch.zeros((n_states, n_actions), dtype=dtype, device=device)
        
        # Numerical properties
        self.eps_mach = torch.finfo(dtype).eps
        
        # Tracking
        self.step_count = 0
        self.episode_count = 0
    
    def select_action(self, state: int) -> int:
        """Epsilon-greedy action selection."""
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            return torch.argmax(self.Q[state]).item()
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Q-learning update with error tracking.
        
        Returns dict with numerical error estimates.
        """
        # Current Q-value
        Q_sa = self.Q[state, action]
        
        # TD target
        if done:
            target = reward
        else:
            max_next_Q = torch.max(self.Q[next_state]).item()
            target = reward + self.gamma * max_next_Q
        
        # TD error
        td_error = target - Q_sa.item()
        
        # Update
        self.Q[state, action] = Q_sa + self.alpha * td_error
        
        # Numerical error estimate
        # Target error: O(ε_mach · (|r| + γ·Q_max))
        Q_max = torch.max(torch.abs(self.Q)).item()
        target_error = float(self.eps_mach) * (abs(reward) + self.gamma * Q_max)
        
        # Update error: O(ε_mach · α · |δ|)
        update_error = float(self.eps_mach) * self.alpha * abs(td_error)
        
        self.step_count += 1
        
        return {
            'td_error': td_error,
            'target_error': target_error,
            'update_error': update_error,
            'Q_max': Q_max
        }


def check_rl_precision(
    gamma: float,
    R_max: float,
    n_states: int,
    target_error: float
) -> Dict[str, float]:
    """
    Usable artifact: Check minimum precision for stable RL.
    
    Args:
        gamma: Discount factor
        R_max: Maximum reward magnitude
        n_states: Number of states (for error accumulation)
        target_error: Desired accuracy
    
    Returns:
        Dict with:
        - min_bits: Minimum precision in bits
        - V_max: Maximum value scale
        - Delta_T_bound: Upper bound on Bellman intrinsic error
        - safe_bits: Recommended bits with 2-bit safety margin
    """
    V_max = R_max / (1 - gamma) if gamma < 1 else R_max * 1000  # Heuristic for γ→1
    
    # Precision lower bound - use sqrt(n_states) for realistic bound
    numerator = R_max + (np.sqrt(n_states) + 1) * V_max
    denominator = (1 - gamma) * target_error
    
    min_bits = np.log2(numerator / denominator) if denominator > 0 else 64
    
    # Intrinsic error bound
    eps_at_min = 2.0 ** (-min_bits)
    Delta_T_bound = eps_at_min * (R_max + (np.sqrt(n_states) + 1) * V_max)
    
    # Safety margin: add 2 bits
    safe_bits = min_bits + 2
    
    return {
        'min_bits': min_bits,
        'safe_bits': safe_bits,
        'V_max': V_max,
        'Delta_T_bound': Delta_T_bound,
        'gamma': gamma,
        'R_max': R_max,
        'n_states': n_states,
        'target_error': target_error
    }
