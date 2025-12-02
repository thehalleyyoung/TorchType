"""
Error Functional Theory for Numerical Geometry

This module implements the error functional algebra from the HNF comprehensive document.
Error functionals Φ(ε) = L·ε + Δ compose algebraically through the Stability Composition Theorem.
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, List
from dataclasses import dataclass
import warnings


@dataclass
class ErrorFunctional:
    """
    Represents a linear error functional Φ(ε) = L·ε + Δ
    
    Attributes:
        lipschitz: The Lipschitz constant L (sensitivity to input error)
        intrinsic: The intrinsic roundoff error Δ (hardware-dependent)
        name: Optional name for debugging
    """
    lipschitz: float  # L in Φ(ε) = L·ε + Δ
    intrinsic: float  # Δ in Φ(ε) = L·ε + Δ
    name: str = ""
    
    def __call__(self, epsilon: float) -> float:
        """Evaluate the error functional at input precision ε"""
        return self.lipschitz * epsilon + self.intrinsic
    
    def compose(self, other: 'ErrorFunctional') -> 'ErrorFunctional':
        """
        Compose two error functionals: (self ∘ other)(ε)
        
        From Stability Composition Theorem:
        If Φ_g(ε) = L_g·ε + Δ_g and Φ_f(ε) = L_f·ε + Δ_f, then
        Φ_{g∘f}(ε) = L_g·L_f·ε + L_g·Δ_f + Δ_g
        """
        return ErrorFunctional(
            lipschitz=self.lipschitz * other.lipschitz,
            intrinsic=self.lipschitz * other.intrinsic + self.intrinsic,
            name=f"({self.name}∘{other.name})" if self.name and other.name else ""
        )
    
    def add(self, other: 'ErrorFunctional') -> 'ErrorFunctional':
        """
        Add two error functionals (for parallel paths)
        Φ + Ψ has Lipschitz L_Φ + L_Ψ and intrinsic max(Δ_Φ, Δ_Ψ)
        """
        return ErrorFunctional(
            lipschitz=self.lipschitz + other.lipschitz,
            intrinsic=max(self.intrinsic, other.intrinsic),  # Shared final roundoff
            name=f"({self.name}+{other.name})" if self.name and other.name else ""
        )
    
    def scale(self, constant: float) -> 'ErrorFunctional':
        """Scale error functional by a constant"""
        return ErrorFunctional(
            lipschitz=constant * self.lipschitz,
            intrinsic=constant * self.intrinsic,
            name=f"{constant}*{self.name}" if self.name else ""
        )
    
    @staticmethod
    def identity() -> 'ErrorFunctional':
        """Identity error functional: Φ(ε) = ε"""
        return ErrorFunctional(lipschitz=1.0, intrinsic=0.0, name="id")
    
    @staticmethod
    def zero() -> 'ErrorFunctional':
        """Zero error functional: Φ(ε) = 0"""
        return ErrorFunctional(lipschitz=0.0, intrinsic=0.0, name="0")


class ErrorTracker:
    """
    Tracks error functionals through computation graphs
    """
    
    def __init__(self, dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        self.dtype = dtype
        self.device = device
        self.eps_machine = self._get_machine_epsilon(dtype)
        self.error_log: List[Dict] = []
        
    def _get_machine_epsilon(self, dtype: torch.dtype) -> float:
        """Get machine epsilon for given dtype"""
        if dtype == torch.float64:
            return 1.1e-16
        elif dtype == torch.float32:
            return 5.96e-8
        elif dtype == torch.float16:
            return 4.88e-4
        else:
            return 5.96e-8  # Default to float32
    
    def log_operation(self, op_name: str, error_func: ErrorFunctional, 
                     value: Optional[torch.Tensor] = None):
        """Log an operation for later analysis"""
        entry = {
            'operation': op_name,
            'lipschitz': error_func.lipschitz,
            'intrinsic': error_func.intrinsic,
            'error_at_eps_machine': error_func(self.eps_machine)
        }
        if value is not None:
            entry['value_magnitude'] = value.abs().max().item()
        self.error_log.append(entry)
    
    def get_error_breakdown(self) -> Dict[str, float]:
        """Get breakdown of error contributions"""
        breakdown = {}
        for entry in self.error_log:
            op = entry['operation']
            error = entry['error_at_eps_machine']
            breakdown[op] = breakdown.get(op, 0) + error
        return breakdown
    
    def check_stability(self, threshold: float = 1e-4) -> List[str]:
        """Check for numerical instabilities"""
        warnings_list = []
        for entry in self.error_log:
            error = entry['error_at_eps_machine']
            if error > threshold:
                warnings_list.append(
                    f"{entry['operation']} has error bound {error:.2e} > threshold {threshold:.2e}"
                )
        return warnings_list


def get_primitive_error_functional(
    op_name: str,
    inputs: List[torch.Tensor],
    eps_machine: float,
    **kwargs
) -> ErrorFunctional:
    """
    Get error functional for primitive operations
    
    Implements error functional derivations from Proposal 17:
    - Addition: Φ(ε) = ε + ε_mach·|x+y|
    - Multiplication: Φ(ε) = (|y|+|x|)·ε + ε_mach·|xy|
    - Division: Φ(ε) = (1 + |x|/|y|)·ε/|y| + ε_mach·|x/y|
    - Exponential: Φ(ε) = exp(x)·ε + ε_mach·exp(x)
    etc.
    """
    
    if op_name == 'add':
        # f(x, y) = x + y
        # Lipschitz = 1 for both inputs
        # Intrinsic = ε_mach·|x+y|
        x, y = inputs[0], inputs[1]
        result = x + y
        return ErrorFunctional(
            lipschitz=1.0,
            intrinsic=eps_machine * result.abs().max().item(),
            name='add'
        )
    
    elif op_name == 'sub':
        # f(x, y) = x - y (cancellation when x ≈ y)
        x, y = inputs[0], inputs[1]
        result = x - y
        # Worse intrinsic error when result is small but inputs are large
        relative_cancellation = (x.abs().max() + y.abs().max()) / (result.abs().max() + 1e-10)
        return ErrorFunctional(
            lipschitz=1.0,
            intrinsic=eps_machine * result.abs().max().item() * relative_cancellation.item(),
            name='sub'
        )
    
    elif op_name == 'mul':
        # f(x, y) = x · y
        # Lipschitz w.r.t. x is |y|, w.r.t. y is |x|
        x, y = inputs[0], inputs[1]
        result = x * y
        x_max = x.abs().max().item()
        y_max = y.abs().max().item()
        return ErrorFunctional(
            lipschitz=x_max + y_max,  # Conservative: max of both
            intrinsic=eps_machine * result.abs().max().item(),
            name='mul'
        )
    
    elif op_name == 'div':
        # f(x, y) = x / y
        # WARNING: Δ → ∞ as y → 0
        x, y = inputs[0], inputs[1]
        result = x / y
        y_min = y.abs().min().item()
        y_max = y.abs().max().item()
        x_max = x.abs().max().item()
        
        if y_min < 1e-6:
            warnings.warn(f"Division near singularity: min|y| = {y_min:.2e}")
        
        # Lipschitz = (1 + |x|/|y|) / |y|
        lipschitz = (1.0 + x_max / (y_min + 1e-10)) / (y_min + 1e-10)
        
        return ErrorFunctional(
            lipschitz=lipschitz,
            intrinsic=eps_machine * result.abs().max().item(),
            name='div'
        )
    
    elif op_name == 'exp':
        # f(x) = exp(x)
        # Lipschitz = exp(x), WARNING: grows exponentially
        x = inputs[0]
        result = torch.exp(x)
        exp_max = result.abs().max().item()
        
        if x.max() > 10:
            warnings.warn(f"Exponential with large input: max(x) = {x.max().item():.2f}")
        
        return ErrorFunctional(
            lipschitz=exp_max,
            intrinsic=eps_machine * exp_max,
            name='exp'
        )
    
    elif op_name == 'log':
        # f(x) = log(x)
        # Lipschitz = 1/min(x)
        x = inputs[0]
        x_min = x.abs().min().item()
        
        if x_min < 1e-6:
            warnings.warn(f"Log near singularity: min(x) = {x_min:.2e}")
        
        result = torch.log(x)
        return ErrorFunctional(
            lipschitz=1.0 / (x_min + 1e-10),
            intrinsic=eps_machine * result.abs().max().item(),
            name='log'
        )
    
    elif op_name == 'relu':
        # f(x) = max(0, x)
        # Piecewise linear: Lipschitz = 1
        x = inputs[0]
        return ErrorFunctional(
            lipschitz=1.0,
            intrinsic=eps_machine,  # Minimal roundoff
            name='relu'
        )
    
    elif op_name == 'matmul':
        # f(X, Y) = X @ Y
        # For X ∈ R^{m×n}, Y ∈ R^{n×p}, Lipschitz w.r.t. X is ||Y||, w.r.t. Y is ||X||
        X, Y = inputs[0], inputs[1]
        result = X @ Y
        
        # Spectral norm approximation (Frobenius norm / sqrt(min dimension))
        X_norm = X.norm().item()
        Y_norm = Y.norm().item()
        
        # Number of operations contributes to roundoff
        n = X.shape[-1]  # Inner dimension
        
        return ErrorFunctional(
            lipschitz=X_norm + Y_norm,
            intrinsic=eps_machine * n * result.abs().max().item(),
            name='matmul'
        )
    
    elif op_name == 'softmax':
        # Detailed analysis from Proposal 17
        x = inputs[0]
        dim = kwargs.get('dim', -1)
        
        # Softmax with max subtraction for stability
        x_max = x.max(dim=dim, keepdim=True)[0]
        x_shifted = x - x_max
        exp_x = torch.exp(x_shifted)
        sum_exp = exp_x.sum(dim=dim, keepdim=True)
        
        # Lipschitz constant of softmax Jacobian is at most 1
        # But intrinsic error depends on conditioning
        max_logit_diff = (x.max() - x.min()).item()
        
        if max_logit_diff > 20:
            warnings.warn(f"Softmax with large logit differences: {max_logit_diff:.2f}")
        
        # Intrinsic error grows with number of elements and poor conditioning
        n = x.shape[dim]
        conditioning_factor = np.exp(min(max_logit_diff, 50))  # Saturate at 50
        
        return ErrorFunctional(
            lipschitz=1.0,
            intrinsic=eps_machine * n * conditioning_factor,
            name='softmax'
        )
    
    elif op_name == 'layernorm':
        # LayerNorm has division by std, can be ill-conditioned
        x = inputs[0]
        eps = kwargs.get('eps', 1e-5)
        
        # Compute std
        std = x.std(dim=-1, keepdim=True)
        std_min = std.min().item()
        
        if std_min < 1e-3:
            warnings.warn(f"LayerNorm with small variance: min(std) = {std_min:.2e}")
        
        # Lipschitz dominated by 1/std
        lipschitz = 1.0 / (std_min + eps)
        
        return ErrorFunctional(
            lipschitz=lipschitz,
            intrinsic=eps_machine * lipschitz,
            name='layernorm'
        )
    
    else:
        # Default: assume Lipschitz 1
        return ErrorFunctional(
            lipschitz=1.0,
            intrinsic=eps_machine,
            name=op_name
        )


def compose_error_functionals(functionals: List[ErrorFunctional]) -> ErrorFunctional:
    """
    Compose a sequence of error functionals using Stability Composition Theorem
    
    For f = f_n ∘ ... ∘ f_1 with error functionals Φ_i(ε) = L_i·ε + Δ_i:
    Φ_f(ε) = (∏L_i)·ε + ∑_i Δ_i·(∏_{j>i} L_j)
    """
    if not functionals:
        return ErrorFunctional.identity()
    
    if len(functionals) == 1:
        return functionals[0]
    
    # Start with first functional
    result = functionals[0]
    
    # Compose with each subsequent functional
    for func in functionals[1:]:
        result = func.compose(result)
    
    return result


def verify_composition_theorem(n_tests: int = 100, max_depth: int = 10):
    """
    Verify the Stability Composition Theorem empirically
    
    Tests that composing error functionals matches the formula:
    Φ_f(ε) = (∏L_i)·ε + ∑_i Δ_i·(∏_{j>i} L_j)
    """
    np.random.seed(42)
    
    for _ in range(n_tests):
        depth = np.random.randint(2, max_depth + 1)
        
        # Generate random error functionals
        lipschitz_constants = np.random.uniform(0.5, 2.0, depth)
        intrinsic_errors = np.random.uniform(0, 1e-6, depth)
        
        functionals = [
            ErrorFunctional(L, Delta, f"f{i}")
            for i, (L, Delta) in enumerate(zip(lipschitz_constants, intrinsic_errors))
        ]
        
        # Compose using our implementation
        composed = compose_error_functionals(functionals)
        
        # Compute expected values from theorem
        expected_lipschitz = np.prod(lipschitz_constants)
        expected_intrinsic = 0.0
        for i in range(depth):
            # Δ_i * ∏_{j>i} L_j
            product = intrinsic_errors[i] * np.prod(lipschitz_constants[i+1:])
            expected_intrinsic += product
        
        # Verify
        assert np.isclose(composed.lipschitz, expected_lipschitz, rtol=1e-10), \
            f"Lipschitz mismatch: {composed.lipschitz} vs {expected_lipschitz}"
        assert np.isclose(composed.intrinsic, expected_intrinsic, rtol=1e-10), \
            f"Intrinsic mismatch: {composed.intrinsic} vs {expected_intrinsic}"
    
    print(f"✓ Stability Composition Theorem verified on {n_tests} random compositions")


if __name__ == '__main__':
    # Run verification
    verify_composition_theorem()
    
    # Example usage
    print("\nExample: Error propagation through a 3-layer network")
    
    # Layer 1: Linear (Lipschitz 2.0)
    phi1 = ErrorFunctional(lipschitz=2.0, intrinsic=1e-7, name="layer1")
    
    # Layer 2: ReLU (Lipschitz 1.0)
    phi2 = ErrorFunctional(lipschitz=1.0, intrinsic=1e-9, name="relu")
    
    # Layer 3: Linear (Lipschitz 1.5)
    phi3 = ErrorFunctional(lipschitz=1.5, intrinsic=1e-7, name="layer2")
    
    # Compose
    network = compose_error_functionals([phi3, phi2, phi1])
    
    print(f"Network Lipschitz constant: {network.lipschitz}")
    print(f"Network intrinsic error: {network.intrinsic:.2e}")
    print(f"Total error at ε=1e-7: {network(1e-7):.2e}")
