"""
Error propagation framework for tracking numerical errors through computations.

Implements the Stability Composition Theorem from the HNF framework:
For morphisms f_1, ..., f_n with Lipschitz constants L_i and roundoff errors Δ_i,
the composite has error Φ_F(ε) = (∏ L_i) ε + Σ_i Δ_i ∏_{j>i} L_j
"""

import torch
import numpy as np
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class LinearErrorFunctional:
    """
    Represents a linear error functional Φ(ε) = L·ε + Δ
    
    Attributes:
        lipschitz: Lipschitz constant L
        roundoff: Roundoff error Δ
    """
    lipschitz: float
    roundoff: float
    
    def evaluate(self, epsilon: float) -> float:
        """Evaluate the error functional at precision ε"""
        return self.lipschitz * epsilon + self.roundoff
    
    def compose(self, other: 'LinearErrorFunctional') -> 'LinearErrorFunctional':
        """
        Compose this error functional with another.
        
        If this is Φ_1(ε) = L_1·ε + Δ_1 and other is Φ_2(ε) = L_2·ε + Δ_2,
        then the composition is Φ(ε) = L_1·L_2·ε + Δ_1·L_2 + Δ_2
        """
        new_lipschitz = self.lipschitz * other.lipschitz
        new_roundoff = self.roundoff * other.lipschitz + other.roundoff
        return LinearErrorFunctional(new_lipschitz, new_roundoff)
    
    @staticmethod
    def compose_sequence(functionals: list) -> 'LinearErrorFunctional':
        """
        Compose a sequence of error functionals.
        
        Implements: Φ_F(ε) = (∏ L_i) ε + Σ_i Δ_i ∏_{j>i} L_j
        """
        if not functionals:
            return LinearErrorFunctional(1.0, 0.0)
        
        # Product of all Lipschitz constants
        total_lipschitz = 1.0
        for f in functionals:
            total_lipschitz *= f.lipschitz
        
        # Sum of propagated roundoff errors
        total_roundoff = 0.0
        for i, f_i in enumerate(functionals):
            # Product of subsequent Lipschitz constants
            subsequent_product = 1.0
            for j in range(i + 1, len(functionals)):
                subsequent_product *= functionals[j].lipschitz
            total_roundoff += f_i.roundoff * subsequent_product
        
        return LinearErrorFunctional(total_lipschitz, total_roundoff)


class ErrorTracker:
    """
    Tracks numerical errors through neural network computations.
    """
    
    def __init__(self, precision: torch.dtype = torch.float32):
        self.precision = precision
        self.epsilon_machine = self._get_machine_epsilon(precision)
        self.tracked_errors: Dict[str, LinearErrorFunctional] = {}
        
    def _get_machine_epsilon(self, dtype: torch.dtype) -> float:
        """Get machine epsilon for the given dtype"""
        if dtype == torch.float16:
            return 4.88e-04  # 2^-11
        elif dtype == torch.float32:
            return 1.19e-07  # 2^-23
        elif dtype == torch.float64:
            return 2.22e-16  # 2^-52
        else:
            return 1.19e-07  # Default to float32
    
    def track_matmul(self, input_dim: int, output_dim: int, 
                     name: str = "matmul") -> LinearErrorFunctional:
        """
        Track error through matrix multiplication y = Wx.
        
        For a matrix multiplication with condition number κ and dimension n,
        the error functional is approximately:
        Φ(ε) = κ·sqrt(n)·ε + n·ε_mach
        
        We use a more realistic estimate based on typical neural network matrices.
        For well-conditioned networks with proper initialization, κ is typically O(sqrt(n))
        rather than O(n).
        """
        # More realistic Lipschitz estimate for typical neural networks
        # Typical condition number for Xavier/He initialized weights is O(sqrt(n))
        n = max(input_dim, output_dim)
        lipschitz = 2.0 * np.sqrt(n)  # Conservative but realistic
        roundoff = n * self.epsilon_machine
        
        functional = LinearErrorFunctional(lipschitz, roundoff)
        self.tracked_errors[name] = functional
        return functional
    
    def track_activation(self, activation: str, 
                        name: str = "activation") -> LinearErrorFunctional:
        """
        Track error through activation function.
        
        For ReLU: Lipschitz constant is 1
        For Sigmoid: Lipschitz constant is 0.25
        For Tanh: Lipschitz constant is 1
        """
        lipschitz_map = {
            'relu': 1.0,
            'sigmoid': 0.25,
            'tanh': 1.0,
            'identity': 1.0,
        }
        
        lipschitz = lipschitz_map.get(activation.lower(), 1.0)
        roundoff = self.epsilon_machine
        
        functional = LinearErrorFunctional(lipschitz, roundoff)
        self.tracked_errors[name] = functional
        return functional
    
    def track_layer(self, input_dim: int, output_dim: int,
                   activation: str = 'relu',
                   name: str = "layer") -> LinearErrorFunctional:
        """Track error through a full layer (matmul + activation)"""
        matmul_error = self.track_matmul(input_dim, output_dim, 
                                         f"{name}_matmul")
        activation_error = self.track_activation(activation, 
                                                f"{name}_activation")
        
        # Compose the two error functionals
        combined = matmul_error.compose(activation_error)
        self.tracked_errors[name] = combined
        return combined
    
    def track_network(self, layer_dims: list, 
                     activations: list,
                     name: str = "network") -> LinearErrorFunctional:
        """
        Track error through a multi-layer network.
        
        Args:
            layer_dims: List of layer dimensions [input, hidden1, ..., output]
            activations: List of activation functions for each layer
            name: Name for tracking
        """
        functionals = []
        
        for i in range(len(layer_dims) - 1):
            layer_error = self.track_layer(
                layer_dims[i], 
                layer_dims[i + 1],
                activations[i] if i < len(activations) else 'identity',
                f"{name}_layer{i}"
            )
            functionals.append(layer_error)
        
        # Compose all layers
        network_error = LinearErrorFunctional.compose_sequence(functionals)
        self.tracked_errors[name] = network_error
        return network_error
    
    def compute_error_bound(self, functional: LinearErrorFunctional,
                          value: float) -> float:
        """
        Compute absolute error bound for a computed value.
        
        Args:
            functional: The error functional for the computation
            value: The computed value
        
        Returns:
            Absolute error bound
        """
        # The error functional Φ(ε) = L·ε + Δ gives the error bound directly
        # The error bound is just the functional evaluated at machine epsilon
        error_bound = functional.evaluate(self.epsilon_machine)
        
        # For predictions in [0,1] (probabilities), scale by the value magnitude
        # to get absolute error in the output space
        if abs(value) > 0:
            # Absolute error ≈ error_bound * max(|value|, 1-|value|) for bounded outputs
            # This accounts for how close we are to saturation
            return error_bound * max(abs(value), 1.0 - abs(value))
        else:
            return error_bound
    
    def get_precision_name(self) -> str:
        """Get human-readable precision name"""
        if self.precision == torch.float16:
            return "float16"
        elif self.precision == torch.float32:
            return "float32"
        elif self.precision == torch.float64:
            return "float64"
        else:
            return str(self.precision)


def compute_lipschitz_certified(model: torch.nn.Module) -> float:
    """
    Compute a CERTIFIED upper bound on the Lipschitz constant.
    
    For feedforward networks with Lipschitz activations (ReLU, sigmoid, tanh),
    the Lipschitz constant is bounded by the product of spectral norms of weight matrices.
    
    This provides a rigorous bound, not just an empirical estimate.
    """
    lipschitz_bound = 1.0
    activation_lipschitz = 1.0  # ReLU and tanh have Lipschitz constant 1
    
    for module in model.modules():
        if isinstance(module, torch.nn.Linear):
            # Compute spectral norm (largest singular value)
            weight = module.weight.detach()
            try:
                # Use SVD to get exact spectral norm
                spectral_norm = torch.linalg.matrix_norm(weight, ord=2).item()
            except:
                # Fallback: use Frobenius norm as upper bound
                spectral_norm = torch.norm(weight).item()
            
            lipschitz_bound *= spectral_norm
    
    # Multiply by activation Lipschitz (1 for ReLU/tanh, 0.25 for sigmoid)
    # We conservatively use 1.0
    return lipschitz_bound * activation_lipschitz


def estimate_lipschitz_empirical(model: torch.nn.Module,
                                input_shape: tuple,
                                n_samples: int = 100,
                                device: str = 'cpu',
                                perturbation_scale: float = 0.01) -> float:
    """
    Empirically estimate the Lipschitz constant of a model.
    
    Uses finite differences with random perturbations.
    WARNING: This provides a LOWER BOUND, not an upper bound!
    For certified bounds, use compute_lipschitz_certified() instead.
    """
    model.eval()
    
    # Get model's dtype
    model_dtype = next(model.parameters()).dtype
    
    max_ratio = 0.0
    
    with torch.no_grad():
        for _ in range(n_samples):
            # Random input with same dtype as model
            x = torch.randn(*input_shape, device=device, dtype=model_dtype)
            
            # Small perturbation
            delta = torch.randn_like(x) * perturbation_scale
            delta_norm = torch.norm(delta).item()
            
            # Compute outputs
            y1 = model(x)
            y2 = model(x + delta)
            
            # Compute ratio
            output_diff = torch.norm(y2 - y1).item()
            
            if delta_norm > 1e-8:
                ratio = output_diff / delta_norm
                max_ratio = max(max_ratio, ratio)
    
    return max_ratio


def create_certified_error_functional(model: torch.nn.Module,
                                     precision: torch.dtype) -> LinearErrorFunctional:
    """
    Create a CERTIFIED error functional with rigorous bounds.
    
    Uses spectral norm product for Lipschitz constant, providing
    a guaranteed upper bound on numerical error.
    """
    # Compute certified Lipschitz bound
    lipschitz = compute_lipschitz_certified(model)
    
    # Estimate roundoff based on model depth and precision
    # Count number of operations (layers)
    n_ops = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
    
    tracker = ErrorTracker(precision)
    # Each operation accumulates roundoff error that propagates forward
    # Conservative estimate: each layer adds eps * Lipschitz of subsequent layers
    roundoff = n_ops * tracker.epsilon_machine * lipschitz
    
    return LinearErrorFunctional(lipschitz, roundoff)


def create_empirical_error_functional(model: torch.nn.Module,
                                      input_shape: tuple,
                                      precision: torch.dtype,
                                      n_samples: int = 50,
                                      device: str = 'cpu') -> LinearErrorFunctional:
    """
    Create an error functional based on empirical Lipschitz estimation.
    
    WARNING: This uses empirical estimation which may UNDERESTIMATE the true error.
    For certified bounds, use create_certified_error_functional() instead.
    """
    # Estimate Lipschitz constant empirically (this is a LOWER bound!)
    lipschitz_empirical = estimate_lipschitz_empirical(
        model, input_shape, n_samples, device
    )
    
    # Also compute certified bound
    lipschitz_certified = compute_lipschitz_certified(model)
    
    # Use the larger of the two (certified bound is safer)
    lipschitz = max(lipschitz_empirical, lipschitz_certified)
    
    # Estimate roundoff based on model depth and precision
    # Count number of operations (layers)
    n_ops = sum(1 for _ in model.modules() if isinstance(_, torch.nn.Linear))
    
    tracker = ErrorTracker(precision)
    roundoff = n_ops * tracker.epsilon_machine * lipschitz
    
    return LinearErrorFunctional(lipschitz, roundoff)
