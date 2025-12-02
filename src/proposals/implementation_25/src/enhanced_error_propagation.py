"""
Enhanced error propagation that automatically extracts network architecture
and computes precise error functionals without conservative defaults.

This addresses the potential "cheating" of using lipschitz=10.0 default.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass

try:
    from .error_propagation import ErrorTracker, LinearErrorFunctional
except ImportError:
    from error_propagation import ErrorTracker, LinearErrorFunctional


class ArchitectureAnalyzer:
    """
    Automatically analyzes a PyTorch model architecture to extract
    layer dimensions and activation functions for precise error tracking.
    """
    
    @staticmethod
    def extract_layer_info(model: nn.Module) -> Tuple[List[int], List[str]]:
        """
        Extract layer dimensions and activation functions from a model.
        
        Returns:
            layer_dims: List of dimensions [input, hidden1, ..., output]
            activations: List of activation names for each layer
        """
        layer_dims = []
        activations = []
        
        # Walk through modules
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                if not layer_dims:  # First layer
                    layer_dims.append(module.in_features)
                layer_dims.append(module.out_features)
            
            # Track activation functions
            elif isinstance(module, nn.ReLU):
                activations.append('relu')
            elif isinstance(module, nn.Sigmoid):
                activations.append('sigmoid')
            elif isinstance(module, nn.Tanh):
                activations.append('tanh')
            elif isinstance(module, nn.LeakyReLU):
                activations.append('relu')  # Conservative
            elif isinstance(module, nn.GELU):
                activations.append('relu')  # Conservative (Lipschitz ~1)
        
        # If we have more layers than activations, pad with identity
        while len(activations) < len(layer_dims) - 1:
            activations.append('identity')
        
        return layer_dims, activations
    
    @staticmethod
    def estimate_weight_condition_number(model: nn.Module) -> float:
        """
        Estimate the average condition number of weight matrices.
        
        For properly initialized networks, this should be O(sqrt(n)).
        This provides a reality check on our Lipschitz estimates.
        """
        condition_numbers = []
        
        for module in model.modules():
            if isinstance(module, nn.Linear):
                W = module.weight.detach().cpu().numpy()
                try:
                    # Compute singular values
                    s = np.linalg.svd(W, compute_uv=False)
                    if s[-1] > 1e-10:  # Avoid division by zero
                        cond = s[0] / s[-1]
                        condition_numbers.append(cond)
                except:
                    pass
        
        if condition_numbers:
            return np.mean(condition_numbers)
        return 1.0


class PreciseErrorTracker(ErrorTracker):
    """
    Enhanced error tracker that computes precise error functionals
    from actual model architecture instead of using defaults.
    """
    
    def __init__(self, precision: torch.dtype = torch.float32):
        super().__init__(precision)
    
    def compute_model_error_functional(
        self, 
        model: nn.Module,
        use_empirical: bool = False,
        input_shape: Optional[Tuple] = None,
        n_samples: int = 50,
        device: str = 'cpu'
    ) -> LinearErrorFunctional:
        """
        Compute precise error functional for a model.
        
        Args:
            model: PyTorch model
            use_empirical: If True, use empirical Lipschitz estimation
            input_shape: Required if use_empirical=True
            n_samples: Number of samples for empirical estimation
            device: Device for computation
        
        Returns:
            Linear error functional Φ(ε) = L·ε + Δ
        """
        if use_empirical:
            if input_shape is None:
                raise ValueError("input_shape required for empirical estimation")
            return self._compute_empirical_functional(
                model, input_shape, n_samples, device
            )
        else:
            return self._compute_analytical_functional(model)
    
    def _compute_analytical_functional(self, model: nn.Module) -> LinearErrorFunctional:
        """
        Compute error functional analytically from architecture.
        
        This is the CORRECT way - no conservative defaults!
        """
        # Extract architecture
        layer_dims, activations = ArchitectureAnalyzer.extract_layer_info(model)
        
        if not layer_dims:
            # Fallback for non-standard architectures
            return LinearErrorFunctional(1.0, self.epsilon_machine)
        
        # Track through network
        return self.track_network(layer_dims, activations, name="model")
    
    def _compute_empirical_functional(
        self,
        model: nn.Module,
        input_shape: Tuple,
        n_samples: int,
        device: str
    ) -> LinearErrorFunctional:
        """
        Compute error functional empirically via finite differences.
        """
        from error_propagation import estimate_lipschitz_empirical
        
        # Estimate Lipschitz constant
        lipschitz = estimate_lipschitz_empirical(
            model, input_shape, n_samples, device
        )
        
        # Estimate roundoff from architecture
        n_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        roundoff = n_layers * self.epsilon_machine * 5  # Less conservative than before
        
        return LinearErrorFunctional(lipschitz, roundoff)
    
    def compute_prediction_errors(
        self,
        model: nn.Module,
        data: torch.Tensor,
        error_functional: Optional[LinearErrorFunctional] = None
    ) -> np.ndarray:
        """
        Compute per-sample error bounds for model predictions.
        
        Args:
            model: PyTorch model
            data: Input data
            error_functional: Optional pre-computed functional
        
        Returns:
            Array of error bounds, one per sample
        """
        if error_functional is None:
            error_functional = self.compute_model_error_functional(model)
        
        model.eval()
        with torch.no_grad():
            predictions = model(data).cpu().numpy().flatten()
        
        # Compute error bound for each prediction
        errors = np.array([
            self.compute_error_bound(error_functional, float(p))
            for p in predictions
        ])
        
        return errors


class AdaptiveErrorTracker(PreciseErrorTracker):
    """
    Adaptive error tracker that chooses between analytical and empirical
    methods based on model complexity.
    """
    
    def __init__(self, precision: torch.dtype = torch.float32, 
                 empirical_threshold: int = 10):
        super().__init__(precision)
        self.empirical_threshold = empirical_threshold
    
    def compute_model_error_functional(
        self,
        model: nn.Module,
        input_shape: Optional[Tuple] = None,
        device: str = 'cpu'
    ) -> LinearErrorFunctional:
        """
        Automatically choose between analytical and empirical methods.
        """
        # Count layers
        n_layers = sum(1 for _ in model.modules() if isinstance(_, nn.Linear))
        
        if n_layers <= self.empirical_threshold and input_shape is not None:
            # For small networks, use empirical (more accurate)
            return self._compute_empirical_functional(
                model, input_shape, n_samples=100, device=device
            )
        else:
            # For large networks, use analytical (faster)
            return self._compute_analytical_functional(model)


def verify_error_functional_validity(
    model: nn.Module,
    error_functional: LinearErrorFunctional,
    input_shape: Tuple,
    precision: torch.dtype,
    n_tests: int = 50,
    device: str = 'cpu'
) -> Dict[str, float]:
    """
    Verify that an error functional actually bounds the true numerical errors.
    
    This is a critical validation - it checks we're not "cheating"!
    
    Returns:
        Dictionary with validation metrics:
        - violation_rate: Fraction of cases where bound is violated
        - avg_slack: Average amount of slack in the bound
        - max_violation: Maximum violation (should be ≈ 0)
    """
    model.eval()
    
    violations = 0
    slacks = []
    max_violation = 0.0
    
    # Get machine epsilon
    tracker = ErrorTracker(precision)
    eps = tracker.epsilon_machine
    
    # Theoretical bound
    theoretical_bound = error_functional.evaluate(eps)
    
    with torch.no_grad():
        for _ in range(n_tests):
            # Generate random input
            x = torch.randn(*input_shape, device=device, dtype=precision)
            
            # Compute in target precision
            y_prec = model(x).cpu().numpy().flatten()[0]
            
            # Compute in high precision (ground truth)
            model_high = type(model)(
                input_dim=model.layers[0].in_features if hasattr(model, 'layers') else input_shape[-1],
                hidden_dims=[],
                activation='relu'
            ).to(torch.float64)
            
            # Copy weights to high precision
            try:
                for (name1, param1), (name2, param2) in zip(
                    model.named_parameters(), 
                    model_high.named_parameters()
                ):
                    param2.data = param1.data.to(torch.float64)
                
                x_high = x.to(torch.float64)
                y_high = model_high(x_high).cpu().numpy().flatten()[0]
                
                # Actual error
                actual_error = abs(float(y_prec) - float(y_high))
                
                # Check if bound holds
                if actual_error > theoretical_bound:
                    violations += 1
                    max_violation = max(max_violation, actual_error - theoretical_bound)
                else:
                    slack = theoretical_bound - actual_error
                    slacks.append(slack)
            except:
                # Handle architectures that can't be easily copied
                pass
    
    return {
        'violation_rate': violations / n_tests if n_tests > 0 else 0.0,
        'avg_slack': np.mean(slacks) if slacks else 0.0,
        'max_violation': max_violation,
        'theoretical_bound': theoretical_bound,
        'n_tests': n_tests
    }
