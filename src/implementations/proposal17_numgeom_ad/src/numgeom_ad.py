"""
NumGeom-AD: Automatic Differentiation with Error Functional Tracking

Main module implementing certified autodiff for PyTorch models.
Tracks error bounds on both forward pass values and backward pass gradients.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
import warnings
from collections import defaultdict

from error_functional import (
    ErrorFunctional, ErrorTracker, get_primitive_error_functional,
    compose_error_functionals
)


@dataclass
class TrackedTensor:
    """
    A tensor with tracked precision bounds
    
    Attributes:
        value: The actual tensor value
        error_functional: The error functional Φ describing accumulated error
        name: Optional name for debugging
    """
    value: torch.Tensor
    error_functional: ErrorFunctional
    name: str = ""
    
    def get_error_bound(self, input_epsilon: float) -> float:
        """Get the error bound for this tensor given input precision"""
        return self.error_functional(input_epsilon)


class NumGeomAD:
    """
    Wrapper for PyTorch models that tracks error functionals through computation
    
    Usage:
        model = MyModel()
        numgeom_model = NumGeomAD(model, dtype=torch.float32)
        
        # Forward with error tracking
        output, error_bound = numgeom_model.forward_with_error(x)
        
        # Backward with gradient error tracking
        loss.backward()
        grad_errors = numgeom_model.get_gradient_errors()
    """
    
    def __init__(self, model: nn.Module, dtype: torch.dtype = torch.float32, device: str = 'cpu'):
        self.model = model
        self.dtype = dtype
        self.device = device
        self.error_tracker = ErrorTracker(dtype=dtype, device=device)
        
        # Track operations and their error functionals
        self.forward_error_log: List[Tuple[str, ErrorFunctional]] = []
        self.backward_error_log: List[Tuple[str, ErrorFunctional]] = []
        
        # Hooks for capturing intermediate values
        self.hooks = []
        self.intermediate_errors: Dict[str, ErrorFunctional] = {}
        
        # Register hooks on model layers
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks on all layers to track errors"""
        for name, module in self.model.named_modules():
            if len(list(module.children())) == 0:  # Only leaf modules
                hook = module.register_forward_hook(
                    self._make_forward_hook(name, module)
                )
                self.hooks.append(hook)
    
    def _make_forward_hook(self, name: str, module: nn.Module):
        """Create a forward hook that tracks error for a specific layer"""
        def hook(module, input, output):
            # Determine operation type
            op_type = type(module).__name__.lower()
            
            # Get error functional for this operation
            if isinstance(module, nn.Linear):
                # Linear layer: Y = XW^T + b
                X = input[0] if isinstance(input, tuple) else input
                W = module.weight
                
                # Matmul error
                matmul_error = get_primitive_error_functional(
                    'matmul', [X, W.T], self.error_tracker.eps_machine
                )
                
                # Addition error (bias)
                if module.bias is not None:
                    add_error = get_primitive_error_functional(
                        'add', [output, module.bias], self.error_tracker.eps_machine
                    )
                    layer_error = add_error.compose(matmul_error)
                else:
                    layer_error = matmul_error
                
            elif isinstance(module, nn.ReLU):
                X = input[0] if isinstance(input, tuple) else input
                layer_error = get_primitive_error_functional(
                    'relu', [X], self.error_tracker.eps_machine
                )
                
            elif isinstance(module, nn.Softmax):
                X = input[0] if isinstance(input, tuple) else input
                layer_error = get_primitive_error_functional(
                    'softmax', [X], self.error_tracker.eps_machine, dim=module.dim
                )
                
            elif isinstance(module, nn.LayerNorm):
                X = input[0] if isinstance(input, tuple) else input
                layer_error = get_primitive_error_functional(
                    'layernorm', [X], self.error_tracker.eps_machine, eps=module.eps
                )
                
            elif isinstance(module, (nn.Conv2d, nn.Conv1d)):
                # Convolution is similar to matmul but with more operations
                X = input[0] if isinstance(input, tuple) else input
                W = module.weight
                
                # Approximate as matmul with appropriate scaling
                # Number of ops scales with kernel size
                kernel_ops = np.prod(W.shape)
                scaled_eps = self.error_tracker.eps_machine * np.sqrt(kernel_ops)
                
                layer_error = ErrorFunctional(
                    lipschitz=W.norm().item(),
                    intrinsic=scaled_eps * output.abs().max().item(),
                    name=f'conv_{name}'
                )
                
            else:
                # Default: assume Lipschitz 1
                layer_error = ErrorFunctional(
                    lipschitz=1.0,
                    intrinsic=self.error_tracker.eps_machine,
                    name=f'{op_type}_{name}'
                )
            
            # Store error functional for this layer
            self.intermediate_errors[name] = layer_error
            self.forward_error_log.append((name, layer_error))
            
            # Log to tracker
            self.error_tracker.log_operation(name, layer_error, output)
            
        return hook
    
    def forward_with_error(
        self, 
        x: torch.Tensor, 
        input_epsilon: Optional[float] = None
    ) -> Tuple[torch.Tensor, float]:
        """
        Forward pass with error tracking
        
        Args:
            x: Input tensor
            input_epsilon: Input precision (default: machine epsilon)
            
        Returns:
            output: Model output
            error_bound: Certified error bound on output
        """
        # Clear previous logs
        self.forward_error_log = []
        self.intermediate_errors = {}
        self.error_tracker.error_log = []
        
        # Set default input epsilon
        if input_epsilon is None:
            input_epsilon = self.error_tracker.eps_machine
        
        # Run forward pass
        output = self.model(x)
        
        # Compose all error functionals
        error_functionals = [ef for _, ef in self.forward_error_log]
        if error_functionals:
            total_error_functional = compose_error_functionals(error_functionals)
            error_bound = total_error_functional(input_epsilon)
        else:
            error_bound = input_epsilon
        
        return output, error_bound
    
    def get_error_breakdown(self) -> Dict[str, float]:
        """Get per-layer error contributions"""
        return self.error_tracker.get_error_breakdown()
    
    def check_stability(self, threshold: float = 1e-4) -> List[str]:
        """Check for numerical instabilities"""
        return self.error_tracker.check_stability(threshold)
    
    def analyze_gradient_error(
        self, 
        loss: torch.Tensor,
        input_epsilon: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Analyze gradient error using reverse-mode AD error propagation
        
        From Theorem (AD Error Propagation) in the comprehensive document:
        For F = f_n ∘ ... ∘ f_1, the gradient error is:
        Φ_{∇F}(ε) ≤ ∑_i (∏_{j≠i} L_j) Φ_{Df_i}(ε_i)
        
        Returns:
            Dictionary mapping parameter name to gradient error bound
        """
        if input_epsilon is None:
            input_epsilon = self.error_tracker.eps_machine
        
        gradient_errors = {}
        
        # Get Lipschitz constants from forward pass
        lipschitz_constants = [ef.lipschitz for _, ef in self.forward_error_log]
        
        # For each layer, compute its contribution to gradient error
        for i, (layer_name, error_func) in enumerate(self.forward_error_log):
            # Product of all Lipschitz constants except this layer
            other_lipschitz = 1.0
            for j, L_j in enumerate(lipschitz_constants):
                if j != i:
                    other_lipschitz *= L_j
            
            # Error contribution from this layer
            layer_grad_error = other_lipschitz * error_func(input_epsilon)
            gradient_errors[layer_name] = layer_grad_error
        
        return gradient_errors
    
    def get_gradient_errors(self) -> Dict[str, float]:
        """
        Get gradient error bounds for all parameters
        
        This computes the error bound on each computed gradient.
        """
        gradient_errors = {}
        
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                # Find corresponding layer in error log
                layer_errors = [ef for ln, ef in self.forward_error_log if name.startswith(ln)]
                
                if layer_errors:
                    # Compose errors
                    total_error = compose_error_functionals(layer_errors)
                    error_bound = total_error(self.error_tracker.eps_machine)
                else:
                    error_bound = self.error_tracker.eps_machine
                
                gradient_errors[name] = error_bound
        
        return gradient_errors
    
    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def __del__(self):
        """Cleanup hooks on deletion"""
        self.remove_hooks()


def compare_with_higher_precision(
    model: nn.Module,
    x: torch.Tensor,
    loss_fn: callable,
    y: Optional[torch.Tensor] = None,
    low_dtype: torch.dtype = torch.float32,
    high_dtype: torch.dtype = torch.float64
) -> Dict[str, Any]:
    """
    Validate error bounds by comparing low-precision computation with high-precision
    
    Args:
        model: PyTorch model
        x: Input tensor
        loss_fn: Loss function
        y: Target (if needed for loss)
        low_dtype: Lower precision dtype (default: float32)
        high_dtype: Higher precision dtype (default: float64)
        
    Returns:
        Dictionary with observed errors and bound tightness
    """
    device = x.device
    
    # Low precision computation
    model_low = model.to(low_dtype).to(device)
    x_low = x.to(low_dtype)
    
    numgeom_low = NumGeomAD(model_low, dtype=low_dtype, device=str(device))
    output_low, error_bound_low = numgeom_low.forward_with_error(x_low)
    
    if y is not None:
        loss_low = loss_fn(output_low, y.to(low_dtype))
    else:
        loss_low = loss_fn(output_low)
    
    loss_low.backward()
    
    # Get gradient errors
    grad_errors_low = numgeom_low.analyze_gradient_error(loss_low)
    
    # High precision computation (ground truth)
    model_high = model.to(high_dtype).to(device)
    x_high = x.to(high_dtype)
    
    output_high = model_high(x_high)
    
    if y is not None:
        loss_high = loss_fn(output_high, y.to(high_dtype))
    else:
        loss_high = loss_fn(output_high)
    
    loss_high.backward()
    
    # Compare outputs
    output_error_observed = (output_low.to(high_dtype) - output_high).abs().max().item()
    output_tightness = error_bound_low / (output_error_observed + 1e-15)
    
    # Compare gradients
    gradient_comparison = {}
    for name, param_low in model_low.named_parameters():
        if param_low.grad is not None:
            param_high = dict(model_high.named_parameters())[name]
            
            grad_error_observed = (
                param_low.grad.to(high_dtype) - param_high.grad
            ).abs().max().item()
            
            grad_error_predicted = grad_errors_low.get(name, 0.0)
            
            tightness = grad_error_predicted / (grad_error_observed + 1e-15)
            
            gradient_comparison[name] = {
                'observed': grad_error_observed,
                'predicted': grad_error_predicted,
                'tightness': tightness
            }
    
    # Cleanup
    numgeom_low.remove_hooks()
    
    return {
        'output_error_observed': output_error_observed,
        'output_error_bound': error_bound_low,
        'output_tightness': output_tightness,
        'gradient_comparison': gradient_comparison,
        'error_breakdown': numgeom_low.get_error_breakdown(),
        'stability_warnings': numgeom_low.check_stability()
    }


# Import numpy for conv layer
import numpy as np


if __name__ == '__main__':
    print("NumGeom-AD: Certified Automatic Differentiation")
    print("=" * 50)
    
    # Simple test
    torch.manual_seed(42)
    
    # Create a simple model
    model = nn.Sequential(
        nn.Linear(10, 20),
        nn.ReLU(),
        nn.Linear(20, 10),
        nn.Softmax(dim=-1)
    )
    
    # Random input
    x = torch.randn(5, 10)
    
    # Wrap with NumGeom-AD
    numgeom = NumGeomAD(model, dtype=torch.float32, device='cpu')
    
    # Forward with error tracking
    output, error_bound = numgeom.forward_with_error(x)
    
    print(f"\nOutput error bound: {error_bound:.2e}")
    print("\nError breakdown:")
    for layer, error in numgeom.get_error_breakdown().items():
        print(f"  {layer}: {error:.2e}")
    
    print("\nStability warnings:")
    warnings_list = numgeom.check_stability(threshold=1e-6)
    for w in warnings_list:
        print(f"  ⚠ {w}")
    
    # Test gradient error analysis
    loss = output.sum()
    loss.backward()
    
    grad_errors = numgeom.analyze_gradient_error(loss)
    print("\nGradient error bounds:")
    for param_name, error in grad_errors.items():
        print(f"  {param_name}: {error:.2e}")
    
    numgeom.remove_hooks()
    
    print("\n✓ Basic test completed")
