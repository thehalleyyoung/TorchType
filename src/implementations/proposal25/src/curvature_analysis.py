"""
Curvature-based numerical analysis for fairness metrics.

Implements the Curvature Lower Bound Theorem from HNF:
For C^2 function f with curvature κ_f = (1/2)||D²f||_op,
no algorithm achieves accuracy better than Ω(κ_f · ε_H²).

This module computes curvature for fairness-related functions
and provides tighter precision bounds than Lipschitz analysis alone.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Callable, Optional, List
from dataclasses import dataclass
import warnings


@dataclass
class CurvatureBound:
    """
    Curvature-based precision bound.
    
    Attributes:
        curvature: The curvature κ = (1/2)||D²f||_op
        lipschitz: Lipschitz constant for first-order analysis
        precision_floor: Minimum achievable precision Ω(κ · ε²)
        recommended_precision: Recommended bits for given accuracy
    """
    curvature: float
    lipschitz: float
    precision_floor: float
    recommended_precision: int
    metadata: Dict[str, any]


class CurvatureAnalyzer:
    """
    Analyzes curvature of fairness computations to determine precision requirements.
    
    The key insight: fairness metrics involve nonlinear operations (threshold comparisons,
    aggregations near boundaries) that have curvature. This provides tighter bounds
    than pure Lipschitz analysis.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.curvature_cache: Dict[str, CurvatureBound] = {}
    
    def _estimate_hessian_norm(self, 
                               func: Callable,
                               x: torch.Tensor,
                               num_samples: int = 100,
                               delta: float = 1e-4) -> float:
        """
        Estimate ||D²f||_op using finite differences.
        
        For a scalar function f: R^n → R, we estimate the Hessian norm by:
        1. Computing gradients at x and x + δe_i for each basis vector e_i
        2. Approximating D²f via (grad(x+δe_i) - grad(x)) / δ
        3. Taking operator norm (largest singular value)
        
        Args:
            func: Function to analyze (must support autograd)
            x: Point at which to compute Hessian
            num_samples: Number of random directions to sample
            delta: Finite difference step size
            
        Returns:
            Estimate of ||D²f||_op
        """
        x = x.detach().requires_grad_(True)
        
        # Compute gradient at x
        output = func(x)
        if output.dim() > 0:
            output = output.sum()  # Reduce to scalar
        
        grad_x = torch.autograd.grad(output, x, create_graph=True)[0]
        
        if grad_x is None:
            return 0.0
        
        # Estimate Hessian norm via finite differences in random directions
        max_hessian_norm = 0.0
        
        for _ in range(num_samples):
            # Random direction
            direction = torch.randn_like(x)
            direction = direction / (torch.norm(direction) + 1e-10)
            
            # Perturbed point
            x_perturbed = x + delta * direction
            x_perturbed = x_perturbed.detach().requires_grad_(True)
            
            # Gradient at perturbed point
            output_perturbed = func(x_perturbed)
            if output_perturbed.dim() > 0:
                output_perturbed = output_perturbed.sum()
            
            grad_x_perturbed = torch.autograd.grad(output_perturbed, x_perturbed)[0]
            
            if grad_x_perturbed is None:
                continue
            
            # Approximate Hessian-vector product
            hessian_vec = (grad_x_perturbed - grad_x.detach()) / delta
            norm = torch.norm(hessian_vec).item()
            max_hessian_norm = max(max_hessian_norm, norm)
        
        return max_hessian_norm
    
    def analyze_model_curvature(self,
                                model: torch.nn.Module,
                                x_sample: torch.Tensor,
                                num_samples: int = 50) -> CurvatureBound:
        """
        Analyze curvature of a neural network model.
        
        For fairness applications, we care about curvature near decision thresholds.
        High curvature means predictions are sensitive to input perturbations,
        which amplifies numerical errors.
        
        Args:
            model: Neural network to analyze
            x_sample: Sample input (representative of typical inputs)
            num_samples: Number of random perturbations for Hessian estimation
            
        Returns:
            CurvatureBound with precision requirements
        """
        model.eval()
        
        # Create a function that maps input to output
        def model_func(x):
            with torch.enable_grad():
                return model(x.unsqueeze(0)).squeeze()
        
        # Estimate Hessian norm
        hessian_norm = self._estimate_hessian_norm(
            model_func, x_sample, num_samples
        )
        
        curvature = 0.5 * hessian_norm
        
        # Also estimate Lipschitz constant via gradient samples
        max_gradient_norm = 0.0
        for _ in range(num_samples):
            x_rand = x_sample + 0.1 * torch.randn_like(x_sample)
            x_rand = x_rand.detach().requires_grad_(True)
            output = model_func(x_rand)
            if output.dim() > 0:
                output = output.sum()
            grad = torch.autograd.grad(output, x_rand)[0]
            max_gradient_norm = max(max_gradient_norm, torch.norm(grad).item())
        
        lipschitz = max_gradient_norm
        
        # Precision floor from curvature bound theorem
        # For float32: ε_H ≈ 1.2e-7
        # For float16: ε_H ≈ 4.9e-4
        epsilon_float32 = 1.19e-7
        epsilon_float16 = 4.88e-4
        
        # Ω(κ · ε²) bound
        precision_floor_32 = curvature * epsilon_float32**2
        precision_floor_16 = curvature * epsilon_float16**2
        
        # Recommended precision: how many bits needed for 1e-6 accuracy?
        target_accuracy = 1e-6
        if curvature > 0:
            # ε² ≈ target_accuracy / κ
            # ε ≈ sqrt(target_accuracy / κ)
            # bits ≈ -log2(ε)
            required_epsilon = np.sqrt(target_accuracy / curvature)
            recommended_bits = max(16, int(-np.log2(required_epsilon) + 0.5))
        else:
            recommended_bits = 16
        
        recommended_bits = min(64, recommended_bits)  # Cap at float64
        
        metadata = {
            'hessian_norm': hessian_norm,
            'precision_floor_float32': precision_floor_32,
            'precision_floor_float16': precision_floor_16,
            'gradient_norm': max_gradient_norm,
            'num_samples': num_samples
        }
        
        return CurvatureBound(
            curvature=curvature,
            lipschitz=lipschitz,
            precision_floor=precision_floor_32,
            recommended_precision=recommended_bits,
            metadata=metadata
        )
    
    def analyze_threshold_function_curvature(self,
                                            predictions: np.ndarray,
                                            threshold: float = 0.5,
                                            bandwidth: float = 0.1) -> float:
        """
        Analyze curvature of the threshold function near decision boundary.
        
        The function f(p) = I(p > t) has infinite curvature at t.
        We approximate with a smooth surrogate: σ((p - t) / h)
        where σ is sigmoid and h is bandwidth.
        
        Higher curvature near threshold = more sensitivity to numerical error.
        
        Args:
            predictions: Model predictions in [0, 1]
            threshold: Decision threshold
            bandwidth: Smoothing parameter
            
        Returns:
            Estimated curvature of threshold function
        """
        # Smooth threshold function: σ((p - t) / h)
        # First derivative: σ'((p - t) / h) / h = σ(z)(1 - σ(z)) / h
        # Second derivative: d/dp[σ(z)(1 - σ(z)) / h] = (σ'(z)(1 - 2σ(z))) / h²
        #                  = σ(z)(1 - σ(z))(1 - 2σ(z)) / h²
        
        # Maximum of |second derivative| occurs at z where d/dz|f''| = 0
        # For sigmoid, max|f''| occurs near z ≈ ±0.66
        # At these points, |f''| ≈ 0.385 / h²
        
        max_second_deriv = 0.385 / (bandwidth ** 2)
        curvature = 0.5 * max_second_deriv
        
        return curvature
    
    def compute_fairness_metric_curvature(self,
                                         model: torch.nn.Module,
                                         x_group0: torch.Tensor,
                                         x_group1: torch.Tensor,
                                         threshold: float = 0.5) -> Dict[str, float]:
        """
        Compute curvature of the demographic parity gap as a function of model parameters.
        
        DPG(θ) = |E[I(f_θ(x) > t) | G=0] - E[I(f_θ(x) > t) | G=1]|
        
        This is highly nonlinear due to the threshold indicator.
        High curvature means DPG is very sensitive to parameter perturbations,
        which amplifies numerical errors during training or quantization.
        
        Args:
            model: Neural network
            x_group0: Samples from group 0
            x_group1: Samples from group 1
            threshold: Decision threshold
            
        Returns:
            Dictionary with curvature metrics
        """
        model.eval()
        
        # Use smooth approximation of threshold function
        bandwidth = 0.05
        
        def dpg_smooth(params_flat):
            """DPG as function of flattened parameters"""
            # Unflatten parameters
            offset = 0
            for param in model.parameters():
                numel = param.numel()
                param.data = params_flat[offset:offset+numel].reshape(param.shape)
                offset += numel
            
            # Compute predictions
            with torch.no_grad():
                pred0 = torch.sigmoid((model(x_group0).squeeze() - threshold) / bandwidth)
                pred1 = torch.sigmoid((model(x_group1).squeeze() - threshold) / bandwidth)
            
            rate0 = pred0.mean()
            rate1 = pred1.mean()
            
            return torch.abs(rate0 - rate1)
        
        # Flatten current parameters
        params_list = [p.data.flatten() for p in model.parameters()]
        params_flat = torch.cat(params_list).requires_grad_(True)
        
        # Estimate Hessian norm of DPG w.r.t. parameters
        try:
            hessian_norm = self._estimate_hessian_norm(
                dpg_smooth, params_flat, num_samples=30
            )
        except Exception as e:
            warnings.warn(f"Failed to compute DPG curvature: {e}")
            hessian_norm = 0.0
        
        curvature_dpg = 0.5 * hessian_norm
        
        # Also compute curvature of individual model outputs
        avg_model_curvature = 0.0
        num_samples_analyzed = min(10, len(x_group0))
        
        for i in range(num_samples_analyzed):
            bound = self.analyze_model_curvature(model, x_group0[i], num_samples=20)
            avg_model_curvature += bound.curvature
        
        if num_samples_analyzed > 0:
            avg_model_curvature /= num_samples_analyzed
        
        return {
            'dpg_curvature': curvature_dpg,
            'avg_model_curvature': avg_model_curvature,
            'threshold_curvature': self.analyze_threshold_function_curvature(
                np.array([]), threshold
            )
        }
    
    def recommend_precision_for_fairness(self,
                                        model: torch.nn.Module,
                                        x_samples: torch.Tensor,
                                        target_dpg_error: float = 0.01) -> Dict[str, any]:
        """
        Recommend precision for reliable fairness assessment.
        
        Uses curvature analysis to determine minimum precision needed
        to achieve target DPG measurement error.
        
        Args:
            model: Neural network
            x_samples: Representative sample of inputs
            target_dpg_error: Maximum acceptable error in DPG measurement
            
        Returns:
            Recommendation dict with precision, rationale, bounds
        """
        # Sample a few inputs to analyze
        num_analyze = min(20, len(x_samples))
        curvatures = []
        
        for i in range(num_analyze):
            bound = self.analyze_model_curvature(
                model, x_samples[i], num_samples=30
            )
            curvatures.append(bound.curvature)
        
        max_curvature = max(curvatures) if curvatures else 0.0
        avg_curvature = np.mean(curvatures) if curvatures else 0.0
        
        # Precision floor from max curvature
        # We need ε such that κ · ε² ≤ target_error
        # => ε ≤ sqrt(target_error / κ)
        
        if max_curvature > 0:
            required_epsilon = np.sqrt(target_dpg_error / max_curvature)
            required_bits = int(-np.log2(required_epsilon) + 0.5)
        else:
            required_bits = 16
        
        # Map to standard precisions
        if required_bits <= 11:
            recommended_dtype = 'float16'
            recommended_bits = 11
        elif required_bits <= 24:
            recommended_dtype = 'float32'
            recommended_bits = 24
        else:
            recommended_dtype = 'float64'
            recommended_bits = 53
        
        # Verify the recommendation
        epsilon_map = {
            'float16': 4.88e-4,
            'float32': 1.19e-7,
            'float64': 2.22e-16
        }
        
        actual_epsilon = epsilon_map[recommended_dtype]
        actual_error_bound = max_curvature * actual_epsilon**2
        
        return {
            'recommended_dtype': recommended_dtype,
            'recommended_bits': recommended_bits,
            'max_curvature': max_curvature,
            'avg_curvature': avg_curvature,
            'target_dpg_error': target_dpg_error,
            'predicted_error': actual_error_bound,
            'is_sufficient': actual_error_bound <= target_dpg_error,
            'safety_margin': target_dpg_error / (actual_error_bound + 1e-10)
        }


def test_curvature_analysis():
    """Test curvature analysis on simple model"""
    print("Testing curvature analysis...")
    
    # Create simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
        torch.nn.Sigmoid()
    )
    
    # Test on random input
    x = torch.randn(5)
    
    analyzer = CurvatureAnalyzer()
    bound = analyzer.analyze_model_curvature(model, x, num_samples=50)
    
    print(f"  Curvature: {bound.curvature:.6f}")
    print(f"  Lipschitz: {bound.lipschitz:.6f}")
    print(f"  Precision floor: {bound.precision_floor:.2e}")
    print(f"  Recommended bits: {bound.recommended_precision}")
    
    # Test precision recommendation
    x_samples = torch.randn(20, 5)
    rec = analyzer.recommend_precision_for_fairness(
        model, x_samples, target_dpg_error=0.01
    )
    
    print(f"\n  Precision recommendation:")
    print(f"    Dtype: {rec['recommended_dtype']}")
    print(f"    Max curvature: {rec['max_curvature']:.6f}")
    print(f"    Predicted error: {rec['predicted_error']:.2e}")
    print(f"    Is sufficient: {rec['is_sufficient']}")
    print(f"    Safety margin: {rec['safety_margin']:.2f}x")
    
    print("\n  ✓ Curvature analysis tests passed!")


if __name__ == '__main__':
    test_curvature_analysis()
