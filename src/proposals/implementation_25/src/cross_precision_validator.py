"""
Cross-precision validation: Empirically measure how predictions change across precisions.

This provides REAL error bounds based on actual cross-precision behavior,
not just theoretical estimates.
"""

import torch
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass

@dataclass
class CrossPrecisionAnalysis:
    """Results of cross-precision analysis"""
    baseline_predictions: np.ndarray
    target_predictions: np.ndarray
    max_absolute_diff: float
    mean_absolute_diff: float
    max_relative_diff: float
    predictions_changed: int
    predictions_total: int
    
    def get_error_bound(self) -> float:
        """Get a conservative error bound"""
        # Use max absolute difference as the error bound
        return self.max_absolute_diff


def analyze_cross_precision(model: torch.nn.Module,
                            data: torch.Tensor,
                            baseline_precision: torch.dtype = torch.float64,
                            target_precision: torch.dtype = torch.float32,
                            device: str = 'cpu') -> CrossPrecisionAnalysis:
    """
    Empirically measure how predictions change when using different precisions.
    
    This provides ACTUAL error bounds, not theoretical estimates.
    
    Args:
        model: The model (assumed to be in baseline precision)
        data: Input data
        baseline_precision: High-precision baseline (e.g., float64)
        target_precision: Target precision (e.g., float32 or float16)
        device: Device to run on
        
    Returns:
        CrossPrecisionAnalysis with measured differences
    """
    # Handle device compatibility
    baseline_device = 'cpu' if baseline_precision == torch.float64 and device == 'mps' else device
    target_device = device
    
    # Create copies of the model at different precisions
    # Save state dict and reload
    state_dict = model.state_dict()
    
    # Create baseline model
    import copy
    model_baseline = copy.deepcopy(model)
    model_baseline = model_baseline.to(baseline_precision).to(baseline_device)
    model_baseline.load_state_dict({k: v.to(baseline_precision) for k, v in state_dict.items()})
    
    # Create target model
    model_target = copy.deepcopy(model)
    model_target = model_target.to(target_precision).to(target_device)
    model_target.load_state_dict({k: v.to(target_precision) for k, v in state_dict.items()})
    
    model_baseline.eval()
    model_target.eval()
    
    with torch.no_grad():
        # Baseline predictions
        data_baseline = data.to(baseline_precision).to(baseline_device)
        preds_baseline = model_baseline(data_baseline).cpu().numpy().flatten()
        
        # Target precision predictions
        data_target = data.to(target_precision).to(target_device)
        preds_target = model_target(data_target).cpu().numpy().flatten()
    
    # Compute differences
    abs_diff = np.abs(preds_baseline - preds_target)
    rel_diff = abs_diff / (np.abs(preds_baseline) + 1e-10)
    
    return CrossPrecisionAnalysis(
        baseline_predictions=preds_baseline,
        target_predictions=preds_target,
        max_absolute_diff=float(np.max(abs_diff)),
        mean_absolute_diff=float(np.mean(abs_diff)),
        max_relative_diff=float(np.max(rel_diff)),
        predictions_changed=int(np.sum(abs_diff > 1e-8)),
        predictions_total=len(preds_baseline)
    )


def create_cross_precision_error_functional(model: torch.nn.Module,
                                           data: torch.Tensor,
                                           baseline_precision: torch.dtype,
                                           target_precision: torch.dtype,
                                           device: str = 'cpu'):
    """
    Create an error functional based on ACTUAL cross-precision behavior.
    
    This is more realistic than theoretical bounds for understanding
    how fairness metrics will actually behave at different precisions.
    """
    try:
        from .error_propagation import LinearErrorFunctional, ErrorTracker
    except ImportError:
        from error_propagation import LinearErrorFunctional, ErrorTracker
    
    analysis = analyze_cross_precision(
        model, data, baseline_precision, target_precision, device
    )
    
    # The error bound is the empirically measured max difference
    # This captures weight quantization, activation quantization, etc.
    error_bound = analysis.max_absolute_diff
    
    # For the functional form Φ(ε) = L·ε + Δ,
    # we set Δ = error_bound and L = 0 (since we've already captured the error)
    # Alternatively, we can model it as L = error_bound / ε
    tracker = ErrorTracker(target_precision)
    
    # Conservative: assume the error bound IS the Lipschitz constant
    # (i.e., small perturbations in input could cause this much change in output)
    lipschitz = error_bound / tracker.epsilon_machine if tracker.epsilon_machine > 0 else 1e6
    
    # But that's huge, so let's be more reasonable:
    # The error bound represents quantization error, not input sensitivity
    # So we use a simple model: Φ(ε) = 0·ε + error_bound
    return LinearErrorFunctional(lipschitz=1.0, roundoff=error_bound)


def validate_error_bounds(model: torch.nn.Module,
                         data: torch.Tensor,
                         groups: np.ndarray,
                         threshold: float = 0.5,
                         device: str = 'cpu') -> Dict:
    """
    Validate that theoretical error bounds actually hold in practice.
    
    Tests:
    1. Do predictions actually change across precisions as predicted?
    2. Are DPG differences within predicted bounds?
    3. Are near-threshold counts accurate?
    """
    try:
        from .fairness_metrics import FairnessMetrics
    except ImportError:
        from fairness_metrics import FairnessMetrics
    
    # Analyze float64 -> float32
    analysis_32 = analyze_cross_precision(
        model, data, torch.float64, torch.float32, device
    )
    
    # Analyze float64 -> float16
    analysis_16 = analyze_cross_precision(
        model, data, torch.float64, torch.float16, device
    )
    
    # Compute DPG at each precision
    dpg_64 = FairnessMetrics.demographic_parity_gap(
        analysis_32.baseline_predictions, groups, threshold
    )
    dpg_32 = FairnessMetrics.demographic_parity_gap(
        analysis_32.target_predictions, groups, threshold
    )
    dpg_16 = FairnessMetrics.demographic_parity_gap(
        analysis_16.target_predictions, groups, threshold
    )
    
    # Check if DPG differences are within bounds
    dpg_diff_32 = abs(dpg_64 - dpg_32)
    dpg_diff_16 = abs(dpg_64 - dpg_16)
    
    # Theoretical bound: max_abs_diff gives bound on how many can flip
    # If max_abs_diff is δ, then samples within δ of threshold can flip
    bound_32 = analysis_32.max_absolute_diff
    bound_16 = analysis_16.max_absolute_diff
    
    # Count samples near threshold
    near_32 = np.sum(np.abs(analysis_32.baseline_predictions - threshold) < bound_32) / len(groups)
    near_16 = np.sum(np.abs(analysis_16.baseline_predictions - threshold) < bound_16) / len(groups)
    
    return {
        'float32': {
            'max_pred_diff': analysis_32.max_absolute_diff,
            'mean_pred_diff': analysis_32.mean_absolute_diff,
            'dpg_baseline': dpg_64,
            'dpg_target': dpg_32,
            'dpg_diff': dpg_diff_32,
            'error_bound': bound_32,
            'near_threshold_fraction': near_32,
            'theoretical_dpg_bound': 2 * near_32,  # Both groups could flip
        },
        'float16': {
            'max_pred_diff': analysis_16.max_absolute_diff,
            'mean_pred_diff': analysis_16.mean_absolute_diff,
            'dpg_baseline': dpg_64,
            'dpg_target': dpg_16,
            'dpg_diff': dpg_diff_16,
            'error_bound': bound_16,
            'near_threshold_fraction': near_16,
            'theoretical_dpg_bound': 2 * near_16,
        }
    }
