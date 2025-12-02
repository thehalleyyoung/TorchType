"""
Fairness metrics with certified numerical bounds.

Implements demographic parity, equalized odds, and calibration metrics
with rigorous error propagation from finite precision arithmetic.
"""

import torch
import numpy as np
from typing import Tuple, Dict, List, Optional
from dataclasses import dataclass

try:
    from .error_propagation import ErrorTracker, LinearErrorFunctional
except ImportError:
    from error_propagation import ErrorTracker, LinearErrorFunctional


@dataclass
class FairnessResult:
    """
    Result of a fairness metric computation with certified bounds.
    
    Attributes:
        metric_value: The computed fairness metric value
        error_bound: Upper bound on numerical error
        is_reliable: Whether the metric is numerically reliable
        reliability_score: Ratio of metric_value to error_bound
        near_threshold_fraction: Fraction of samples near decision boundary
        metadata: Additional information
    """
    metric_value: float
    error_bound: float
    is_reliable: bool
    reliability_score: float
    near_threshold_fraction: Dict[str, float]
    metadata: Dict[str, any]


class FairnessMetrics:
    """
    Compute fairness metrics with numerical error tracking.
    """
    
    @staticmethod
    def demographic_parity_gap(predictions: np.ndarray,
                              groups: np.ndarray,
                              threshold: float = 0.5,
                              group_0: int = 0,
                              group_1: int = 1) -> float:
        """
        Compute demographic parity gap: |P(ŷ=1|G=0) - P(ŷ=1|G=1)|
        """
        mask_0 = groups == group_0
        mask_1 = groups == group_1
        
        if mask_0.sum() == 0 or mask_1.sum() == 0:
            return 0.0
        
        positive_rate_0 = (predictions[mask_0] > threshold).mean()
        positive_rate_1 = (predictions[mask_1] > threshold).mean()
        
        return abs(positive_rate_0 - positive_rate_1)
    
    @staticmethod
    def equalized_odds_gap(predictions: np.ndarray,
                          labels: np.ndarray,
                          groups: np.ndarray,
                          threshold: float = 0.5,
                          group_0: int = 0,
                          group_1: int = 1,
                          label_value: int = 1) -> float:
        """
        Compute equalized odds gap: |P(ŷ=1|Y=y,G=0) - P(ŷ=1|Y=y,G=1)|
        """
        mask_0 = (groups == group_0) & (labels == label_value)
        mask_1 = (groups == group_1) & (labels == label_value)
        
        if mask_0.sum() == 0 or mask_1.sum() == 0:
            return 0.0
        
        tpr_0 = (predictions[mask_0] > threshold).mean()
        tpr_1 = (predictions[mask_1] > threshold).mean()
        
        return abs(tpr_0 - tpr_1)
    
    @staticmethod
    def calibration_error(predictions: np.ndarray,
                         labels: np.ndarray,
                         n_bins: int = 10) -> Tuple[float, np.ndarray, np.ndarray]:
        """
        Compute expected calibration error.
        
        Returns:
            ece: Expected calibration error
            bin_accuracies: Accuracy in each bin
            bin_confidences: Average confidence in each bin
        """
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_accuracies = np.zeros(n_bins)
        bin_confidences = np.zeros(n_bins)
        bin_counts = np.zeros(n_bins)
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            if i == n_bins - 1:  # Include right boundary for last bin
                in_bin = (predictions >= bin_lower) & (predictions <= bin_upper)
            
            if in_bin.sum() > 0:
                bin_accuracies[i] = labels[in_bin].mean()
                bin_confidences[i] = predictions[in_bin].mean()
                bin_counts[i] = in_bin.sum()
        
        # Expected calibration error
        ece = 0.0
        total_samples = len(predictions)
        for i in range(n_bins):
            if bin_counts[i] > 0:
                ece += (bin_counts[i] / total_samples) * abs(bin_accuracies[i] - bin_confidences[i])
        
        return ece, bin_accuracies, bin_confidences


class CertifiedFairnessEvaluator:
    """
    Evaluate fairness metrics with certified numerical bounds.
    """
    
    def __init__(self, 
                 error_tracker: ErrorTracker,
                 reliability_threshold: float = 2.0):
        """
        Args:
            error_tracker: ErrorTracker for the model
            reliability_threshold: Minimum ratio of metric/error for reliability
        """
        self.error_tracker = error_tracker
        self.reliability_threshold = reliability_threshold
    
    def evaluate_demographic_parity(self,
                                   model: torch.nn.Module,
                                   data: torch.Tensor,
                                   groups: np.ndarray,
                                   threshold: float = 0.5,
                                   group_0: int = 0,
                                   group_1: int = 1,
                                   model_error_functional: Optional[LinearErrorFunctional] = None
                                   ) -> FairnessResult:
        """
        Evaluate demographic parity with certified bounds.
        """
        model.eval()
        
        with torch.no_grad():
            predictions = model(data).cpu().numpy().flatten()
        
        # Compute demographic parity gap
        dpg = FairnessMetrics.demographic_parity_gap(
            predictions, groups, threshold, group_0, group_1
        )
        
        # Compute error bound
        if model_error_functional is None:
            # Use a conservative default
            model_error_functional = LinearErrorFunctional(
                lipschitz=10.0,
                roundoff=self.error_tracker.epsilon_machine
            )
        
        # Identify near-threshold samples
        prediction_errors = np.array([
            self.error_tracker.compute_error_bound(model_error_functional, p)
            for p in predictions
        ])
        
        near_threshold_mask = np.abs(predictions - threshold) < prediction_errors
        
        # Compute near-threshold fractions
        mask_0 = groups == group_0
        mask_1 = groups == group_1
        
        p_near_0 = near_threshold_mask[mask_0].mean() if mask_0.sum() > 0 else 0.0
        p_near_1 = near_threshold_mask[mask_1].mean() if mask_1.sum() > 0 else 0.0
        
        # Error bound from Theorem (Fairness Metric Error)
        error_bound = p_near_0 + p_near_1
        
        # Reliability assessment
        reliability_score = dpg / error_bound if error_bound > 0 else float('inf')
        is_reliable = reliability_score >= self.reliability_threshold
        
        return FairnessResult(
            metric_value=dpg,
            error_bound=error_bound,
            is_reliable=is_reliable,
            reliability_score=reliability_score,
            near_threshold_fraction={
                f'group_{group_0}': p_near_0,
                f'group_{group_1}': p_near_1,
                'overall': near_threshold_mask.mean()
            },
            metadata={
                'threshold': threshold,
                'n_samples': len(predictions),
                'n_near_threshold': near_threshold_mask.sum(),
                'predictions_mean': predictions.mean(),
                'predictions_std': predictions.std()
            }
        )
    
    def evaluate_equalized_odds(self,
                               model: torch.nn.Module,
                               data: torch.Tensor,
                               labels: np.ndarray,
                               groups: np.ndarray,
                               threshold: float = 0.5,
                               group_0: int = 0,
                               group_1: int = 1,
                               model_error_functional: Optional[LinearErrorFunctional] = None
                               ) -> FairnessResult:
        """
        Evaluate equalized odds with certified bounds.
        """
        model.eval()
        
        with torch.no_grad():
            predictions = model(data).cpu().numpy().flatten()
        
        # Compute equalized odds gap (for positive class)
        eog = FairnessMetrics.equalized_odds_gap(
            predictions, labels, groups, threshold, group_0, group_1, label_value=1
        )
        
        # Compute error bound
        if model_error_functional is None:
            model_error_functional = LinearErrorFunctional(
                lipschitz=10.0,
                roundoff=self.error_tracker.epsilon_machine
            )
        
        # Identify near-threshold samples (only for positive labels)
        prediction_errors = np.array([
            self.error_tracker.compute_error_bound(model_error_functional, p)
            for p in predictions
        ])
        
        near_threshold_mask = np.abs(predictions - threshold) < prediction_errors
        
        # Focus on positive labels
        positive_mask = labels == 1
        mask_0 = (groups == group_0) & positive_mask
        mask_1 = (groups == group_1) & positive_mask
        
        p_near_0 = near_threshold_mask[mask_0].mean() if mask_0.sum() > 0 else 0.0
        p_near_1 = near_threshold_mask[mask_1].mean() if mask_1.sum() > 0 else 0.0
        
        # Error bound
        error_bound = p_near_0 + p_near_1
        
        # Reliability assessment
        reliability_score = eog / error_bound if error_bound > 0 else float('inf')
        is_reliable = reliability_score >= self.reliability_threshold
        
        return FairnessResult(
            metric_value=eog,
            error_bound=error_bound,
            is_reliable=is_reliable,
            reliability_score=reliability_score,
            near_threshold_fraction={
                f'group_{group_0}': p_near_0,
                f'group_{group_1}': p_near_1,
                'overall': near_threshold_mask[positive_mask].mean()
            },
            metadata={
                'threshold': threshold,
                'n_samples': len(predictions),
                'n_positive': positive_mask.sum(),
                'n_near_threshold': near_threshold_mask[positive_mask].sum()
            }
        )
    
    def evaluate_calibration(self,
                           model: torch.nn.Module,
                           data: torch.Tensor,
                           labels: np.ndarray,
                           n_bins: int = 10,
                           model_error_functional: Optional[LinearErrorFunctional] = None
                           ) -> Dict:
        """
        Evaluate calibration with bin-wise uncertainty.
        """
        model.eval()
        
        with torch.no_grad():
            predictions = model(data).cpu().numpy().flatten()
        
        # Compute calibration error
        ece, bin_acc, bin_conf = FairnessMetrics.calibration_error(
            predictions, labels, n_bins
        )
        
        # Compute error bound
        if model_error_functional is None:
            model_error_functional = LinearErrorFunctional(
                lipschitz=10.0,
                roundoff=self.error_tracker.epsilon_machine
            )
        
        # Compute uncertainty for each bin
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_uncertainties = []
        
        prediction_errors = np.array([
            self.error_tracker.compute_error_bound(model_error_functional, p)
            for p in predictions
        ])
        
        for i in range(n_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            
            # Samples near bin boundaries
            near_lower = np.abs(predictions - bin_lower) < prediction_errors
            near_upper = np.abs(predictions - bin_upper) < prediction_errors
            
            in_bin = (predictions >= bin_lower) & (predictions < bin_upper)
            if i == n_bins - 1:
                in_bin = (predictions >= bin_lower) & (predictions <= bin_upper)
            
            n_uncertain = (near_lower | near_upper).sum()
            n_in_bin = in_bin.sum()
            
            uncertainty = n_uncertain / max(n_in_bin, 1)
            bin_uncertainties.append(uncertainty)
        
        return {
            'ece': ece,
            'bin_accuracies': bin_acc,
            'bin_confidences': bin_conf,
            'bin_uncertainties': np.array(bin_uncertainties),
            'reliable_bins': [u < 0.1 for u in bin_uncertainties]
        }


class ThresholdStabilityAnalyzer:
    """
    Analyze stability of fairness metrics across threshold choices.
    """
    
    def __init__(self, evaluator: CertifiedFairnessEvaluator):
        self.evaluator = evaluator
    
    def analyze_threshold_stability(self,
                                   model: torch.nn.Module,
                                   data: torch.Tensor,
                                   groups: np.ndarray,
                                   threshold_range: Tuple[float, float] = (0.1, 0.9),
                                   n_points: int = 17,
                                   model_error_functional: Optional[LinearErrorFunctional] = None
                                   ) -> Dict:
        """
        Analyze demographic parity gap across different thresholds.
        
        Returns:
            Dictionary with thresholds, DPG values, error bounds, and stability info
        """
        thresholds = np.linspace(threshold_range[0], threshold_range[1], n_points)
        
        dpg_values = []
        error_bounds = []
        reliability_scores = []
        is_reliable = []
        
        for threshold in thresholds:
            result = self.evaluator.evaluate_demographic_parity(
                model, data, groups, threshold,
                model_error_functional=model_error_functional
            )
            
            dpg_values.append(result.metric_value)
            error_bounds.append(result.error_bound)
            reliability_scores.append(result.reliability_score)
            is_reliable.append(result.is_reliable)
        
        dpg_values = np.array(dpg_values)
        error_bounds = np.array(error_bounds)
        
        # Identify stable regions (low variation)
        stability_tolerance = 0.05
        stable_regions = []
        
        for i in range(len(thresholds)):
            # Check if nearby thresholds give similar DPG
            if i > 0 and i < len(thresholds) - 1:
                variation = max(
                    abs(dpg_values[i] - dpg_values[i-1]),
                    abs(dpg_values[i] - dpg_values[i+1])
                )
                is_stable = variation < stability_tolerance
            else:
                is_stable = False
            
            stable_regions.append(is_stable)
        
        return {
            'thresholds': thresholds,
            'dpg_values': dpg_values,
            'error_bounds': error_bounds,
            'reliability_scores': np.array(reliability_scores),
            'is_reliable': np.array(is_reliable),
            'stable_regions': np.array(stable_regions)
        }
    
    def find_stable_thresholds(self,
                             model: torch.nn.Module,
                             data: torch.Tensor,
                             groups: np.ndarray,
                             model_error_functional: Optional[LinearErrorFunctional] = None
                             ) -> List[Tuple[float, float]]:
        """
        Find threshold ranges where fairness metrics are numerically stable.
        
        Returns:
            List of (threshold_min, threshold_max) tuples for stable regions
        """
        analysis = self.analyze_threshold_stability(
            model, data, groups,
            model_error_functional=model_error_functional
        )
        
        stable_regions = []
        in_region = False
        region_start = None
        
        for i, (threshold, is_stable) in enumerate(
            zip(analysis['thresholds'], analysis['stable_regions'])
        ):
            if is_stable and not in_region:
                region_start = threshold
                in_region = True
            elif not is_stable and in_region:
                stable_regions.append((region_start, analysis['thresholds'][i-1]))
                in_region = False
        
        # Close last region if needed
        if in_region:
            stable_regions.append((region_start, analysis['thresholds'][-1]))
        
        return stable_regions


class AdversarialSignFlipGenerator:
    """
    Generate adversarial examples showing sign flips under numerical perturbation.
    
    This demonstrates the theoretical possibility of sign flips when predictions
    are highly concentrated near the decision threshold.
    """
    
    @staticmethod
    def create_near_threshold_scenario(n_samples: int = 500,
                                       threshold: float = 0.5,
                                       concentration: float = 0.01,
                                       group_imbalance: float = 0.005,
                                       seed: int = 42) -> tuple:
        """
        Create a scenario where predictions are highly concentrated near threshold.
        
        Args:
            n_samples: Number of samples
            threshold: Decision threshold
            concentration: How tightly predictions cluster around threshold
            group_imbalance: Small DPG to create borderline case
            seed: Random seed
            
        Returns:
            (predictions, groups, true_dpg_signed, numerical_uncertainty)
        """
        np.random.seed(seed)
        
        # Half samples in each group
        n_group_0 = n_samples // 2
        n_group_1 = n_samples - n_group_0
        
        # Generate predictions tightly clustered around threshold
        # Group 0: slightly above threshold
        preds_0 = np.random.normal(
            threshold + group_imbalance, 
            concentration, 
            n_group_0
        )
        
        # Group 1: slightly below threshold
        preds_1 = np.random.normal(
            threshold - group_imbalance,
            concentration,
            n_group_1
        )
        
        predictions = np.concatenate([preds_0, preds_1])
        groups = np.array([0] * n_group_0 + [1] * n_group_1)
        
        # True DPG (signed)
        pos_rate_0 = (preds_0 > threshold).mean()
        pos_rate_1 = (preds_1 > threshold).mean()
        true_dpg_signed = pos_rate_0 - pos_rate_1
        
        # Numerical uncertainty: fraction near threshold
        near_threshold = np.abs(predictions - threshold) < 3 * concentration
        numerical_uncertainty = near_threshold.mean()
        
        return predictions, groups, true_dpg_signed, numerical_uncertainty
    
    @staticmethod
    def simulate_precision_effects(predictions: np.ndarray,
                                   groups: np.ndarray,
                                   threshold: float = 0.5,
                                   epsilon_16: float = 0.0009765625,  # 2^-10 for float16
                                   epsilon_32: float = 1.1920929e-07,  # 2^-23 for float32
                                   epsilon_64: float = 2.220446e-16   # 2^-52 for float64
                                   ) -> dict:
        """
        Simulate how different precisions affect fairness metrics.
        
        Adds random perturbations scaled by machine epsilon to simulate
        the accumulated roundoff errors in real computation.
        """
        results = {}
        
        for precision_name, epsilon in [
            ('float64', epsilon_64),
            ('float32', epsilon_32),
            ('float16', epsilon_16)
        ]:
            # Simulate accumulated roundoff
            # In a deep network, errors accumulate roughly as sqrt(n_ops) * epsilon * Lipschitz
            # For a 3-layer network with ~1000 operations and Lipschitz constant ~10:
            # error ~ sqrt(1000) * 10 * epsilon ~ 300 * epsilon
            # For float16, this is ~0.3, which is realistic for deep networks
            n_operations = 1000  # Typical for 3-layer MLP
            lipschitz_bound = 10  # Typical for ReLU networks
            perturbation_scale = np.sqrt(n_operations) * lipschitz_bound * epsilon
            
            # Add Gaussian noise to model roundoff accumulation
            perturbed_preds = predictions + np.random.normal(
                0, 
                perturbation_scale,
                len(predictions)
            )
            
            # Compute DPG with perturbation
            mask_0 = groups == 0
            mask_1 = groups == 1
            
            pos_rate_0 = (perturbed_preds[mask_0] > threshold).mean()
            pos_rate_1 = (perturbed_preds[mask_1] > threshold).mean()
            
            dpg_signed = pos_rate_0 - pos_rate_1
            
            results[precision_name] = {
                'dpg_signed': dpg_signed,
                'perturbation_scale': perturbation_scale,
                'predictions_mean': perturbed_preds.mean(),
                'predictions_std': perturbed_preds.std()
            }
        
        # Check for sign flip
        dpg_values = [results[p]['dpg_signed'] for p in ['float64', 'float32', 'float16']]
        signs = [np.sign(v) for v in dpg_values if abs(v) > 1e-6]
        has_sign_flip = len(set(signs)) > 1 if signs else False
        
        results['has_sign_flip'] = has_sign_flip
        results['dpg_range'] = (min(dpg_values), max(dpg_values))
        
        return results
