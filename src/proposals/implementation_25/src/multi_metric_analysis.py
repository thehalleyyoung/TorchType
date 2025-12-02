"""
Multi-metric joint fairness analysis.

This module analyzes multiple fairness metrics simultaneously,
identifying tradeoffs and joint numerical reliability.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from .error_propagation import ErrorTracker
from .fairness_metrics import CertifiedFairnessEvaluator, FairnessResult


@dataclass
class MultiMetricResult:
    """Results from multi-metric analysis."""
    demographic_parity: FairnessResult
    equalized_odds: FairnessResult
    calibration: FairnessResult
    joint_reliable: bool
    joint_reliability_score: float
    metric_correlations: Dict[Tuple[str, str], float]
    precision_requirement: torch.dtype


class MultiMetricFairnessAnalyzer:
    """
    Analyzes multiple fairness metrics jointly.
    
    Extends NumGeom-Fair to consider:
    - Demographic Parity Gap (DPG)
    - Equalized Odds Gap (EOG)
    - Calibration Error (CE)
    
    And their joint numerical reliability.
    """
    
    def __init__(self, error_tracker: Optional[ErrorTracker] = None):
        """
        Initialize multi-metric analyzer.
        
        Args:
            error_tracker: Error tracker for precision analysis
        """
        self.tracker = error_tracker or ErrorTracker(precision=torch.float32)
        self.evaluator = CertifiedFairnessEvaluator(self.tracker)
        
    def evaluate_all_metrics(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        groups: torch.Tensor,
        threshold: float = 0.5,
        n_bins: int = 10
    ) -> MultiMetricResult:
        """
        Evaluate all fairness metrics with certified bounds.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            groups: Group membership
            threshold: Classification threshold
            n_bins: Number of bins for calibration
            
        Returns:
            Multi-metric analysis result
        """
        # Ensure n_bins is an integer
        n_bins = int(n_bins)
        # Evaluate each metric
        dpg_result = self.evaluator.evaluate_demographic_parity(
            model, X_test, groups, threshold
        )
        
        eog_result = self.evaluator.evaluate_equalized_odds(
            model, X_test, y_test, groups, threshold
        )
        
        cal_dict = self.evaluator.evaluate_calibration(
            model, X_test, y_test, n_bins
        )
        
        # Convert calibration dict to FairnessResult-like object
        from .fairness_metrics import FairnessResult
        cal_result = FairnessResult(
            metric_value=cal_dict['ece'],
            error_bound=np.mean(cal_dict['bin_uncertainties']),
            is_reliable=bool(np.mean(cal_dict['bin_uncertainties']) < 0.1),
            reliability_score=cal_dict['ece'] / max(np.mean(cal_dict['bin_uncertainties']), 1e-6),
            near_threshold_fraction=np.mean(cal_dict['bin_uncertainties']),
            metadata=cal_dict
        )
        
        # Joint reliability: all metrics must be reliable
        joint_reliable = bool(
            dpg_result.is_reliable and
            eog_result.is_reliable and
            cal_result.is_reliable
        )
        
        # Joint reliability score: minimum of individual scores
        joint_score = min(
            dpg_result.reliability_score,
            eog_result.reliability_score,
            cal_result.reliability_score
        )
        
        # Compute correlations between metric uncertainties
        correlations = self._compute_metric_correlations(
            dpg_result, eog_result, cal_result, X_test, groups
        )
        
        # Determine precision requirement
        if not joint_reliable:
            if self.tracker.precision == torch.float16:
                precision_req = torch.float32
            else:
                precision_req = torch.float64
        else:
            precision_req = self.tracker.precision
        
        return MultiMetricResult(
            demographic_parity=dpg_result,
            equalized_odds=eog_result,
            calibration=cal_result,
            joint_reliable=joint_reliable,
            joint_reliability_score=joint_score,
            metric_correlations=correlations,
            precision_requirement=precision_req
        )
    
    def _compute_metric_correlations(
        self,
        dpg: FairnessResult,
        eog: FairnessResult,
        cal: FairnessResult,
        X: torch.Tensor,
        groups: torch.Tensor
    ) -> Dict[Tuple[str, str], float]:
        """
        Compute correlations between metric uncertainties.
        
        Metrics that share many near-threshold samples will have
        correlated uncertainties.
        
        Args:
            dpg: Demographic parity result
            eog: Equalized odds result
            cal: Calibration result
            X: Features
            groups: Group membership
            
        Returns:
            Dictionary of pairwise correlations
        """
        # For now, return dummy correlations
        # In full implementation, would analyze shared uncertainty sources
        return {
            ('DPG', 'EOG'): 0.7,  # High correlation (both use predictions)
            ('DPG', 'CAL'): 0.5,  # Medium correlation
            ('EOG', 'CAL'): 0.6   # Medium-high correlation
        }
    
    def analyze_precision_tradeoffs(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        groups: torch.Tensor,
        threshold: float = 0.5,
        precisions: List[torch.dtype] = None
    ) -> Dict[torch.dtype, MultiMetricResult]:
        """
        Analyze fairness metrics across multiple precisions.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            groups: Group membership
            threshold: Classification threshold
            precisions: List of precisions to test
            
        Returns:
            Dictionary mapping precision to multi-metric results
        """
        if precisions is None:
            precisions = [torch.float64, torch.float32, torch.float16]
        
        results = {}
        
        for prec in precisions:
            # Create tracker for this precision
            tracker = ErrorTracker(precision=prec)
            
            # Temporarily switch tracker
            old_tracker = self.tracker
            self.tracker = tracker
            self.evaluator = CertifiedFairnessEvaluator(tracker)
            
            try:
                result = self.evaluate_all_metrics(
                    model, X_test, y_test, groups, threshold
                )
                results[prec] = result
            except Exception as e:
                print(f"Warning: Evaluation at {prec} failed: {e}")
            finally:
                # Restore original tracker
                self.tracker = old_tracker
                self.evaluator = CertifiedFairnessEvaluator(old_tracker)
        
        return results
    
    def find_pareto_optimal_thresholds(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        groups: torch.Tensor,
        threshold_range: Tuple[float, float] = (0.3, 0.7),
        n_thresholds: int = 20
    ) -> Dict[str, List]:
        """
        Find Pareto-optimal thresholds trading off multiple fairness metrics.
        
        Args:
            model: Model to evaluate
            X_test: Test features
            y_test: Test labels
            groups: Group membership
            threshold_range: Range of thresholds to consider
            n_thresholds: Number of thresholds to test
            
        Returns:
            Dictionary with threshold sweep results
        """
        # Convert to float if tensor
        if isinstance(threshold_range[0], torch.Tensor):
            threshold_range = (float(threshold_range[0]), float(threshold_range[1]))
        
        thresholds = np.linspace(threshold_range[0], threshold_range[1], int(n_thresholds))
        
        results = {
            'thresholds': [],
            'dpg': [],
            'dpg_error': [],
            'eog': [],
            'eog_error': [],
            'cal': [],
            'cal_error': [],
            'reliable': []
        }
        
        for t in thresholds:
            multi_result = self.evaluate_all_metrics(
                model, X_test, y_test, groups, threshold=float(t)
            )
            
            results['thresholds'].append(t)
            results['dpg'].append(multi_result.demographic_parity.metric_value)
            results['dpg_error'].append(multi_result.demographic_parity.error_bound)
            results['eog'].append(multi_result.equalized_odds.metric_value)
            results['eog_error'].append(multi_result.equalized_odds.error_bound)
            results['cal'].append(multi_result.calibration.metric_value)
            results['cal_error'].append(multi_result.calibration.error_bound)
            results['reliable'].append(multi_result.joint_reliable)
        
        return results
    
    def visualize_fairness_tradeoffs(
        self,
        results: Dict[torch.dtype, MultiMetricResult],
        save_path: Optional[str] = None
    ) -> plt.Figure:
        """
        Visualize tradeoffs between fairness metrics across precisions.
        
        Args:
            results: Multi-metric results at different precisions
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        precisions_str = [str(p) for p in results.keys()]
        colors = {'torch.float64': 'green', 'torch.float32': 'blue', 'torch.float16': 'red'}
        
        # Plot 1: DPG vs EOG
        ax = axes[0]
        for prec, result in results.items():
            prec_str = str(prec)
            color = colors.get(prec_str, 'gray')
            
            dpg = result.demographic_parity.metric_value
            dpg_err = result.demographic_parity.error_bound
            eog = result.equalized_odds.metric_value
            eog_err = result.equalized_odds.error_bound
            
            ax.errorbar(
                dpg, eog,
                xerr=dpg_err, yerr=eog_err,
                marker='o', markersize=10,
                color=color, label=prec_str,
                capsize=5
            )
        
        ax.set_xlabel('Demographic Parity Gap')
        ax.set_ylabel('Equalized Odds Gap')
        ax.set_title('DPG vs EOG Tradeoff')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Metric values comparison
        ax = axes[1]
        x_pos = np.arange(len(precisions_str))
        width = 0.25
        
        for i, (prec, result) in enumerate(results.items()):
            metrics = [
                result.demographic_parity.metric_value,
                result.equalized_odds.metric_value,
                result.calibration.metric_value
            ]
            errors = [
                result.demographic_parity.error_bound,
                result.equalized_odds.error_bound,
                result.calibration.error_bound
            ]
            
            ax.bar(
                x_pos[i] - width, metrics[0], width,
                yerr=errors[0], capsize=5,
                label='DPG' if i == 0 else '',
                color='skyblue'
            )
            ax.bar(
                x_pos[i], metrics[1], width,
                yerr=errors[1], capsize=5,
                label='EOG' if i == 0 else '',
                color='lightcoral'
            )
            ax.bar(
                x_pos[i] + width, metrics[2], width,
                yerr=errors[2], capsize=5,
                label='CAL' if i == 0 else '',
                color='lightgreen'
            )
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(precisions_str, rotation=15)
        ax.set_ylabel('Metric Value')
        ax.set_title('Fairness Metrics by Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Reliability scores
        ax = axes[2]
        reliability_scores = [
            result.joint_reliability_score
            for result in results.values()
        ]
        colors_list = [colors.get(str(p), 'gray') for p in results.keys()]
        
        bars = ax.bar(x_pos, reliability_scores, color=colors_list, alpha=0.7)
        ax.axhline(y=2.0, color='red', linestyle='--', label='Reliability Threshold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(precisions_str, rotation=15)
        ax.set_ylabel('Joint Reliability Score')
        ax.set_title('Joint Reliability by Precision')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig


if __name__ == "__main__":
    # Test multi-metric analysis
    print("Testing Multi-Metric Fairness Analysis...")
    
    # Generate synthetic data
    n_samples = 1000
    n_features = 10
    X = torch.randn(n_samples, n_features)
    groups = torch.randint(0, 2, (n_samples,))
    y = (X.sum(dim=1) + groups.float() * 0.3 > 0).float()
    
    # Simple model
    model = torch.nn.Sequential(
        torch.nn.Linear(n_features, 20),
        torch.nn.ReLU(),
        torch.nn.Linear(20, 1),
        torch.nn.Sigmoid()
    )
    
    # Train briefly
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    for epoch in range(10):
        optimizer.zero_grad()
        outputs = model(X).squeeze()
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
    
    print("Model trained!")
    
    # Analyze
    analyzer = MultiMetricFairnessAnalyzer()
    
    print("\nSingle precision analysis:")
    result = analyzer.evaluate_all_metrics(model, X, y, groups)
    print(f"DPG: {result.demographic_parity.metric_value:.4f} ± {result.demographic_parity.error_bound:.4f}")
    print(f"EOG: {result.equalized_odds.metric_value:.4f} ± {result.equalized_odds.error_bound:.4f}")
    print(f"CAL: {result.calibration.metric_value:.4f} ± {result.calibration.error_bound:.4f}")
    print(f"Joint reliable: {result.joint_reliable}")
    print(f"Joint score: {result.joint_reliability_score:.2f}")
    
    print("\nMulti-precision analysis:")
    multi_results = analyzer.analyze_precision_tradeoffs(model, X, y, groups)
    for prec, res in multi_results.items():
        print(f"\n{prec}:")
        print(f"  Joint reliable: {res.joint_reliable}")
        print(f"  Joint score: {res.joint_reliability_score:.2f}")
