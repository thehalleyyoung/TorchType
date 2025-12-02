"""
Tests for cross-precision validation functionality.
"""

import pytest
import torch
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from cross_precision_validator import (
    analyze_cross_precision,
    validate_error_bounds,
    create_cross_precision_error_functional,
    CrossPrecisionAnalysis
)
from models import FairMLPClassifier
from datasets import generate_synthetic_tabular


class TestCrossPrecisionAnalysis:
    """Test cross-precision analysis functionality"""
    
    def test_dataclass_creation(self):
        """Test CrossPrecisionAnalysis dataclass"""
        analysis = CrossPrecisionAnalysis(
            baseline_predictions=np.array([0.5, 0.6]),
            target_predictions=np.array([0.51, 0.61]),
            max_absolute_diff=0.01,
            mean_absolute_diff=0.01,
            max_relative_diff=0.02,
            predictions_changed=2,
            predictions_total=2
        )
        
        assert analysis.get_error_bound() == 0.01
    
    def test_analyze_cross_precision_basic(self):
        """Test basic cross-precision analysis"""
        # Create a simple model
        model = FairMLPClassifier(input_dim=10, hidden_dims=[16, 8]).to(torch.float64)
        data = torch.randn(50, 10)
        
        # Analyze float64 -> float32
        analysis = analyze_cross_precision(
            model, data,
            baseline_precision=torch.float64,
            target_precision=torch.float32,
            device='cpu'
        )
        
        assert isinstance(analysis, CrossPrecisionAnalysis)
        assert analysis.predictions_total == 50
        assert analysis.max_absolute_diff >= 0
        assert analysis.mean_absolute_diff >= 0
        assert analysis.mean_absolute_diff <= analysis.max_absolute_diff
    
    def test_analyze_cross_precision_float16(self):
        """Test cross-precision analysis with float16"""
        model = FairMLPClassifier(input_dim=10, hidden_dims=[16, 8]).to(torch.float64)
        data = torch.randn(50, 10)
        
        # Analyze float64 -> float16
        analysis = analyze_cross_precision(
            model, data,
            baseline_precision=torch.float64,
            target_precision=torch.float16,
            device='cpu'
        )
        
        # Float16 should have larger differences than float32
        assert analysis.max_absolute_diff > 0
        assert analysis.predictions_changed > 0
    
    def test_validate_error_bounds(self):
        """Test full error bounds validation"""
        # Generate synthetic data
        _, _, _, test_data, _, test_groups = generate_synthetic_tabular(
            300, 10, fairness_gap=0.1, seed=42
        )
        
        # Create and train model
        model = FairMLPClassifier(input_dim=10, hidden_dims=[16]).to(torch.float64).cpu()
        
        # Validate bounds
        results = validate_error_bounds(
            model, test_data, test_groups.numpy(),
            threshold=0.5, device='cpu'
        )
        
        assert 'float32' in results
        assert 'float16' in results
        
        # Check structure
        for precision in ['float32', 'float16']:
            r = results[precision]
            assert 'max_pred_diff' in r
            assert 'dpg_baseline' in r
            assert 'dpg_target' in r
            assert 'dpg_diff' in r
            assert 'error_bound' in r
            assert 'near_threshold_fraction' in r
            assert 'theoretical_dpg_bound' in r
            
            # Bounds should be non-negative
            assert r['max_pred_diff'] >= 0
            assert r['error_bound'] >= 0
            assert r['near_threshold_fraction'] >= 0
            assert r['theoretical_dpg_bound'] >= 0
    
    def test_theoretical_bound_holds(self):
        """Test that theoretical bounds actually hold"""
        # Generate data
        _, _, _, test_data, _, test_groups = generate_synthetic_tabular(
            200, 10, fairness_gap=0.1, seed=43
        )
        
        model = FairMLPClassifier(input_dim=10, hidden_dims=[16]).to(torch.float64).cpu()
        
        results = validate_error_bounds(
            model, test_data, test_groups.numpy(),
            threshold=0.5, device='cpu'
        )
        
        # The theoretical bound should be >= actual DPG difference
        for precision in ['float32', 'float16']:
            r = results[precision]
            # Add small tolerance for numerical issues
            assert r['dpg_diff'] <= r['theoretical_dpg_bound'] + 1e-6, \
                f"{precision}: DPG diff {r['dpg_diff']} exceeds bound {r['theoretical_dpg_bound']}"
    
    def test_float16_has_larger_errors(self):
        """Test that float16 has larger errors than float32"""
        _, _, _, test_data, _, test_groups = generate_synthetic_tabular(
            200, 10, fairness_gap=0.1, seed=44
        )
        
        model = FairMLPClassifier(input_dim=10, hidden_dims=[32, 16]).to(torch.float64).cpu()
        
        results = validate_error_bounds(
            model, test_data, test_groups.numpy(),
            threshold=0.5, device='cpu'
        )
        
        # Float16 should have larger max prediction difference
        assert results['float16']['max_pred_diff'] >= results['float32']['max_pred_diff']
    
    def test_create_cross_precision_functional(self):
        """Test creation of cross-precision error functional"""
        model = FairMLPClassifier(input_dim=10, hidden_dims=[16]).to(torch.float64)
        data = torch.randn(50, 10)
        
        functional = create_cross_precision_error_functional(
            model, data,
            baseline_precision=torch.float64,
            target_precision=torch.float32,
            device='cpu'
        )
        
        # Should return a LinearErrorFunctional
        assert hasattr(functional, 'lipschitz')
        assert hasattr(functional, 'roundoff')
        assert functional.lipschitz >= 0
        assert functional.roundoff >= 0


class TestAdversarialScenarios:
    """Test generation of adversarial scenarios"""
    
    def test_tight_clustering_scenario(self):
        """Test predictions tightly clustered around threshold"""
        n_samples = 100
        threshold = 0.5
        spread = 0.001
        
        # Create predictions
        np.random.seed(42)
        predictions = threshold + np.random.uniform(-spread, spread, n_samples)
        predictions = np.clip(predictions, 0.001, 0.999)
        
        # Check clustering
        near_threshold = np.abs(predictions - threshold) < spread
        assert near_threshold.mean() >= 0.9, "Should be tightly clustered"
        
        # Simulate precision effects
        preds_f16 = predictions + np.random.normal(0, 2e-4, n_samples)
        
        # Count how many could flip
        could_flip = np.abs(predictions - threshold) < 2e-4
        assert could_flip.sum() > 0, "Some predictions should be near threshold"
    
    def test_bimodal_scenario(self):
        """Test bimodal distribution straddling threshold"""
        n_samples = 100
        threshold = 0.5
        
        np.random.seed(42)
        n_half = n_samples // 2
        
        # Half below, half above
        preds_bimodal = np.concatenate([
            threshold - 0.0003 + np.random.uniform(-0.0001, 0.0001, n_half),
            threshold + 0.0003 + np.random.uniform(-0.0001, 0.0001, n_samples - n_half)
        ])
        
        # Should be roughly balanced
        frac_above = (preds_bimodal > threshold).mean()
        assert 0.4 < frac_above < 0.6, "Should be roughly balanced"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
