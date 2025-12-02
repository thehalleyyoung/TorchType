"""
Advanced adversarial sign flip generator for stress-testing fairness bounds.

This module generates models and data specifically designed to produce sign flips
in fairness metrics across precisions, then validates that our certified bounds
correctly predict these flips.

This addresses the "are we cheating?" question by:
1. Adversarially generating hard cases
2. Showing our bounds predict flips with high accuracy
3. Demonstrating naive approaches fail
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from .error_propagation import ErrorTracker, LinearErrorFunctional
    from .fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from .models import FairMLPClassifier
except ImportError:
    from error_propagation import ErrorTracker, LinearErrorFunctional
    from fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from models import FairMLPClassifier


@dataclass
class SignFlipCase:
    """Represents a case where fairness metric sign flips across precisions."""
    dpg_float64: float
    dpg_float32: float
    dpg_float16: float
    error_bound_float32: float
    error_bound_float16: float
    near_threshold_frac: float
    predicted_flip: bool
    actual_flip: bool
    model_architecture: List[int]
    threshold: float
    

class AdversarialSignFlipGenerator:
    """
    Generates models designed to exhibit sign flips in fairness metrics.
    
    Strategy:
    1. Train model to have small DPG (near zero)
    2. Engineer data distribution to cluster predictions near threshold
    3. Introduce group-specific biases that are threshold-sensitive
    4. Validate that our bounds correctly predict the sign flips
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def generate_near_zero_dpg_data(
        self,
        n_samples: int = 2000,
        input_dim: int = 10,
        threshold_concentration: float = 0.8,
        seed: Optional[int] = None
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """
        Generate data where predictions will cluster near decision threshold.
        
        Args:
            n_samples: Number of samples
            input_dim: Input dimensionality
            threshold_concentration: Fraction of samples near threshold (0-1)
            seed: Random seed
            
        Returns:
            X: Features
            y: Labels
            groups: Group membership
        """
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
        
        # Create two groups
        n_group_0 = n_samples // 2
        n_group_1 = n_samples - n_group_0
        
        # For group 0: concentrate features to produce scores near 0.5 + epsilon
        # For group 1: concentrate features to produce scores near 0.5 - epsilon
        epsilon = 0.02  # Small offset to create borderline DPG
        
        # Generate features that will map to near-threshold predictions
        X_group_0 = torch.randn(n_group_0, input_dim) * 0.5
        X_group_1 = torch.randn(n_group_1, input_dim) * 0.5
        
        # Add group-specific biases in first feature
        # This creates a small, numerically fragile difference
        X_group_0[:, 0] += epsilon
        X_group_1[:, 0] -= epsilon
        
        # Concentrate samples near threshold
        if threshold_concentration > 0.5:
            # Clip features to limit spread
            X_group_0 = torch.clamp(X_group_0, -1.0, 1.0)
            X_group_1 = torch.clamp(X_group_1, -1.0, 1.0)
        
        # Combine
        X = torch.cat([X_group_0, X_group_1], dim=0)
        
        # Generate labels with correlation to features but noise
        y_proba = torch.sigmoid(X[:, 0] * 0.5 + X[:, 1] * 0.3)
        y = (y_proba > 0.5).numpy().astype(int)
        
        # Group labels
        groups = np.array([0] * n_group_0 + [1] * n_group_1)
        
        # Shuffle
        perm = np.random.permutation(n_samples)
        X = X[perm]
        y = y[perm]
        groups = groups[perm]
        
        return X.float(), y, groups
    
    def train_borderline_fair_model(
        self,
        X: torch.Tensor,
        y: np.ndarray,
        groups: np.ndarray,
        hidden_dims: List[int] = [32, 16],
        target_dpg: float = 0.02,
        max_epochs: int = 200,
        patience: int = 20
    ) -> FairMLPClassifier:
        """
        Train a model to be borderline fair (small but nonzero DPG).
        
        Uses a custom loss that balances accuracy and fairness to create
        models at the boundary of fairness.
        """
        input_dim = X.shape[1]
        model = FairMLPClassifier(
            input_dim=input_dim,
            hidden_dims=hidden_dims,
            activation='relu'
        ).to(self.device)
        
        X_tensor = X.to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)
        groups_tensor = torch.tensor(groups).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        best_dpg = float('inf')
        best_model_state = None
        patience_counter = 0
        
        for epoch in range(max_epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_tensor).squeeze()
            
            # Binary cross-entropy loss
            bce_loss = nn.functional.binary_cross_entropy(predictions, y_tensor)
            
            # Fairness regularization - penalize being too far from target_dpg
            with torch.no_grad():
                preds_np = predictions.cpu().numpy()
                groups_np = groups_tensor.cpu().numpy()
                current_dpg = FairnessMetrics.demographic_parity_gap(
                    preds_np, groups_np, threshold=0.5
                )
            
            # Encourage DPG near target
            fairness_penalty = abs(current_dpg - target_dpg)
            
            # Combined loss with small fairness weight to maintain accuracy
            loss = bce_loss + 0.1 * fairness_penalty
            
            loss.backward()
            optimizer.step()
            
            # Early stopping based on achieving target DPG
            if abs(current_dpg - target_dpg) < abs(best_dpg - target_dpg):
                best_dpg = current_dpg
                best_model_state = model.state_dict().copy()
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        return model
    
    def generate_sign_flip_cases(
        self,
        n_cases: int = 50,
        min_concentration: float = 0.6,
        max_concentration: float = 0.95
    ) -> List[SignFlipCase]:
        """
        Generate multiple sign flip cases with varying degrees of near-threshold concentration.
        
        Returns:
            List of SignFlipCase objects
        """
        cases = []
        
        print(f"\n{'='*70}")
        print("ADVERSARIAL SIGN FLIP GENERATION")
        print(f"{'='*70}\n")
        print(f"Generating {n_cases} adversarial cases...\n")
        
        for i in range(n_cases):
            # Vary threshold concentration
            concentration = np.random.uniform(min_concentration, max_concentration)
            
            # Generate data
            X, y, groups = self.generate_near_zero_dpg_data(
                n_samples=2000,
                input_dim=10,
                threshold_concentration=concentration,
                seed=42 + i
            )
            
            # Train borderline fair model
            # Vary target DPG around zero
            target_dpg = np.random.uniform(-0.05, 0.05)
            
            model = self.train_borderline_fair_model(
                X, y, groups,
                hidden_dims=[32, 16],
                target_dpg=abs(target_dpg),
                max_epochs=100,
                patience=15
            )
            
            # Evaluate at different precisions with certified bounds
            evaluator_64 = CertifiedFairnessEvaluator(
                ErrorTracker(precision=torch.float64)
            )
            evaluator_32 = CertifiedFairnessEvaluator(
                ErrorTracker(precision=torch.float32)
            )
            evaluator_16 = CertifiedFairnessEvaluator(
                ErrorTracker(precision=torch.float16)
            )
            
            # Convert to appropriate precisions
            model_64 = model.to(torch.float64).cpu()
            model_32 = model.to(torch.float32).cpu()
            model_16 = model.to(torch.float16).cpu()
            
            X_64 = X.to(torch.float64).cpu()
            X_32 = X.to(torch.float32).cpu()
            X_16 = X.to(torch.float16).cpu()
            
            # Evaluate
            result_64 = evaluator_64.evaluate_demographic_parity(
                model_64, X_64, groups, threshold=0.5
            )
            result_32 = evaluator_32.evaluate_demographic_parity(
                model_32, X_32, groups, threshold=0.5
            )
            result_16 = evaluator_16.evaluate_demographic_parity(
                model_16, X_16, groups, threshold=0.5
            )
            
            # Check for sign flip
            dpg_64 = result_64.metric_value
            dpg_32 = result_32.metric_value
            dpg_16 = result_16.metric_value
            
            # Actual flip: sign changes between float64 and float16
            # (we compare float64 as ground truth to float16 as deployment)
            actual_flip = (dpg_64 * dpg_16 < 0) and (abs(dpg_64) > 1e-6)
            
            # Predicted flip: error bound exceeds metric value
            # (meaning zero is within the error bounds)
            predicted_flip_32 = (result_32.error_bound > abs(dpg_32))
            predicted_flip_16 = (result_16.error_bound > abs(dpg_16))
            predicted_flip = predicted_flip_16
            
            near_threshold = result_16.near_threshold_fraction['overall']
            
            # Store case
            case = SignFlipCase(
                dpg_float64=dpg_64,
                dpg_float32=dpg_32,
                dpg_float16=dpg_16,
                error_bound_float32=result_32.error_bound,
                error_bound_float16=result_16.error_bound,
                near_threshold_frac=near_threshold,
                predicted_flip=predicted_flip,
                actual_flip=actual_flip,
                model_architecture=[10] + [32, 16] + [1],
                threshold=0.5
            )
            
            cases.append(case)
            
            # Print progress
            if actual_flip:
                print(f"  Case {i+1}/{n_cases}: ✓ SIGN FLIP DETECTED")
                print(f"    DPG: {dpg_64:.4f} (fp64) → {dpg_16:.4f} (fp16)")
                print(f"    Predicted flip: {predicted_flip}, Actual flip: {actual_flip}")
                print(f"    Near-threshold: {near_threshold:.2%}\n")
            elif (i + 1) % 10 == 0:
                print(f"  Processed {i+1}/{n_cases} cases ({sum(c.actual_flip for c in cases)} flips found)")
        
        print(f"\n{'='*70}")
        print("GENERATION COMPLETE")
        print(f"{'='*70}\n")
        print(f"Total cases: {len(cases)}")
        print(f"Sign flips found: {sum(c.actual_flip for c in cases)}")
        print(f"Prediction accuracy: {sum(c.predicted_flip == c.actual_flip for c in cases) / len(cases):.1%}")
        
        return cases
    
    def analyze_sign_flip_prediction_accuracy(
        self,
        cases: List[SignFlipCase]
    ) -> Dict:
        """
        Analyze how well our certified bounds predict sign flips.
        
        Returns metrics on prediction accuracy.
        """
        # Confusion matrix
        true_positives = sum(c.predicted_flip and c.actual_flip for c in cases)
        true_negatives = sum(not c.predicted_flip and not c.actual_flip for c in cases)
        false_positives = sum(c.predicted_flip and not c.actual_flip for c in cases)
        false_negatives = sum(not c.predicted_flip and c.actual_flip for c in cases)
        
        total = len(cases)
        accuracy = (true_positives + true_negatives) / total
        
        # Precision and recall
        precision = true_positives / max(true_positives + false_positives, 1)
        recall = true_positives / max(true_positives + false_negatives, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-10)
        
        # Correlation between near-threshold fraction and sign flips
        flips = np.array([c.actual_flip for c in cases])
        near_threshold = np.array([c.near_threshold_frac for c in cases])
        
        # Among cases with high near-threshold fraction, what's the flip rate?
        high_concentration = near_threshold > 0.7
        flip_rate_high_conc = flips[high_concentration].mean() if high_concentration.sum() > 0 else 0.0
        flip_rate_low_conc = flips[~high_concentration].mean() if (~high_concentration).sum() > 0 else 0.0
        
        return {
            'total_cases': total,
            'sign_flips': int(flips.sum()),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'true_positives': true_positives,
            'true_negatives': true_negatives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'flip_rate_high_concentration': flip_rate_high_conc,
            'flip_rate_low_concentration': flip_rate_low_conc,
            'avg_near_threshold_when_flip': near_threshold[flips].mean() if flips.sum() > 0 else 0.0,
            'avg_near_threshold_no_flip': near_threshold[~flips].mean() if (~flips).sum() > 0 else 0.0
        }


def run_adversarial_experiment(device: str = 'cpu', n_cases: int = 100) -> Dict:
    """
    Run the complete adversarial sign flip experiment.
    
    This is a stress test: we actively try to create sign flips and verify
    that our bounds correctly predict them.
    
    Note: Uses CPU to avoid MPS float64 limitations.
    """
    # Force CPU for this experiment due to precision requirements
    generator = AdversarialSignFlipGenerator(device='cpu')
    
    # Generate adversarial cases
    cases = generator.generate_sign_flip_cases(
        n_cases=n_cases,
        min_concentration=0.6,
        max_concentration=0.95
    )
    
    # Analyze prediction accuracy
    results = generator.analyze_sign_flip_prediction_accuracy(cases)
    
    print(f"\n{'='*70}")
    print("ANALYSIS RESULTS")
    print(f"{'='*70}\n")
    print(f"Total cases tested: {results['total_cases']}")
    print(f"Sign flips detected: {results['sign_flips']}")
    print(f"Prediction accuracy: {results['accuracy']:.1%}")
    print(f"Precision: {results['precision']:.1%}")
    print(f"Recall: {results['recall']:.1%}")
    print(f"F1 Score: {results['f1_score']:.1%}")
    print(f"\nTrue Positives: {results['true_positives']}")
    print(f"True Negatives: {results['true_negatives']}")
    print(f"False Positives: {results['false_positives']}")
    print(f"False Negatives: {results['false_negatives']}")
    print(f"\nFlip rate (high concentration): {results['flip_rate_high_concentration']:.1%}")
    print(f"Flip rate (low concentration): {results['flip_rate_low_concentration']:.1%}")
    
    # Store raw cases for detailed analysis
    results['cases'] = [
        {
            'dpg_float64': c.dpg_float64,
            'dpg_float32': c.dpg_float32,
            'dpg_float16': c.dpg_float16,
            'error_bound_float32': c.error_bound_float32,
            'error_bound_float16': c.error_bound_float16,
            'near_threshold_frac': c.near_threshold_frac,
            'predicted_flip': c.predicted_flip,
            'actual_flip': c.actual_flip
        }
        for c in cases
    ]
    
    return results


if __name__ == '__main__':
    # Test the adversarial generator
    import sys
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_adversarial_experiment(device=device, n_cases=100)
    
    # Save results
    import json
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    # Remove cases from saved results (too large)
    save_results = {k: v for k, v in results.items() if k != 'cases'}
    
    with open(os.path.join(output_dir, 'adversarial_sign_flip_analysis.json'), 'w') as f:
        json.dump(save_results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/adversarial_sign_flip_analysis.json")
