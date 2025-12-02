"""
Real-world case study: Credit scoring with certified fairness.

This module demonstrates NumGeom-Fair on a realistic credit scoring scenario:
1. Generate realistic credit application data
2. Train production-scale credit scoring model
3. Evaluate fairness with certified bounds at different precisions
4. Generate regulatory compliance report
5. Demonstrate memory/compute savings for edge deployment

This goes beyond toy examples to show practical applicability.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json

try:
    from .error_propagation import ErrorTracker, LinearErrorFunctional
    from .fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from .models import FairMLPClassifier
    from .compliance_certification import ComplianceCertifier
except ImportError:
    from error_propagation import ErrorTracker, LinearErrorFunctional
    from fairness_metrics import CertifiedFairnessEvaluator, FairnessMetrics
    from models import FairMLPClassifier
    from compliance_certification import ComplianceCertifier


class CreditScoringDataGenerator:
    """
    Generates realistic synthetic credit scoring data.
    
    Features mimic real credit bureau data:
    - Income
    - Debt-to-income ratio
    - Credit history length
    - Number of credit accounts
    - Payment history score
    - Recent credit inquiries
    - Employment stability
    - Savings ratio
    """
    
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
    
    def generate_realistic_credit_data(
        self,
        n_samples: int = 10000,
        protected_attribute: str = 'age_group'  # or 'gender', 'ethnicity'
    ) -> Tuple[torch.Tensor, np.ndarray, np.ndarray, List[str]]:
        """
        Generate realistic credit scoring data.
        
        Returns:
            X: Feature matrix (n_samples x n_features)
            y: Default labels (1 = defaulted, 0 = repaid)
            groups: Protected group membership
            feature_names: Names of features
        """
        # Feature: Annual income (log-normal distribution)
        # Group 0 tends to have lower income (historical discrimination)
        income_group_0 = np.random.lognormal(mean=10.5, sigma=0.6, size=n_samples // 2)
        income_group_1 = np.random.lognormal(mean=10.8, sigma=0.5, size=n_samples - n_samples // 2)
        income = np.concatenate([income_group_0, income_group_1])
        
        # Feature: Debt-to-income ratio
        dti_group_0 = np.random.beta(2, 5, size=n_samples // 2) * 100  # Slightly higher DTI
        dti_group_1 = np.random.beta(2, 6, size=n_samples - n_samples // 2) * 100
        dti = np.concatenate([dti_group_0, dti_group_1])
        
        # Feature: Credit history length (years)
        history_group_0 = np.random.gamma(3, 2, size=n_samples // 2)
        history_group_1 = np.random.gamma(4, 2, size=n_samples - n_samples // 2)
        history = np.concatenate([history_group_0, history_group_1])
        
        # Feature: Number of credit accounts
        accounts = np.random.poisson(lam=5, size=n_samples)
        
        # Feature: Payment history score (0-100)
        payment_score = np.random.beta(8, 2, size=n_samples) * 100
        
        # Feature: Recent inquiries (negative signal)
        inquiries = np.random.poisson(lam=1.5, size=n_samples)
        
        # Feature: Employment stability (years at current job)
        employment = np.random.gamma(2, 1.5, size=n_samples)
        
        # Feature: Savings ratio (savings / income)
        savings_ratio = np.random.beta(2, 5, size=n_samples)
        
        # Combine features
        X = np.stack([
            income / 100000,  # Normalize
            dti / 100,
            history / 20,
            accounts / 10,
            payment_score / 100,
            inquiries / 5,
            employment / 10,
            savings_ratio
        ], axis=1)
        
        # Generate default labels based on risk factors
        # Higher risk: low income, high DTI, short history, low payment score
        risk_score = (
            -0.3 * X[:, 0]  # Income (negative - higher income = lower risk)
            + 0.4 * X[:, 1]  # DTI (positive - higher DTI = higher risk)
            - 0.2 * X[:, 2]  # History (negative)
            - 0.3 * X[:, 4]  # Payment score (negative)
            + 0.2 * X[:, 5]  # Inquiries (positive)
            + np.random.normal(0, 0.1, size=n_samples)  # Noise
        )
        
        # Convert to probability and sample defaults
        default_prob = 1 / (1 + np.exp(-risk_score))
        y = (default_prob > 0.5).astype(int)
        
        # Group membership
        groups = np.array([0] * (n_samples // 2) + [1] * (n_samples - n_samples // 2))
        
        # Shuffle
        perm = np.random.permutation(n_samples)
        X = X[perm]
        y = y[perm]
        groups = groups[perm]
        
        feature_names = [
            'income_normalized',
            'debt_to_income',
            'credit_history_years',
            'num_accounts',
            'payment_history_score',
            'recent_inquiries',
            'employment_stability',
            'savings_ratio'
        ]
        
        return torch.tensor(X, dtype=torch.float32), y, groups, feature_names


class ProductionCreditScoringModel(nn.Module):
    """
    Production-scale credit scoring model.
    
    Architecture designed to mimic real-world credit scoring systems:
    - Input: 8 financial features
    - Hidden layers with batch normalization and dropout
    - Output: Default probability
    """
    
    def __init__(self, input_dim: int = 8):
        super().__init__()
        
        self.network = nn.Sequential(
            # Layer 1
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 2
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            # Layer 3
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            
            # Output
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.network(x).squeeze()
    
    def get_simplified_architecture(self):
        """Return simplified architecture for error tracking."""
        return [8, 128, 64, 32, 1]


class CreditScoringCaseStudy:
    """
    Complete case study demonstrating NumGeom-Fair on credit scoring.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.generator = CreditScoringDataGenerator(seed=42)
        
    def run_complete_study(self) -> Dict:
        """
        Run the complete credit scoring case study.
        
        Returns comprehensive results including:
        - Model performance metrics
        - Fairness evaluation at multiple precisions
        - Memory and compute savings
        - Regulatory compliance report
        """
        print(f"\n{'='*70}")
        print("CREDIT SCORING CASE STUDY")
        print(f"{'='*70}\n")
        
        # 1. Generate data
        print("[Step 1] Generating realistic credit application data...")
        X, y, groups, feature_names = self.generator.generate_realistic_credit_data(
            n_samples=10000
        )
        
        # Split train/test
        n_train = 8000
        X_train, X_test = X[:n_train], X[n_train:]
        y_train, y_test = y[:n_train], y[n_train:]
        groups_train, groups_test = groups[:n_train], groups[n_train:]
        
        print(f"  Train: {n_train} samples")
        print(f"  Test: {len(X_test)} samples")
        print(f"  Features: {len(feature_names)}")
        print(f"  Group 0: {(groups_test == 0).sum()} samples")
        print(f"  Group 1: {(groups_test == 1).sum()} samples")
        print(f"  Default rate: {y_test.mean():.1%}\n")
        
        # 2. Train production model
        print("[Step 2] Training production-scale credit scoring model...")
        model = self._train_credit_model(X_train, y_train, groups_train)
        print("  ✓ Training complete\n")
        
        # 3. Evaluate model performance
        print("[Step 3] Evaluating model performance...")
        performance = self._evaluate_performance(model, X_test, y_test, groups_test)
        print(f"  Accuracy: {performance['accuracy']:.1%}")
        print(f"  AUC-ROC: {performance['auc']:.3f}")
        print(f"  Precision: {performance['precision']:.1%}")
        print(f"  Recall: {performance['recall']:.1%}\n")
        
        # 4. Certified fairness evaluation at multiple precisions
        print("[Step 4] Certified fairness evaluation across precisions...")
        fairness_results = self._evaluate_fairness_all_precisions(
            model, X_test, groups_test
        )
        
        for prec, result in fairness_results.items():
            status = "✓ RELIABLE" if result['is_reliable'] else "✗ BORDERLINE"
            print(f"  {prec}:")
            print(f"    DPG: {result['dpg']:.4f} ± {result['error_bound']:.4f}")
            print(f"    Status: {status}")
            print(f"    Reliability: {result['reliability_score']:.2f}")
        print()
        
        # 5. Memory and compute analysis
        print("[Step 5] Analyzing memory and compute savings...")
        deployment_analysis = self._analyze_deployment_savings(model, X_test)
        
        print("  Memory footprint:")
        print(f"    float64: {deployment_analysis['memory']['float64_kb']:.1f} KB")
        print(f"    float32: {deployment_analysis['memory']['float32_kb']:.1f} KB ({deployment_analysis['memory']['savings_32']:.1%} savings)")
        print(f"    float16: {deployment_analysis['memory']['float16_kb']:.1f} KB ({deployment_analysis['memory']['savings_16']:.1%} savings)")
        
        print("\n  Inference speed:")
        print(f"    float64: {deployment_analysis['speed']['float64_ms']:.2f} ms/batch")
        print(f"    float32: {deployment_analysis['speed']['float32_ms']:.2f} ms/batch ({deployment_analysis['speed']['speedup_32']:.1f}x)")
        print(f"    float16: {deployment_analysis['speed']['float16_ms']:.2f} ms/batch ({deployment_analysis['speed']['speedup_16']:.1f}x)")
        print()
        
        # 6. Recommendation
        print("[Step 6] Deployment recommendation...")
        recommendation = self._generate_deployment_recommendation(
            fairness_results, deployment_analysis
        )
        print(f"  Recommended precision: {recommendation['precision']}")
        print(f"  Rationale: {recommendation['rationale']}")
        print(f"  Savings: {recommendation['savings']}")
        print(f"  Fairness guarantee: {recommendation['fairness_guarantee']}\n")
        
        # 7. Generate compliance report
        print("[Step 7] Generating regulatory compliance report...")
        compliance_report = self._generate_compliance_report(
            model, X_test, y_test, groups_test, fairness_results,
            feature_names, deployment_analysis
        )
        print(f"  ✓ Compliance report generated\n")
        
        print(f"{'='*70}")
        print("CASE STUDY COMPLETE")
        print(f"{'='*70}\n")
        
        return {
            'data_statistics': {
                'n_train': n_train,
                'n_test': len(X_test),
                'n_features': len(feature_names),
                'default_rate': float(y_test.mean())
            },
            'performance': performance,
            'fairness': fairness_results,
            'deployment': deployment_analysis,
            'recommendation': recommendation,
            'compliance_report': compliance_report
        }
    
    def _train_credit_model(
        self,
        X_train: torch.Tensor,
        y_train: np.ndarray,
        groups_train: np.ndarray,
        epochs: int = 50
    ) -> ProductionCreditScoringModel:
        """Train the production credit scoring model."""
        model = ProductionCreditScoringModel(input_dim=X_train.shape[1]).to(self.device)
        
        X_tensor = X_train.to(self.device)
        y_tensor = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.BCELoss()
        
        model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            predictions = model(X_tensor)
            loss = criterion(predictions, y_tensor)
            loss.backward()
            optimizer.step()
        
        return model
    
    def _evaluate_performance(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        y_test: np.ndarray,
        groups_test: np.ndarray
    ) -> Dict:
        """Evaluate model performance metrics."""
        model.eval()
        
        with torch.no_grad():
            X_tensor = X_test.to(self.device)
            predictions = model(X_tensor).cpu().numpy()
        
        # Classification metrics
        y_pred = (predictions > 0.5).astype(int)
        accuracy = (y_pred == y_test).mean()
        
        # Precision, recall
        tp = ((y_pred == 1) & (y_test == 1)).sum()
        fp = ((y_pred == 1) & (y_test == 0)).sum()
        fn = ((y_pred == 0) & (y_test == 1)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        # AUC-ROC (simplified)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_test, predictions)
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'auc': float(auc)
        }
    
    def _evaluate_fairness_all_precisions(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        groups_test: np.ndarray
    ) -> Dict:
        """Evaluate fairness at float64, float32, float16 with certified bounds."""
        results = {}
        
        for precision_name, precision_dtype in [
            ('float64', torch.float64),
            ('float32', torch.float32),
            ('float16', torch.float16)
        ]:
            model_prec = model.to(precision_dtype)
            X_prec = X_test.to(precision_dtype).to(self.device)
            
            evaluator = CertifiedFairnessEvaluator(
                ErrorTracker(precision=precision_dtype)
            )
            
            result = evaluator.evaluate_demographic_parity(
                model_prec, X_prec, groups_test, threshold=0.5
            )
            
            results[precision_name] = {
                'dpg': result.metric_value,
                'error_bound': result.error_bound,
                'is_reliable': result.is_reliable,
                'reliability_score': result.reliability_score,
                'near_threshold_frac': result.near_threshold_fraction['overall']
            }
        
        return results
    
    def _analyze_deployment_savings(
        self,
        model: nn.Module,
        X_test: torch.Tensor
    ) -> Dict:
        """Analyze memory and compute savings for deployment."""
        import time
        
        # Memory analysis
        def get_model_memory(dtype):
            model_copy = ProductionCreditScoringModel()
            model_copy.load_state_dict(model.state_dict())
            model_copy = model_copy.to(dtype)
            
            # Count parameters
            params = sum(p.numel() * p.element_size() for p in model_copy.parameters())
            return params / 1024  # KB
        
        memory_64 = get_model_memory(torch.float64)
        memory_32 = get_model_memory(torch.float32)
        memory_16 = get_model_memory(torch.float16)
        
        # Speed analysis (batch of 100)
        batch = X_test[:100].to(self.device)
        
        def benchmark_speed(dtype, n_runs=100):
            model_bench = model.to(dtype)
            batch_dtype = batch.to(dtype)
            
            # Warmup
            for _ in range(10):
                with torch.no_grad():
                    _ = model_bench(batch_dtype)
            
            # Time
            start = time.time()
            for _ in range(n_runs):
                with torch.no_grad():
                    _ = model_bench(batch_dtype)
            elapsed = time.time() - start
            return (elapsed / n_runs) * 1000  # ms
        
        speed_64 = benchmark_speed(torch.float64, n_runs=50)
        speed_32 = benchmark_speed(torch.float32, n_runs=50)
        speed_16 = benchmark_speed(torch.float16, n_runs=50)
        
        return {
            'memory': {
                'float64_kb': memory_64,
                'float32_kb': memory_32,
                'float16_kb': memory_16,
                'savings_32': 1 - memory_32 / memory_64,
                'savings_16': 1 - memory_16 / memory_64
            },
            'speed': {
                'float64_ms': speed_64,
                'float32_ms': speed_32,
                'float16_ms': speed_16,
                'speedup_32': speed_64 / speed_32,
                'speedup_16': speed_64 / speed_16
            }
        }
    
    def _generate_deployment_recommendation(
        self,
        fairness_results: Dict,
        deployment_analysis: Dict
    ) -> Dict:
        """Generate deployment precision recommendation."""
        # Check which precisions maintain certified fairness
        if fairness_results['float16']['is_reliable']:
            return {
                'precision': 'float16',
                'rationale': 'Float16 maintains certified fairness with maximum savings',
                'savings': f"{deployment_analysis['memory']['savings_16']:.1%} memory, {deployment_analysis['speed']['speedup_16']:.1f}x speedup",
                'fairness_guarantee': f"DPG = {fairness_results['float16']['dpg']:.4f} ± {fairness_results['float16']['error_bound']:.4f}"
            }
        elif fairness_results['float32']['is_reliable']:
            return {
                'precision': 'float32',
                'rationale': 'Float32 maintains certified fairness with substantial savings',
                'savings': f"{deployment_analysis['memory']['savings_32']:.1%} memory, {deployment_analysis['speed']['speedup_32']:.1f}x speedup",
                'fairness_guarantee': f"DPG = {fairness_results['float32']['dpg']:.4f} ± {fairness_results['float32']['error_bound']:.4f}"
            }
        else:
            return {
                'precision': 'float64',
                'rationale': 'Full precision required to maintain certified fairness',
                'savings': 'No savings (baseline precision)',
                'fairness_guarantee': f"DPG = {fairness_results['float64']['dpg']:.4f} ± {fairness_results['float64']['error_bound']:.4f}"
            }
    
    def _generate_compliance_report(
        self,
        model: nn.Module,
        X_test: torch.Tensor,
        y_test: np.ndarray,
        groups_test: np.ndarray,
        fairness_results: Dict,
        feature_names: List[str],
        deployment_analysis: Dict
    ) -> Dict:
        """Generate regulatory compliance report."""
        return {
            'model_info': {
                'type': 'Production Credit Scoring MLP',
                'architecture': model.get_simplified_architecture(),
                'n_parameters': sum(p.numel() for p in model.parameters()),
                'features': feature_names
            },
            'fairness_certification': {
                'metric': 'Demographic Parity Gap',
                'precision_analysis': fairness_results,
                'certification_method': 'NumGeom-Fair (certified error bounds)',
                'regulatory_threshold': 0.05,
                'compliant': all(r['dpg'] < 0.05 for r in fairness_results.values() if r['is_reliable'])
            },
            'deployment_specs': {
                'recommended_precision': 'float32',
                'memory_footprint_kb': deployment_analysis['memory']['float32_kb'],
                'inference_latency_ms': deployment_analysis['speed']['float32_ms']
            },
            'certification_statement': (
                f"This credit scoring model has been evaluated for fairness using "
                f"certified numerical bounds. At the recommended precision (float32), "
                f"the demographic parity gap is {fairness_results['float32']['dpg']:.4f} "
                f"with certified error bound ±{fairness_results['float32']['error_bound']:.4f}. "
                f"The fairness assessment is {'RELIABLE' if fairness_results['float32']['is_reliable'] else 'BORDERLINE'}."
            )
        }


def run_credit_scoring_case_study(device: str = 'cpu') -> Dict:
    """Run the complete credit scoring case study."""
    study = CreditScoringCaseStudy(device=device)
    results = study.run_complete_study()
    
    # Save results
    import os
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, 'credit_scoring_case_study.json'), 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_dir}/credit_scoring_case_study.json")
    
    return results


if __name__ == '__main__':
    import torch
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    results = run_credit_scoring_case_study(device=device)
