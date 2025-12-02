"""
Interactive Fairness Certification Dashboard

Extension 1: Real-time fairness assessment with visual feedback.
Provides practitioners with immediate insight into numerical reliability.
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from dataclasses import dataclass, asdict

try:
    from .fairness_metrics import FairnessMetrics, CertifiedFairnessEvaluator, FairnessResult
    from .error_propagation import ErrorTracker, LinearErrorFunctional
    from .curvature_analysis import CurvatureAnalyzer
except ImportError:
    from fairness_metrics import FairnessMetrics, CertifiedFairnessEvaluator, FairnessResult
    from error_propagation import ErrorTracker, LinearErrorFunctional
    from curvature_analysis import CurvatureAnalyzer


@dataclass
class DashboardReport:
    """Comprehensive fairness report"""
    model_name: str
    dataset_name: str
    timestamp: str
    
    # Main fairness metrics
    demographic_parity: Dict[str, any]
    equalized_odds: Dict[str, any]
    calibration: Dict[str, any]
    
    # Numerical analysis
    precision_recommendation: Dict[str, any]
    curvature_analysis: Dict[str, any]
    threshold_stability: Dict[str, any]
    
    # Reliability summary
    overall_reliability: str  # "Reliable", "Borderline", "Unreliable"
    warnings: List[str]
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)
    
    def to_json(self, filepath: str):
        """Save report as JSON"""
        import json
        
        def convert_types(obj):
            """Convert numpy types to Python types"""
            if isinstance(obj, np.bool_):
                return bool(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, dict):
                return {k: convert_types(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_types(v) for v in obj]
            return obj
        
        data = convert_types(self.to_dict())
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def to_html(self, filepath: str):
        """Generate HTML dashboard"""
        html = self._generate_html()
        with open(filepath, 'w') as f:
            f.write(html)
    
    def _generate_html(self) -> str:
        """Generate HTML dashboard"""
        # Color coding
        reliability_colors = {
            'Reliable': '#28a745',
            'Borderline': '#ffc107',
            'Unreliable': '#dc3545'
        }
        
        color = reliability_colors.get(self.overall_reliability, '#6c757d')
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Fairness Certification Report - {self.model_name}</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{
            margin: 0;
            font-size: 32px;
        }}
        .header p {{
            margin: 5px 0 0 0;
            opacity: 0.9;
        }}
        .reliability-badge {{
            display: inline-block;
            background-color: {color};
            color: white;
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 14px;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            margin-bottom: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .section h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 15px;
            margin-top: 15px;
        }}
        .metric-card {{
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            background: #fafafa;
        }}
        .metric-card h3 {{
            margin: 0 0 10px 0;
            font-size: 16px;
            color: #555;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #667eea;
        }}
        .metric-detail {{
            font-size: 12px;
            color: #777;
            margin-top: 5px;
        }}
        .warning {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .recommendation {{
            background-color: #d1ecf1;
            border-left: 4px solid #0c5460;
            padding: 12px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .data-table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        .data-table th, .data-table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #e0e0e0;
        }}
        .data-table th {{
            background-color: #f5f5f5;
            font-weight: 600;
        }}
        .check {{
            color: #28a745;
            font-weight: bold;
        }}
        .cross {{
            color: #dc3545;
            font-weight: bold;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Fairness Certification Report</h1>
        <p><strong>Model:</strong> {self.model_name} | <strong>Dataset:</strong> {self.dataset_name}</p>
        <p><strong>Generated:</strong> {self.timestamp}</p>
        <div class="reliability-badge">{self.overall_reliability}</div>
    </div>
    
    <div class="section">
        <h2>📊 Fairness Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <h3>Demographic Parity Gap</h3>
                <div class="metric-value">{self.demographic_parity['value']:.4f}</div>
                <div class="metric-detail">
                    Error Bound: ±{self.demographic_parity['error_bound']:.4f}<br>
                    Reliability: {self.demographic_parity['reliability_score']:.2f}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Equalized Odds Gap</h3>
                <div class="metric-value">{self.equalized_odds['value']:.4f}</div>
                <div class="metric-detail">
                    Error Bound: ±{self.equalized_odds['error_bound']:.4f}<br>
                    Reliability: {self.equalized_odds['reliability_score']:.2f}
                </div>
            </div>
            
            <div class="metric-card">
                <h3>Calibration Error</h3>
                <div class="metric-value">{self.calibration['value']:.4f}</div>
                <div class="metric-detail">
                    Bins analyzed: {self.calibration['n_bins']}<br>
                    Uncertain bins: {self.calibration['uncertain_bins']}
                </div>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>🎯 Precision Recommendation</h2>
        <table class="data-table">
            <tr>
                <th>Precision Type</th>
                <th>Recommended</th>
                <th>Predicted Error</th>
                <th>Sufficient?</th>
            </tr>
            <tr>
                <td><strong>{self.precision_recommendation['recommended_dtype']}</strong></td>
                <td>{self.precision_recommendation['recommended_bits']} bits</td>
                <td>{self.precision_recommendation['predicted_error']:.2e}</td>
                <td><span class="{'check' if self.precision_recommendation['is_sufficient'] else 'cross'}">
                    {'✓ Yes' if self.precision_recommendation['is_sufficient'] else '✗ No'}
                </span></td>
            </tr>
        </table>
        <p style="margin-top: 15px; color: #555;">
            <strong>Safety Margin:</strong> {self.precision_recommendation['safety_margin']:.2f}x
        </p>
    </div>
    
    <div class="section">
        <h2>📈 Curvature Analysis</h2>
        <p><strong>Maximum Curvature:</strong> {self.curvature_analysis['max_curvature']:.6f}</p>
        <p><strong>Average Curvature:</strong> {self.curvature_analysis['avg_curvature']:.6f}</p>
        <p style="color: #666; font-size: 14px; margin-top: 10px;">
            Higher curvature indicates greater sensitivity to numerical errors.
            Our analysis shows the model has {'low' if self.curvature_analysis['max_curvature'] < 0.01 else 'moderate' if self.curvature_analysis['max_curvature'] < 0.1 else 'high'} curvature.
        </p>
    </div>
    
    <div class="section">
        <h2>🎚️ Threshold Stability</h2>
        <p><strong>Stable Range:</strong> [{self.threshold_stability['stable_min']:.3f}, {self.threshold_stability['stable_max']:.3f}]</p>
        <p><strong>Current Threshold:</strong> {self.threshold_stability['current_threshold']:.3f}</p>
        <p style="color: #666; font-size: 14px; margin-top: 10px;">
            Fairness metrics are numerically stable within the indicated threshold range.
            {'✓ Current threshold is in stable region.' if self.threshold_stability['is_stable'] else '⚠️ Warning: Current threshold is in unstable region!'}
        </p>
    </div>
"""
        
        if self.warnings:
            html += """
    <div class="section">
        <h2>⚠️ Warnings</h2>
"""
            for warning in self.warnings:
                html += f'        <div class="warning">⚠️ {warning}</div>\n'
            html += "    </div>\n"
        
        if self.recommendations:
            html += """
    <div class="section">
        <h2>💡 Recommendations</h2>
"""
            for rec in self.recommendations:
                html += f'        <div class="recommendation">💡 {rec}</div>\n'
            html += "    </div>\n"
        
        html += """
    <div class="section">
        <h2>📝 About This Report</h2>
        <p>
            This report was generated using <strong>NumGeom-Fair</strong>, a framework for
            certified fairness assessment under finite-precision arithmetic. All error bounds
            are mathematically certified using the Numerical Geometry framework.
        </p>
        <p style="margin-top: 10px; font-size: 14px; color: #666;">
            For more information, see: <em>Numerical Geometry of Fairness Metrics: When Does Precision Affect Equity?</em>
        </p>
    </div>
</body>
</html>
"""
        return html


class FairnessDashboard:
    """
    Interactive dashboard for fairness certification.
    """
    
    def __init__(self):
        self.curvature_analyzer = CurvatureAnalyzer()
    
    def generate_report(self,
                       model: torch.nn.Module,
                       X: torch.Tensor,
                       y: np.ndarray,
                       groups: np.ndarray,
                       threshold: float = 0.5,
                       precision: torch.dtype = torch.float32,
                       model_name: str = "Model",
                       dataset_name: str = "Dataset") -> DashboardReport:
        """
        Generate comprehensive fairness report.
        
        Args:
            model: Neural network model
            X: Input features
            y: True labels
            groups: Group membership (0 or 1)
            threshold: Decision threshold
            precision: Numerical precision to analyze
            model_name: Name for the model
            dataset_name: Name for the dataset
            
        Returns:
            DashboardReport with all analysis
        """
        import datetime
        
        # Create error tracker and evaluator
        error_tracker = ErrorTracker(precision=precision)
        evaluator = CertifiedFairnessEvaluator(
            error_tracker=error_tracker,
            reliability_threshold=2.0
        )
        
        # Evaluate demographic parity
        dpg_result = evaluator.evaluate_demographic_parity(
            model, X, groups, threshold
        )
        
        # Evaluate equalized odds
        eog_result = evaluator.evaluate_equalized_odds(
            model, X, y, groups, threshold
        )
        
        # Evaluate calibration
        cal_result = evaluator.evaluate_calibration(
            model, X, y
        )
        
        # Curvature analysis
        curvature_rec = self.curvature_analyzer.recommend_precision_for_fairness(
            model, X[:20], target_dpg_error=0.01
        )
        
        # Threshold stability analysis
        threshold_stability = self._analyze_threshold_stability(
            model, X, groups, threshold
        )
        
        # Determine overall reliability
        cal_reliable = sum(cal_result['reliable_bins']) / len(cal_result['reliable_bins']) > 0.7
        all_reliable = (dpg_result.is_reliable and 
                       eog_result.is_reliable and 
                       cal_reliable)
        any_unreliable = (not dpg_result.is_reliable or 
                         not eog_result.is_reliable or 
                         not cal_reliable)
        
        if all_reliable:
            overall_reliability = "Reliable"
        elif any_unreliable and not all_reliable:
            overall_reliability = "Borderline"
        else:
            overall_reliability = "Unreliable"
        
        # Generate warnings
        warnings = []
        if dpg_result.reliability_score < 2.0:
            warnings.append(
                f"Demographic parity gap ({dpg_result.metric_value:.4f}) has low reliability score ({dpg_result.reliability_score:.2f}). "
                f"Consider using higher precision or different threshold."
            )
        if eog_result.reliability_score < 2.0:
            warnings.append(
                f"Equalized odds gap has borderline numerical reliability."
            )
        if not threshold_stability['is_stable']:
            warnings.append(
                f"Current threshold {threshold:.3f} is outside stable region [{threshold_stability['stable_min']:.3f}, {threshold_stability['stable_max']:.3f}]"
            )
        if not curvature_rec['is_sufficient']:
            warnings.append(
                f"Current precision ({precision}) may be insufficient. Recommended: {curvature_rec['recommended_dtype']}"
            )
        
        # Generate recommendations
        recommendations = []
        if not curvature_rec['is_sufficient']:
            recommendations.append(
                f"Use {curvature_rec['recommended_dtype']} precision for reliable fairness assessment."
            )
        if not threshold_stability['is_stable']:
            recommendations.append(
                f"Consider using threshold in range [{threshold_stability['stable_min']:.3f}, {threshold_stability['stable_max']:.3f}] for more stable fairness metrics."
            )
        if dpg_result.near_threshold_fraction['group_0'] > 0.2 or dpg_result.near_threshold_fraction['group_1'] > 0.2:
            recommendations.append(
                "High concentration of predictions near threshold. Consider retraining with different regularization or threshold."
            )
        
        return DashboardReport(
            model_name=model_name,
            dataset_name=dataset_name,
            timestamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            demographic_parity={
                'value': dpg_result.metric_value,
                'error_bound': dpg_result.error_bound,
                'reliability_score': dpg_result.reliability_score,
                'is_reliable': dpg_result.is_reliable
            },
            equalized_odds={
                'value': eog_result.metric_value,
                'error_bound': eog_result.error_bound,
                'reliability_score': eog_result.reliability_score,
                'is_reliable': eog_result.is_reliable
            },
            calibration={
                'value': cal_result['ece'],
                'error_bound': np.mean(cal_result['bin_uncertainties']),
                'n_bins': len(cal_result['bin_uncertainties']),
                'uncertain_bins': sum(1 for r in cal_result['reliable_bins'] if not r)
            },
            precision_recommendation=curvature_rec,
            curvature_analysis={
                'max_curvature': curvature_rec['max_curvature'],
                'avg_curvature': curvature_rec['avg_curvature']
            },
            threshold_stability=threshold_stability,
            overall_reliability=overall_reliability,
            warnings=warnings,
            recommendations=recommendations
        )
    
    def _analyze_threshold_stability(self,
                                    model: torch.nn.Module,
                                    X: torch.Tensor,
                                    groups: np.ndarray,
                                    current_threshold: float) -> Dict[str, any]:
        """Analyze stability of fairness metrics across threshold choices"""
        thresholds = np.linspace(0.2, 0.8, 20)
        dpgs = []
        
        model.eval()
        with torch.no_grad():
            predictions = model(X).cpu().numpy().flatten()
        
        for t in thresholds:
            dpg = FairnessMetrics.demographic_parity_gap(
                predictions, groups, t
            )
            dpgs.append(dpg)
        
        # Find stable region (low variation)
        window_size = 5
        variations = []
        for i in range(len(dpgs) - window_size):
            window = dpgs[i:i+window_size]
            var = np.std(window)
            variations.append(var)
        
        # Stable region: variation < threshold
        stable_threshold = 0.02
        stable_indices = [i for i, v in enumerate(variations) if v < stable_threshold]
        
        if stable_indices:
            stable_min = thresholds[stable_indices[0]]
            stable_max = thresholds[min(stable_indices[-1] + window_size, len(thresholds) - 1)]
            is_stable = stable_min <= current_threshold <= stable_max
        else:
            stable_min = 0.5
            stable_max = 0.5
            is_stable = False
        
        return {
            'stable_min': stable_min,
            'stable_max': stable_max,
            'current_threshold': current_threshold,
            'is_stable': is_stable
        }


def demo_dashboard():
    """Demonstrate dashboard generation"""
    print("="*80)
    print("FAIRNESS DASHBOARD DEMO")
    print("="*80)
    
    # Create model and data
    torch.manual_seed(42)
    np.random.seed(42)
    
    model = torch.nn.Sequential(
        torch.nn.Linear(5, 10),
        torch.nn.ReLU(),
        torch.nn.Linear(10, 1),
        torch.nn.Sigmoid()
    )
    
    n = 200
    X = torch.randn(n, 5)
    y = np.random.randint(0, 2, n)
    groups = np.array([0]*(n//2) + [1]*(n//2))
    
    # Generate dashboard
    dashboard = FairnessDashboard()
    report = dashboard.generate_report(
        model, X, y, groups,
        threshold=0.5,
        precision=torch.float32,
        model_name="Demo MLP",
        dataset_name="Synthetic Tabular"
    )
    
    print(f"\n✅ Generated report for {report.model_name}")
    print(f"   Overall Reliability: {report.overall_reliability}")
    print(f"   Warnings: {len(report.warnings)}")
    print(f"   Recommendations: {len(report.recommendations)}")
    
    # Save outputs
    import tempfile
    import os
    
    temp_dir = tempfile.mkdtemp()
    json_path = os.path.join(temp_dir, "fairness_report.json")
    html_path = os.path.join(temp_dir, "fairness_report.html")
    
    report.to_json(json_path)
    report.to_html(html_path)
    
    print(f"\n📄 Saved JSON report: {json_path}")
    print(f"🌐 Saved HTML dashboard: {html_path}")
    print(f"\n   Open {html_path} in a browser to view the interactive dashboard!")
    
    print("\n✓ Dashboard demo complete!")


if __name__ == '__main__':
    demo_dashboard()
