"""
Regulatory compliance and certification report generator.

This module generates detailed fairness certification reports suitable
for regulatory review, compliance documentation, and stakeholder communication.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import os

from .fairness_metrics import CertifiedFairnessEvaluator, FairnessResult
from .error_propagation import ErrorTracker
from .multi_metric_analysis import MultiMetricFairnessAnalyzer, MultiMetricResult


@dataclass
class CertificationReport:
    """Fairness certification report for regulatory compliance."""
    
    # Metadata
    report_id: str
    timestamp: str
    model_name: str
    dataset_name: str
    
    # Model specifications
    precision: str
    num_parameters: int
    architecture: str
    
    # Dataset specifications
    num_samples: int
    num_features: int
    group_name: str
    group_distribution: Dict[int, int]
    
    # Fairness metrics
    demographic_parity: Dict[str, float]
    equalized_odds: Dict[str, float]
    calibration: Dict[str, float]
    
    # Certification status
    certified: bool
    certification_level: str  # 'PASS', 'BORDERLINE', 'FAIL'
    reliability_score: float
    
    # Recommendations
    recommendations: List[str]
    warnings: List[str]
    
    # Technical details
    error_bounds: Dict[str, float]
    near_threshold_fraction: float
    threshold_used: float
    
    # Compliance statements
    compliance_summary: str
    technical_notes: str


class ComplianceCertifier:
    """
    Generates regulatory compliance certification reports.
    
    Produces detailed reports suitable for:
    - Regulatory review (e.g., GDPR Article 22)
    - Internal compliance documentation
    - Stakeholder communication
    - Audit trails
    """
    
    def __init__(
        self,
        certification_threshold: float = 2.0,
        borderline_threshold: float = 1.5
    ):
        """
        Initialize certifier.
        
        Args:
            certification_threshold: Minimum reliability score for PASS
            borderline_threshold: Minimum score for BORDERLINE (else FAIL)
        """
        self.cert_threshold = certification_threshold
        self.borderline_threshold = borderline_threshold
        
    def generate_certification_report(
        self,
        model: torch.nn.Module,
        X_test: torch.Tensor,
        y_test: torch.Tensor,
        groups: torch.Tensor,
        model_name: str = "Model",
        dataset_name: str = "Dataset",
        threshold: float = 0.5,
        precision: torch.dtype = torch.float32,
        architecture: str = "MLP"
    ) -> CertificationReport:
        """
        Generate comprehensive certification report.
        
        Args:
            model: Model to certify
            X_test: Test features
            y_test: Test labels
            groups: Group membership
            model_name: Name of model for report
            dataset_name: Name of dataset
            threshold: Classification threshold
            precision: Precision used for computation
            architecture: Model architecture description
            
        Returns:
            Certification report
        """
        # Analyze fairness
        tracker = ErrorTracker(precision=precision)
        analyzer = MultiMetricFairnessAnalyzer(tracker)
        
        result = analyzer.evaluate_all_metrics(
            model, X_test, y_test, groups, threshold
        )
        
        # Determine certification status
        if result.joint_reliable and result.joint_reliability_score >= self.cert_threshold:
            cert_level = "PASS"
            certified = True
        elif result.joint_reliability_score >= self.borderline_threshold:
            cert_level = "BORDERLINE"
            certified = False
        else:
            cert_level = "FAIL"
            certified = False
        
        # Generate recommendations
        recommendations = self._generate_recommendations(result, precision)
        warnings = self._generate_warnings(result)
        
        # Compute statistics
        num_params = sum(p.numel() for p in model.parameters())
        group_dist = {
            i: int((groups == i).sum().item())
            for i in range(int(groups.max().item()) + 1)
        }
        
        # Near-threshold analysis
        model.eval()
        with torch.no_grad():
            predictions = model(X_test)
            if predictions.dim() > 1:
                predictions = predictions.squeeze()
            near_threshold = ((predictions - threshold).abs() < result.demographic_parity.error_bound).float().mean().item()
        
        # Compliance summary
        compliance_summary = self._generate_compliance_summary(result, cert_level)
        technical_notes = self._generate_technical_notes(result)
        
        # Create report
        report = CertificationReport(
            report_id=f"CERT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            timestamp=datetime.now().isoformat(),
            model_name=model_name,
            dataset_name=dataset_name,
            precision=str(precision),
            num_parameters=num_params,
            architecture=architecture,
            num_samples=len(X_test),
            num_features=X_test.shape[1],
            group_name="Protected Group",
            group_distribution=group_dist,
            demographic_parity={
                'value': float(result.demographic_parity.metric_value),
                'error_bound': float(result.demographic_parity.error_bound),
                'reliable': bool(result.demographic_parity.is_reliable),
                'reliability_score': float(result.demographic_parity.reliability_score)
            },
            equalized_odds={
                'value': float(result.equalized_odds.metric_value),
                'error_bound': float(result.equalized_odds.error_bound),
                'reliable': bool(result.equalized_odds.is_reliable),
                'reliability_score': float(result.equalized_odds.reliability_score)
            },
            calibration={
                'value': float(result.calibration.metric_value),
                'error_bound': float(result.calibration.error_bound),
                'reliable': bool(result.calibration.is_reliable),
                'reliability_score': float(result.calibration.reliability_score)
            },
            certified=certified,
            certification_level=cert_level,
            reliability_score=float(result.joint_reliability_score),
            recommendations=recommendations,
            warnings=warnings,
            error_bounds={
                'dpg': float(result.demographic_parity.error_bound),
                'eog': float(result.equalized_odds.error_bound),
                'cal': float(result.calibration.error_bound)
            },
            near_threshold_fraction=float(near_threshold),
            threshold_used=float(threshold),
            compliance_summary=compliance_summary,
            technical_notes=technical_notes
        )
        
        return report
    
    def _generate_recommendations(
        self,
        result: MultiMetricResult,
        precision: torch.dtype
    ) -> List[str]:
        """Generate actionable recommendations."""
        recommendations = []
        
        if not result.joint_reliable:
            if precision == torch.float16:
                recommendations.append(
                    "CRITICAL: Increase precision to float32 for reliable fairness assessment"
                )
            elif precision == torch.float32:
                recommendations.append(
                    "WARNING: Consider float64 for higher precision fairness guarantees"
                )
        
        if result.demographic_parity.metric_value > 0.1:
            recommendations.append(
                f"Demographic parity gap ({result.demographic_parity.metric_value:.3f}) "
                "exceeds common fairness threshold of 0.1. Consider model retraining."
            )
        
        if result.equalized_odds.metric_value > 0.15:
            recommendations.append(
                f"Equalized odds gap ({result.equalized_odds.metric_value:.3f}) "
                "is high. Review model predictions for different groups."
            )
        
        if result.joint_reliability_score < 2.0:
            recommendations.append(
                "Joint reliability score is low. Many predictions near decision threshold."
            )
        
        if not recommendations:
            recommendations.append("Model meets fairness certification standards.")
        
        return recommendations
    
    def _generate_warnings(self, result: MultiMetricResult) -> List[str]:
        """Generate warnings about numerical reliability."""
        warnings = []
        
        if not result.demographic_parity.is_reliable:
            warnings.append(
                "Demographic parity measurement is numerically unreliable. "
                "Results may change with different precision."
            )
        
        if not result.equalized_odds.is_reliable:
            warnings.append(
                "Equalized odds measurement is numerically unreliable."
            )
        
        if not result.calibration.is_reliable:
            warnings.append(
                "Calibration measurement is numerically unreliable."
            )
        
        return warnings
    
    def _generate_compliance_summary(
        self,
        result: MultiMetricResult,
        level: str
    ) -> str:
        """Generate compliance summary statement."""
        if level == "PASS":
            return (
                f"This model PASSES fairness certification with reliability score "
                f"{result.joint_reliability_score:.2f}. All fairness metrics are "
                f"numerically reliable and within acceptable bounds. The model is "
                f"suitable for deployment with the tested configuration."
            )
        elif level == "BORDERLINE":
            return (
                f"This model receives BORDERLINE certification (score "
                f"{result.joint_reliability_score:.2f}). Some fairness measurements "
                f"are numerically borderline. Recommend reviewing precision settings "
                f"or model architecture before deployment."
            )
        else:
            return (
                f"This model FAILS fairness certification (score "
                f"{result.joint_reliability_score:.2f}). Fairness measurements are "
                f"numerically unreliable. Do NOT deploy without addressing precision "
                f"issues or retraining model."
            )
    
    def _generate_technical_notes(self, result: MultiMetricResult) -> str:
        """Generate technical notes for experts."""
        return (
            f"Numerical analysis based on Numerical Geometry framework. "
            f"Error bounds computed via linear error functionals tracking precision "
            f"propagation through model. Reliability scores indicate ratio of metric "
            f"value to error bound. Joint reliability requires all metrics to be "
            f"individually reliable."
        )
    
    def save_report_json(self, report: CertificationReport, filepath: str):
        """Save report to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(asdict(report), f, indent=2)
        print(f"Report saved to {filepath}")
    
    def save_report_html(self, report: CertificationReport, filepath: str):
        """Save report as HTML document."""
        html = self._generate_html_report(report)
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(html)
        print(f"HTML report saved to {filepath}")
    
    def _generate_html_report(self, report: CertificationReport) -> str:
        """Generate HTML report."""
        # Color coding
        if report.certification_level == "PASS":
            status_color = "#28a745"  # Green
            status_bg = "#d4edda"
        elif report.certification_level == "BORDERLINE":
            status_color = "#ffc107"  # Yellow
            status_bg = "#fff3cd"
        else:
            status_color = "#dc3545"  # Red
            status_bg = "#f8d7da"
        
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Fairness Certification Report - {report.model_name}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background-color: #2c3e50;
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .status-box {{
            background-color: {status_bg};
            border-left: 5px solid {status_color};
            padding: 20px;
            margin: 20px 0;
            border-radius: 5px;
        }}
        .status-badge {{
            display: inline-block;
            background-color: {status_color};
            color: white;
            padding: 10px 20px;
            border-radius: 5px;
            font-weight: bold;
            font-size: 18px;
        }}
        .section {{
            background-color: white;
            padding: 20px;
            margin: 20px 0;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 8px;
            background-color: #fafafa;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            text-transform: uppercase;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #f8f9fa;
            font-weight: bold;
        }}
        .recommendation {{
            background-color: #e3f2fd;
            border-left: 4px solid #2196f3;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .warning {{
            background-color: #fff3e0;
            border-left: 4px solid #ff9800;
            padding: 15px;
            margin: 10px 0;
            border-radius: 4px;
        }}
        .footer {{
            margin-top: 40px;
            padding: 20px;
            background-color: #ecf0f1;
            border-radius: 10px;
            font-size: 12px;
            color: #666;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔒 Fairness Certification Report</h1>
        <p><strong>Report ID:</strong> {report.report_id}</p>
        <p><strong>Generated:</strong> {report.timestamp}</p>
    </div>
    
    <div class="status-box">
        <h2>Certification Status</h2>
        <div class="status-badge">{report.certification_level}</div>
        <p style="margin-top: 15px;"><strong>Reliability Score:</strong> {report.reliability_score:.2f}/10</p>
        <p>{report.compliance_summary}</p>
    </div>
    
    <div class="section">
        <h2>📊 Model Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Model Name</td><td>{report.model_name}</td></tr>
            <tr><td>Architecture</td><td>{report.architecture}</td></tr>
            <tr><td>Parameters</td><td>{report.num_parameters:,}</td></tr>
            <tr><td>Precision</td><td>{report.precision}</td></tr>
            <tr><td>Dataset</td><td>{report.dataset_name}</td></tr>
            <tr><td>Samples</td><td>{report.num_samples:,}</td></tr>
            <tr><td>Features</td><td>{report.num_features}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>⚖️ Fairness Metrics</h2>
        <div class="metric-grid">
            <div class="metric-card">
                <div class="metric-label">Demographic Parity Gap</div>
                <div class="metric-value">{report.demographic_parity['value']:.4f} ± {report.demographic_parity['error_bound']:.4f}</div>
                <p>Reliable: {'✓ Yes' if report.demographic_parity['reliable'] else '✗ No'}</p>
                <p>Score: {report.demographic_parity['reliability_score']:.2f}</p>
            </div>
            <div class="metric-card">
                <div class="metric-label">Equalized Odds Gap</div>
                <div class="metric-value">{report.equalized_odds['value']:.4f} ± {report.equalized_odds['error_bound']:.4f}</div>
                <p>Reliable: {'✓ Yes' if report.equalized_odds['reliable'] else '✗ No'}</p>
                <p>Score: {report.equalized_odds['reliability_score']:.2f}</p>
            </div>
            <div class="metric-card">
                <div class="metric-label">Calibration Error</div>
                <div class="metric-value">{report.calibration['value']:.4f} ± {report.calibration['error_bound']:.4f}</div>
                <p>Reliable: {'✓ Yes' if report.calibration['reliable'] else '✗ No'}</p>
                <p>Score: {report.calibration['reliability_score']:.2f}</p>
            </div>
        </div>
    </div>
    
    <div class="section">
        <h2>💡 Recommendations</h2>
        {''.join(f'<div class="recommendation">• {rec}</div>' for rec in report.recommendations)}
    </div>
    
    {f'''<div class="section">
        <h2>⚠️ Warnings</h2>
        {''.join(f'<div class="warning">• {warn}</div>' for warn in report.warnings)}
    </div>''' if report.warnings else ''}
    
    <div class="section">
        <h2>🔬 Technical Details</h2>
        <p><strong>Near-Threshold Fraction:</strong> {report.near_threshold_fraction:.2%}</p>
        <p><strong>Classification Threshold:</strong> {report.threshold_used:.2f}</p>
        <p><strong>Group Distribution:</strong> {report.group_distribution}</p>
        <p><strong>Error Bounds:</strong></p>
        <ul>
            <li>DPG: ±{report.error_bounds['dpg']:.6f}</li>
            <li>EOG: ±{report.error_bounds['eog']:.6f}</li>
            <li>CAL: ±{report.error_bounds['cal']:.6f}</li>
        </ul>
        <p style="margin-top: 15px;"><em>{report.technical_notes}</em></p>
    </div>
    
    <div class="footer">
        <p><strong>NumGeom-Fair</strong> - Numerical Geometry of Fairness Metrics</p>
        <p>This report was generated using certified error bounds from the Numerical Geometry framework.</p>
        <p>For questions or concerns, contact your ML compliance team.</p>
    </div>
</body>
</html>
"""
        return html


if __name__ == "__main__":
    # Test certification report generator
    print("Testing Compliance Certifier...")
    
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
    
    # Generate report
    certifier = ComplianceCertifier()
    report = certifier.generate_certification_report(
        model, X, y, groups,
        model_name="Test MLP",
        dataset_name="Synthetic Fair Data",
        precision=torch.float32
    )
    
    print(f"\nReport ID: {report.report_id}")
    print(f"Certification: {report.certification_level}")
    print(f"Reliability Score: {report.reliability_score:.2f}")
    print(f"\nRecommendations:")
    for rec in report.recommendations:
        print(f"  - {rec}")
    
    # Save reports
    os.makedirs("data/compliance_reports", exist_ok=True)
    certifier.save_report_json(report, "data/compliance_reports/test_report.json")
    certifier.save_report_html(report, "data/compliance_reports/test_report.html")
    
    print("\n✓ Reports saved!")
