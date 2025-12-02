"""
NumGeom-Fair: Numerical Geometry of Fairness Metrics

This package implements proposal 25 from the HNF framework, studying how
finite-precision arithmetic affects algorithmic fairness metrics and providing
certified bounds on fairness assessments.
"""

try:
    from .error_propagation import ErrorTracker, LinearErrorFunctional
    from .fairness_metrics import (
        FairnessMetrics,
        CertifiedFairnessEvaluator,
        ThresholdStabilityAnalyzer
    )
    from .models import FairMLPClassifier, train_fair_classifier
    from .datasets import (
        load_adult_income,
        generate_synthetic_compas,
        generate_synthetic_tabular
    )
except ImportError:
    from error_propagation import ErrorTracker, LinearErrorFunctional
    from fairness_metrics import (
        FairnessMetrics,
        CertifiedFairnessEvaluator,
        ThresholdStabilityAnalyzer
    )
    from models import FairMLPClassifier, train_fair_classifier
    from datasets import (
        load_adult_income,
        generate_synthetic_compas,
        generate_synthetic_tabular
    )

__version__ = "1.0.0"
__all__ = [
    "ErrorTracker",
    "LinearErrorFunctional",
    "FairnessMetrics",
    "CertifiedFairnessEvaluator",
    "ThresholdStabilityAnalyzer",
    "FairMLPClassifier",
    "load_adult_income",
    "generate_synthetic_compas",
    "generate_synthetic_tabular",
]
