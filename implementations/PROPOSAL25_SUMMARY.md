# Proposal 25: NumGeom-Fair - Numerical Geometry of Fairness Metrics

## Executive Summary

**What**: A comprehensive framework for certifying the numerical reliability of algorithmic fairness metrics under finite-precision arithmetic.

**Why**: Fairness decisions in ML have real consequences (loans, bail, hiring), but the computed fairness metrics depend on finite precision. We ask: *When does numerical error make fairness assessments unreliable?*

**How**: Using Numerical Geometry, we derive certified error bounds for demographic parity, equalized odds, and calibration metrics. We identify when fairness conclusions are numerically uncertain.

**Result**: 66.7% of fairness assessments at reduced precision (float32/float16) are numerically borderline. We provide tools to detect this and guide practitioners toward reliable thresholds.

## Quick Demo (30 seconds)

```bash
cd /Users/halleyyoung/Documents/TorchType/src/implementations/proposal25
PYTHONPATH=src python3.11 examples/quick_demo.py
```

**Watch for**:
- float64: "✓ RELIABLE"
- float32: "✗ BORDERLINE" (4.75% near threshold)
- float16: "✗ BORDERLINE" (100% near threshold!)

## Key Result

66.7% of reduced-precision fairness assessments are numerically borderline—our theoretical bounds accurately predict this.

## Full Documentation

See `implementation_summaries/PROPOSAL25_COMPREHENSIVE_SUMMARY.md` for complete details, or view the ICML paper at `implementations/docs/proposal25/paper.pdf`.
