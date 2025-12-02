# HNF Implementation Proposals for Transformers

This folder contains detailed proposals for 10 projects that translate Homotopy Numerical Foundations theory into practical tools for **training, evaluating, using, and fine-tuning transformers**.

## Projects

| # | Project | Transformer Application |
|---|---------|------------------------|
| 1 | [Precision-Aware Autodiff](./01_precision_aware_ad.md) | Debug precision loss during transformer fine-tuning |
| 2 | [Sheaf Mixed-Precision](./02_sheaf_mixed_precision.md) | Optimal mixed-precision configs for transformer training |
| 3 | [Attention Stability Analysis](./03_attention_stability.md) | Analyze/fix unstable attention patterns |
| 4 | [Stability-Preserving Rewriter](./04_stability_rewriter.md) | Auto-fuse and stabilize transformer graphs |
| 5 | [Condition Profiler](./05_condition_profiler.md) | Predict loss spikes during LLM training |
| 6 | [Certified Precision Bounds](./06_certified_bounds.md) | Know min precision for transformer deployment |
| 7 | [Curvature-Adaptive LR](./07_homotopy_lr.md) | Geometry-based learning rate scheduling |
| 8 | [KV-Cache Precision Analyzer](./08_kv_cache_precision.md) | Optimize memory vs. quality for long-context inference |
| 9 | [Curvature-Guided Quantization](./09_precision_quantization.md) | Per-layer quantization for transformer deployment |
| 10 | [Stability Linter](./10_stability_linter.md) | Lint transformer code for numerical bugs |

## Roadmap

See [ROADMAP.md](./ROADMAP.md) for prioritization and dependencies.

## Quick Start

Recommended starting points:
- **Project 10** (Linter): Catch attention/LayerNorm bugs before training
- **Project 5** (Profiler): Predict and prevent LLM training instabilities
- **Project 8** (KV-Cache): Enable longer context windows with same memory

