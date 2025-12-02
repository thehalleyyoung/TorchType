# Quick Start Guide: HNF Curvature Profiling

Get started with curvature-aware training in 5 minutes!

## Installation

```bash
cd /path/to/TorchType/src/implementations/proposal5
./build.sh
```

Requirements:
- LibTorch 2.0+
- C++17 compiler
- Eigen 3.4+ (auto-detected from proposal2)

## Minimal Example

```cpp
#include "curvature_profiler.hpp"
#include <torch/torch.h>

int main() {
    // Your model
    auto model = torch::nn::Sequential(
        torch::nn::Linear(784, 256),
        torch::nn::ReLU(),
        torch::nn::Linear(256, 10)
    );
    
    // Create profiler
    hnf::profiler::CurvatureProfiler profiler(*model);
    
    // Track layers
    profiler.track_layer_shared("fc1", model->ptr(0));
    profiler.track_layer_shared("fc2", model->ptr(2));
    
    // Training loop
    for (int step = 0; step < 100; ++step) {
        auto x = torch::randn({32, 784});
        auto y = torch::randint(0, 10, {32});
        
        auto output = model->forward(x);
        auto loss = torch::nn::functional::cross_entropy(output, y);
        
        // Compute curvature (do this periodically, not every step!)
        if (step % 10 == 0) {
            auto metrics = profiler.compute_curvature(loss, step);
            
            for (const auto& [name, m] : metrics) {
                std::cout << "Layer " << name 
                          << ": κ=" << m.kappa_curv 
                          << ", L=" << m.lipschitz_constant << std::endl;
            }
        }
        
        loss.backward();
        // optimizer.step();
    }
    
    // Export to CSV for plotting
    profiler.export_to_csv("curvature_history.csv");
    
    return 0;
}
```

## With Training Monitor

Add automatic warnings and LR suggestions:

```cpp
#include "curvature_profiler.hpp"

int main() {
    auto model = /* your model */;
    hnf::profiler::CurvatureProfiler profiler(*model);
    
    // Configure monitor
    hnf::profiler::TrainingMonitor::Config config;
    config.warning_threshold = 1e6;   // Warn when κ > 10⁶
    config.danger_threshold = 1e9;    // Danger when κ > 10⁹
    
    hnf::profiler::TrainingMonitor monitor(profiler, config);
    
    for (int step = 0; step < num_steps; ++step) {
        auto loss = /* compute loss */;
        
        // Check for issues
        auto warnings = monitor.on_step(loss, step);
        
        if (!warnings.empty()) {
            for (const auto& w : warnings) {
                std::cout << "[Step " << step << "] " << w << std::endl;
            }
            
            // Suggested action
            if (monitor.is_danger_state()) {
                double lr_scale = monitor.suggest_lr_adjustment();
                std::cout << "→ Reduce LR by " << lr_scale << "x" << std::endl;
                // Actually reduce it:
                for (auto& pg : optimizer.param_groups()) {
                    pg.options().lr() *= lr_scale;
                }
            }
        }
        
        loss.backward();
        optimizer.step();
    }
    
    return 0;
}
```

## With Adaptive Learning Rate

Let curvature automatically control LR:

```cpp
#include "curvature_profiler.hpp"

int main() {
    auto model = /* your model */;
    hnf::profiler::CurvatureProfiler profiler(*model);
    // ... track layers ...
    
    // Configure adaptive LR
    hnf::profiler::CurvatureAdaptiveLR::Config lr_config;
    lr_config.base_lr = 0.01;
    lr_config.target_curvature = 1e4;  // Try to keep κ ≈ 10⁴
    
    hnf::profiler::CurvatureAdaptiveLR scheduler(profiler, lr_config);
    
    torch::optim::SGD optimizer(model->parameters(), 
                                 torch::optim::SGDOptions(lr_config.base_lr));
    
    for (int step = 0; step < num_steps; ++step) {
        auto loss = /* compute loss */;
        
        // Profiler updates curvature
        profiler.compute_curvature(loss, step);
        
        // Scheduler adjusts LR based on curvature
        scheduler.step(optimizer, step);
        
        loss.backward();
        optimizer.step();
    }
    
    return 0;
}
```

## Running Examples

```bash
cd build/

# Basic functionality test
./test_profiler

# Comprehensive theoretical validation
./test_comprehensive

# In-depth rigorous tests
./test_rigorous

# MNIST with full HNF analysis
./mnist_complete_validation

# See precision requirements
./mnist_precision

# Simple training demo
./simple_training
```

## Understanding the Output

### Curvature Metrics

When you call `compute_curvature()`, you get:

```cpp
struct CurvatureMetrics {
    double kappa_curv;           // κ^{curv} = (1/2)||D²f||_op
    double lipschitz_constant;   // L_f = ||Df||_op
    double spectral_norm_hessian;// ||D²f||_op
    double gradient_norm;        // ||∇f||
    double condition_number;     // For layer operations
    
    // Precision requirement from Theorem 4.7
    double required_mantissa_bits(double diameter, double eps);
};
```

**Interpreting κ^{curv}**:
- κ < 10³: Very stable, low curvature
- 10³ < κ < 10⁶: Moderate curvature, watch it
- 10⁶ < κ < 10⁹: High curvature, potential issues
- κ > 10⁹: Dangerous, likely to cause instability

### Precision Requirements

```cpp
auto metrics = profiler.compute_curvature(loss, step);
double required_bits = metrics["fc1"].required_mantissa_bits(
    /*diameter=*/2.0,      // Domain diameter
    /*target_eps=*/1e-6    // Target accuracy
);

// Formula: p ≥ log₂(κ·D²/ε)
std::cout << "Required: " << required_bits << " bits\n";
std::cout << "fp16 has: 10 bits (insufficient)\n";
std::cout << "fp32 has: 23 bits (sufficient!)\n";
std::cout << "fp64 has: 52 bits (overkill)\n";
```

### Monitor Warnings

```
[Step 1234] WARNING: Layer 'attention.softmax' curvature 1.5e7 exceeds 1.0e6
[Step 1235] PREDICTION: Training likely to fail in 100 steps. Layer 'attention.softmax' projected curvature: 5.2e12
[Step 1235] → Recommended: reduce LR by 0.5x for next 100 steps
```

## Tips & Best Practices

### 1. Sample Periodically

Don't profile every step (too expensive). Sample every 10-100 steps:

```cpp
if (step % 10 == 0) {
    profiler.compute_curvature(loss, step);
}
```

### 2. Track Critical Layers Only

For large models, track only critical layers:

```cpp
// In transformers, track:
profiler.track_layer("attention.softmax", attn_layer);
profiler.track_layer("ffn.up_proj", ffn_up);

// Skip less critical layers
```

### 3. Export for Analysis

```cpp
profiler.export_to_csv("metrics.csv");

// Then plot with Python:
// import pandas as pd
// import matplotlib.pyplot as plt
// df = pd.read_csv("metrics.csv")
// df[df.layer == "fc1"].plot(x="step", y="kappa_curv")
```

### 4. Combine with Regular Monitoring

```cpp
// Standard monitoring
if (step % 100 == 0) {
    std::cout << "Loss: " << loss.item<double>() << std::endl;
}

// Curvature monitoring (less frequent)
if (step % 1000 == 0) {
    auto metrics = profiler.compute_curvature(loss, step);
    // Check for issues...
}
```

### 5. Use Thresholds Appropriate to Your Model

For small models (< 10M params):
```cpp
config.warning_threshold = 1e5;
config.danger_threshold = 1e8;
```

For large models (> 100M params):
```cpp
config.warning_threshold = 1e7;
config.danger_threshold = 1e10;
```

## Troubleshooting

### "Curvature is always 0"

Check that:
1. Loss requires grad: `loss.requires_grad() == true`
2. Layers have parameters: `layer->parameters().size() > 0`
3. Using `compute_curvature()` correctly with scalar loss

### "Computation is too slow"

Try:
1. Profile less frequently (every 100 steps instead of 10)
2. Track fewer layers
3. Use approximate mode (if available)

### "Warnings but no actual failures"

This is normal! Warnings are *predictive* - they tell you about potential issues. If you see warnings:
1. Check if training is getting unstable (loss oscillating)
2. Consider reducing LR preemptively
3. Monitor for 10-50 more steps

### "Precision requirement says >64 bits needed"

This means:
1. Your problem is numerically challenging
2. Standard fp64 may not suffice
3. Consider:
   - Reformulating the problem
   - Using higher precision libraries
   - Redesigning the architecture

## Next Steps

Once you're comfortable with basic profiling:

1. **Read the full README.md** for advanced features
2. **Check IMPLEMENTATION_STATUS.md** for what's available
3. **Look at mnist_complete_validation.cpp** for complete example
4. **Explore visualization.hpp** for plotting tools

## Support

This is part of the TorchType/HNF project. See parent repository for:
- Full documentation
- Theoretical background (hnf_paper.tex)
- Other proposals

## Quick Reference

```cpp
// Essential includes
#include "curvature_profiler.hpp"
#include "visualization.hpp"

// Essential classes
hnf::profiler::CurvatureProfiler      // Main profiler
hnf::profiler::TrainingMonitor        // Warnings & predictions
hnf::profiler::CurvatureAdaptiveLR    // Adaptive learning rate
hnf::profiler::ExactHessianComputer   // Rigorous analysis (small models)

// Key methods
profiler.track_layer(name, module)
profiler.compute_curvature(loss, step)
profiler.get_history(layer_name)
profiler.export_to_csv(filename)

monitor.on_step(loss, step)
monitor.predict_failure()
monitor.suggest_lr_adjustment()

scheduler.compute_lr(step)
scheduler.step(optimizer, step)
```

---

**That's it!** You're ready to use HNF curvature profiling to make your training more stable and theoretically principled. 🎉
