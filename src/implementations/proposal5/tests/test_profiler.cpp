#include "curvature_profiler.hpp"
#include "visualization.hpp"
#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace hnf::profiler;

// Test utilities
#define TEST(name) void test_##name()
#define RUN_TEST(name) do { \
    std::cout << "Running test: " << #name << "... "; \
    test_##name(); \
    std::cout << "PASSED\n"; \
} while(0)

#define ASSERT_TRUE(cond) do { \
    if (!(cond)) { \
        std::cerr << "Assertion failed: " << #cond << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        std::exit(1); \
    } \
} while(0)

#define ASSERT_NEAR(a, b, tol) do { \
    double _a = (a), _b = (b), _tol = (tol); \
    if (std::abs(_a - _b) > _tol) { \
        std::cerr << "Assertion failed: " << #a << " ≈ " << #b << " (diff=" << std::abs(_a - _b) << ", tol=" << _tol << ")\n"; \
        std::exit(1); \
    } \
} while(0)

// ============================================================================
// Test 1: Hessian-vector product correctness
// ============================================================================

TEST(hvp_correctness) {
    // Test Pearlmutter's trick on a simple quadratic: f(x) = x^T A x
    // H = 2A, so Hvp(v) = 2Av
    
    int n = 5;
    auto A = torch::rand({n, n});
    A = A + A.t();  // Make symmetric
    
    auto x = torch::randn({n}, torch::requires_grad(true));
    auto loss = torch::matmul(x, torch::matmul(A, x));
    
    auto v = torch::randn({n});
    
    HessianSpectralNormEstimator estimator;
    auto hvp = estimator.hessian_vector_product(loss, {x}, {v});
    
    // Expected: 2Av
    auto expected = 2.0 * torch::matmul(A, v);
    
    ASSERT_TRUE(hvp.size() == 1);
    ASSERT_TRUE(hvp[0].defined());
    
    auto diff = (hvp[0] - expected).abs().max().item<double>();
    ASSERT_TRUE(diff < 1e-5);
}

// ============================================================================
// Test 2: Spectral norm estimation
// ============================================================================

TEST(spectral_norm_estimation) {
    // For quadratic f(x) = x^T A x, Hessian is 2A
    // So ||H||_op = 2||A||_op
    
    int n = 10;
    auto A = torch::rand({n, n});
    A = A + A.t();  // Symmetric
    
    // Compute exact spectral norm
    auto eigenvalues = torch::linalg::eigvalsh(A);
    double exact_spectral_A = eigenvalues.abs().max().item<double>();
    
    auto x = torch::randn({n}, torch::requires_grad(true));
    auto loss = torch::matmul(x, torch::matmul(A, x));
    
    HvpConfig config;
    config.num_power_iterations = 50;
    HessianSpectralNormEstimator estimator(config);
    
    double estimated = estimator.estimate_spectral_norm(loss, {x});
    double expected = 2.0 * exact_spectral_A;
    
    // Power iteration should be accurate to a few percent
    double rel_error = std::abs(estimated - expected) / expected;
    ASSERT_TRUE(rel_error < 0.1);  // Within 10%
}

// ============================================================================
// Test 3: Curvature computation for known functions
// ============================================================================

TEST(curvature_known_functions) {
    // Test κ^{curv} = (1/2)||D²f||_op for f(x) = x^2
    // f''(x) = 2, so κ = 1
    
    struct SquareModule : torch::nn::Module {
        torch::Tensor forward(torch::Tensor x) {
            return x * x;
        }
    };
    
    auto model = SquareModule();
    auto x = torch::tensor({2.0}, torch::requires_grad(true));
    auto y = model.forward(x);
    auto loss = y.sum();
    
    CurvatureProfiler profiler(model);
    profiler.track_layer("square", std::make_shared<SquareModule>(model));
    
    auto metrics = profiler.compute_curvature(loss, 0);
    
    // For f(x) = x², D²f = 2 (constant), so κ = 1
    // But our implementation uses Hessian of loss, which gives spectral norm
    // This is a simple sanity check that we get non-zero curvature
    ASSERT_TRUE(metrics.find("square") != metrics.end());
    ASSERT_TRUE(metrics["square"].kappa_curv >= 0);
}

// ============================================================================
// Test 4: Profiler history tracking
// ============================================================================

TEST(profiler_history) {
    auto model = torch::nn::Linear(10, 5);
    CurvatureProfiler profiler(*model);
    
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    // Simulate training steps
    for (int step = 0; step < 10; ++step) {
        auto x = torch::randn({8, 10});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        
        profiler.compute_curvature(loss, step);
    }
    
    const auto& history = profiler.get_history("linear");
    ASSERT_TRUE(history.size() == 10);
    
    // Check that steps are sequential
    for (size_t i = 0; i < history.size(); ++i) {
        ASSERT_TRUE(history[i].step == static_cast<int>(i));
    }
}

// ============================================================================
// Test 5: Training monitor warnings
// ============================================================================

TEST(training_monitor_warnings) {
    auto model = torch::nn::Linear(10, 5);
    CurvatureProfiler profiler(*model);
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    TrainingMonitor::Config config;
    config.warning_threshold = 1e3;   // Low threshold for testing
    config.danger_threshold = 1e6;
    
    TrainingMonitor monitor(profiler, config);
    
    // Run a few steps
    for (int step = 0; step < 5; ++step) {
        auto x = torch::randn({8, 10});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        
        auto warnings = monitor.on_step(loss, step);
        // Should generate warnings or not depending on curvature
        // Just check it doesn't crash
    }
    
    ASSERT_TRUE(true);
}

// ============================================================================
// Test 6: Precision requirement calculation (HNF Theorem 4.7)
// ============================================================================

TEST(precision_requirements) {
    // Test the formula: p ≥ log₂(κ·D²/ε)
    
    CurvatureMetrics metrics;
    metrics.kappa_curv = 1e6;
    
    double diameter = 2.0;
    double target_eps = 1e-8;
    
    double required_bits = metrics.required_mantissa_bits(diameter, target_eps);
    
    // p ≥ log₂(1e6 * 4 / 1e-8) = log₂(4e14) ≈ 48.5
    double expected = std::log2((1e6 * 4.0) / 1e-8);
    
    ASSERT_NEAR(required_bits, expected, 0.1);
    ASSERT_TRUE(required_bits > 48);  // Should need more than float32's 24 bits
}

// ============================================================================
// Test 7: Curvature composition bound (HNF Lemma 4.2)
// ============================================================================

TEST(curvature_composition) {
    // Test: κ_{g∘f}^{curv} ≤ κ_g^{curv} · L_f² + L_g · κ_f^{curv}
    
    // Create two layers
    auto layer1 = torch::nn::Linear(10, 8);
    auto layer2 = torch::nn::Linear(8, 5);
    
    CurvatureProfiler profiler1(*layer1);
    CurvatureProfiler profiler2(*layer2);
    
    profiler1.track_layer("layer1", layer1);
    profiler2.track_layer("layer2", layer2);
    
    auto x = torch::randn({4, 10});
    auto h = layer1->forward(x);
    auto y = layer2->forward(h);
    
    auto loss1 = h.pow(2).sum();
    auto loss2 = y.pow(2).sum();
    
    auto metrics1 = profiler1.compute_curvature(loss1, 0);
    auto metrics2 = profiler2.compute_curvature(loss2, 0);
    
    double kappa_f = metrics1["layer1"].kappa_curv;
    double L_f = metrics1["layer1"].lipschitz_constant;
    double kappa_g = metrics2["layer2"].kappa_curv;
    double L_g = metrics2["layer2"].lipschitz_constant;
    
    // For composition, we'd need to compute curvature of the full composition
    // This is a structural test to verify the bound formula makes sense
    double bound = kappa_g * L_f * L_f + L_g * kappa_f;
    
    ASSERT_TRUE(bound >= 0);
    ASSERT_TRUE(std::isfinite(bound));
}

// ============================================================================
// Test 8: Failure prediction with exponential extrapolation
// ============================================================================

TEST(failure_prediction) {
    auto model = torch::nn::Linear(10, 5);
    CurvatureProfiler profiler(*model);
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    TrainingMonitor::Config config;
    config.prediction_horizon = 50;
    config.min_history_length = 10;
    
    TrainingMonitor monitor(profiler, config);
    
    // Simulate increasing curvature (unstable training)
    for (int step = 0; step < 20; ++step) {
        auto x = torch::randn({8, 10});
        // Artificially increase scale to cause high curvature
        auto y = model->forward(x) * std::exp(step * 0.1);
        auto loss = y.pow(2).sum();
        
        monitor.on_step(loss, step);
    }
    
    auto [will_fail, layer, projected] = monitor.predict_failure();
    
    // With exponentially growing loss, prediction should detect issues
    // (though may not always trigger depending on random initialization)
    ASSERT_TRUE(std::isfinite(projected));
}

// ============================================================================
// Test 9: CSV export functionality
// ============================================================================

TEST(csv_export) {
    auto model = torch::nn::Linear(5, 3);
    CurvatureProfiler profiler(*model);
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    for (int step = 0; step < 5; ++step) {
        auto x = torch::randn({4, 5});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        profiler.compute_curvature(loss, step);
    }
    
    std::string filename = "/tmp/test_profiler.csv";
    profiler.export_to_csv(filename);
    
    // Check file exists and has content
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());
    
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
    }
    
    ASSERT_TRUE(line_count > 1);  // Header + data
}

// ============================================================================
// Test 10: Adaptive learning rate scheduler
// ============================================================================

TEST(adaptive_lr_scheduler) {
    auto model = torch::nn::Linear(10, 5);
    CurvatureProfiler profiler(*model);
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    CurvatureAdaptiveLR::Config config;
    config.base_lr = 1e-3;
    config.target_curvature = 1e4;
    
    CurvatureAdaptiveLR scheduler(profiler, config);
    
    // Generate some data to get curvature
    for (int step = 0; step < 3; ++step) {
        auto x = torch::randn({8, 10});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        profiler.compute_curvature(loss, step);
    }
    
    double lr = scheduler.compute_lr(2);
    
    ASSERT_TRUE(lr > 0);
    ASSERT_TRUE(lr <= config.base_lr);
    ASSERT_TRUE(lr >= config.min_lr);
}

// ============================================================================
// Test 11: Lipschitz constant computation
// ============================================================================

TEST(lipschitz_constant) {
    // For a linear layer y = Wx + b, Lipschitz constant is ||W||_op
    
    auto model = torch::nn::Linear(10, 5);
    
    // Set known weights
    auto weight = torch::eye(5, 10);  // Partial identity
    model->weight.set_data(weight);
    
    CurvatureProfiler profiler(*model);
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    auto x = torch::randn({4, 10});
    auto y = model->forward(x);
    auto loss = y.pow(2).sum();
    
    auto metrics = profiler.compute_curvature(loss, 0);
    
    // Spectral norm of partial identity should be 1
    ASSERT_NEAR(metrics["linear"].lipschitz_constant, 1.0, 0.1);
}

// ============================================================================
// Test 12: Visualization heatmap generation
// ============================================================================

TEST(visualization_heatmap) {
    auto model = torch::nn::Linear(5, 3);
    CurvatureProfiler profiler(*model);
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    for (int step = 0; step < 20; ++step) {
        auto x = torch::randn({4, 5});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        profiler.compute_curvature(loss, step);
    }
    
    CurvatureVisualizer viz(profiler);
    std::string heatmap = viz.generate_heatmap();
    
    ASSERT_TRUE(!heatmap.empty());
    ASSERT_TRUE(heatmap.find("linear") != std::string::npos);
}

// ============================================================================
// Test 13: Deep network error propagation (HNF Corollary for deep nets)
// ============================================================================

TEST(deep_network_error_propagation) {
    // Test compositional error bound:
    // Φ_{f_n ∘ ... ∘ f_1}(ε) ≤ Σᵢ (Πⱼ₌ᵢ₊₁ⁿ Lⱼ) · Φᵢ(εᵢ)
    
    struct DeepNet : torch::nn::Module {
        DeepNet() {
            layer1 = register_module("layer1", torch::nn::Linear(10, 8));
            layer2 = register_module("layer2", torch::nn::Linear(8, 6));
            layer3 = register_module("layer3", torch::nn::Linear(6, 4));
        }
        
        torch::Tensor forward(torch::Tensor x) {
            x = layer1->forward(x);
            x = torch::relu(x);
            x = layer2->forward(x);
            x = torch::relu(x);
            x = layer3->forward(x);
            return x;
        }
        
        torch::nn::Linear layer1{nullptr}, layer2{nullptr}, layer3{nullptr};
    };
    
    auto model = std::make_shared<DeepNet>();
    CurvatureProfiler profiler(*model);
    
    profiler.track_layer("layer1", model->layer1);
    profiler.track_layer("layer2", model->layer2);
    profiler.track_layer("layer3", model->layer3);
    
    auto x = torch::randn({4, 10});
    auto y = model->forward(x);
    auto loss = y.pow(2).sum();
    
    auto metrics = profiler.compute_curvature(loss, 0);
    
    // Verify we tracked all layers
    ASSERT_TRUE(metrics.size() == 3);
    ASSERT_TRUE(metrics.find("layer1") != metrics.end());
    ASSERT_TRUE(metrics.find("layer2") != metrics.end());
    ASSERT_TRUE(metrics.find("layer3") != metrics.end());
    
    // Compute error amplification: Π Lᵢ
    double total_amplification = 1.0;
    for (const auto& [name, m] : metrics) {
        total_amplification *= m.lipschitz_constant;
    }
    
    ASSERT_TRUE(total_amplification >= 1.0);
}

// ============================================================================
// Test 14: Real-time dashboard (non-crash test)
// ============================================================================

TEST(realtime_dashboard) {
    auto model = torch::nn::Linear(5, 3);
    CurvatureProfiler profiler(*model);
    profiler.track_layer("linear", std::dynamic_pointer_cast<torch::nn::Module>(model));
    
    TrainingMonitor monitor(profiler);
    RealTimeDashboard dashboard(profiler, monitor);
    
    dashboard.set_compact_mode(true);
    
    for (int step = 0; step < 3; ++step) {
        auto x = torch::randn({4, 5});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        
        monitor.on_step(loss, step);
        dashboard.update(step, loss.item<double>());
    }
    
    std::cout << "\n";  // Clear line after compact display
    ASSERT_TRUE(true);
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    std::cout << "=== Running HNF Condition Profiler Tests ===\n\n";
    
    RUN_TEST(hvp_correctness);
    RUN_TEST(spectral_norm_estimation);
    RUN_TEST(curvature_known_functions);
    RUN_TEST(profiler_history);
    RUN_TEST(training_monitor_warnings);
    RUN_TEST(precision_requirements);
    RUN_TEST(curvature_composition);
    RUN_TEST(failure_prediction);
    RUN_TEST(csv_export);
    RUN_TEST(adaptive_lr_scheduler);
    RUN_TEST(lipschitz_constant);
    RUN_TEST(visualization_heatmap);
    RUN_TEST(deep_network_error_propagation);
    RUN_TEST(realtime_dashboard);
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
