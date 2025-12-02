#include "curvature_profiler.hpp"
#include "visualization.hpp"
#include <torch/torch.h>
#include <iostream>
#include <cassert>
#include <cmath>

using namespace hnf::profiler;

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

// ============================================================================
// Test 1: Basic profiler setup
// ============================================================================

TEST(basic_setup) {
    auto model = torch::nn::Sequential(
        torch::nn::Linear(10, 5)
    );
    
    CurvatureProfiler profiler(*model);
    
    // Track layers
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        profiler.track_layer("layer" + std::to_string(idx++), child.get());
    }
    
    ASSERT_TRUE(profiler.get_tracked_layers().size() == 1);
}

// ============================================================================
// Test 2: Curvature computation
// ============================================================================

TEST(curvature_computation) {
    auto model = torch::nn::Sequential(
        torch::nn::Linear(10, 5)
    );
    
    CurvatureProfiler profiler(*model);
    
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        profiler.track_layer("layer" + std::to_string(idx++), child.get());
    }
    
    // Generate data
    auto x = torch::randn({8, 10});
    auto y = model->forward(x);
    auto loss = y.pow(2).sum();
    
    auto metrics = profiler.compute_curvature(loss, 0);
    
    ASSERT_TRUE(metrics.size() > 0);
    ASSERT_TRUE(metrics["layer0"].kappa_curv >= 0);
    ASSERT_TRUE(std::isfinite(metrics["layer0"].kappa_curv));
}

// ============================================================================
// Test 3: History tracking
// ============================================================================

TEST(history_tracking) {
    auto model = torch::nn::Sequential(
        torch::nn::Linear(10, 5)
    );
    
    CurvatureProfiler profiler(*model);
    
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        profiler.track_layer("layer" + std::to_string(idx++), child.get());
    }
    
    // Run multiple steps
    for (int step = 0; step < 5; ++step) {
        auto x = torch::randn({8, 10});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        profiler.compute_curvature(loss, step);
    }
    
    const auto& history = profiler.get_history("layer0");
    ASSERT_TRUE(history.size() == 5);
}

// ============================================================================
// Test 4: Training monitor
// ============================================================================

TEST(training_monitor) {
    auto model = torch::nn::Sequential(
        torch::nn::Linear(10, 5)
    );
    
    CurvatureProfiler profiler(*model);
    
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        profiler.track_layer("layer" + std::to_string(idx++), child.get());
    }
    
    TrainingMonitor monitor(profiler);
    
    for (int step = 0; step < 3; ++step) {
        auto x = torch::randn({8, 10});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        
        auto warnings = monitor.on_step(loss, step);
        // Just check it doesn't crash
    }
    
    ASSERT_TRUE(true);
}

// ============================================================================
// Test 5: Precision requirements (HNF Theorem 4.7)
// ============================================================================

TEST(precision_requirements) {
    CurvatureMetrics metrics;
    metrics.kappa_curv = 1e6;
    
    double diameter = 2.0;
    double target_eps = 1e-8;
    
    double required_bits = metrics.required_mantissa_bits(diameter, target_eps);
    
    ASSERT_TRUE(required_bits > 0);
    ASSERT_TRUE(std::isfinite(required_bits));
}

// ============================================================================
// Test 6: CSV export
// ============================================================================

TEST(csv_export) {
    auto model = torch::nn::Sequential(
        torch::nn::Linear(5, 3)
    );
    
    CurvatureProfiler profiler(*model);
    
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        profiler.track_layer("layer" + std::to_string(idx++), child.get());
    }
    
    for (int step = 0; step < 3; ++step) {
        auto x = torch::randn({4, 5});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        profiler.compute_curvature(loss, step);
    }
    
    std::string filename = "/tmp/test_profiler.csv";
    profiler.export_to_csv(filename);
    
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());
}

// ============================================================================
// Test 7: Visualization
// ============================================================================

TEST(visualization) {
    auto model = torch::nn::Sequential(
        torch::nn::Linear(5, 3)
    );
    
    CurvatureProfiler profiler(*model);
    
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        profiler.track_layer("layer" + std::to_string(idx++), child.get());
    }
    
    for (int step = 0; step < 10; ++step) {
        auto x = torch::randn({4, 5});
        auto y = model->forward(x);
        auto loss = y.pow(2).sum();
        profiler.compute_curvature(loss, step);
    }
    
    CurvatureVisualizer viz(profiler);
    std::string heatmap = viz.generate_heatmap();
    
    ASSERT_TRUE(!heatmap.empty());
}

// ============================================================================
// Main test runner
// ============================================================================

int main() {
    std::cout << "=== Running HNF Condition Profiler Tests ===\n\n";
    
    RUN_TEST(basic_setup);
    RUN_TEST(curvature_computation);
    RUN_TEST(history_tracking);
    RUN_TEST(training_monitor);
    RUN_TEST(precision_requirements);
    RUN_TEST(csv_export);
    RUN_TEST(visualization);
    
    std::cout << "\n=== All tests passed! ===\n";
    return 0;
}
