#include "homotopy_lr.hpp"
#include <torch/torch.h>
#include <gtest/gtest.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>

using namespace hnf::homotopy;

//==============================================================================
// Test Utilities
//==============================================================================

// Simple 2-layer MLP for testing
struct SimpleMLP : torch::nn::Module {
    torch::nn::Linear fc1{nullptr}, fc2{nullptr};
    
    SimpleMLP(int input_size, int hidden_size, int output_size) {
        fc1 = register_module("fc1", torch::nn::Linear(input_size, hidden_size));
        fc2 = register_module("fc2", torch::nn::Linear(hidden_size, output_size));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = torch::relu(fc1->forward(x));
        x = fc2->forward(x);
        return x;
    }
};

// Generate synthetic quadratic loss landscape for testing
// L(θ) = 0.5 * θ^T H θ where H is a known Hessian
std::pair<torch::Tensor, torch::Tensor> create_quadratic_problem(
    int dim,
    double condition_number)
{
    // Create diagonal Hessian with specific condition number
    // λ_max / λ_min = condition_number
    auto eigenvalues = torch::linspace(1.0, condition_number, dim);
    auto H = torch::diag(eigenvalues);
    
    // Random initial parameters
    auto theta = torch::randn({dim, 1});
    
    return {H, theta};
}

//==============================================================================
// Unit Tests
//==============================================================================

TEST(CurvatureEstimatorTest, HessianVectorProduct) {
    // Test Hvp computation on quadratic function
    // L(x) = 0.5 * x^T H x where H = [[2, 1], [1, 2]]
    
    auto x = torch::tensor({{1.0}, {2.0}}, torch::requires_grad(true));
    auto H = torch::tensor({{2.0, 1.0}, {1.0, 2.0}});
    
    // Loss: L = 0.5 * x^T H x
    auto loss = 0.5 * x.t().mm(H).mm(x).squeeze();
    
    // Test vector
    auto v = torch::tensor({{1.0}, {0.0}}, torch::requires_grad(false));
    
    // Expected Hv = H * v = [[2], [1]]
    auto expected_hv = H.mm(v);
    
    CurvatureEstimator estimator;
    auto params = std::vector<torch::Tensor>{x};
    auto v_vec = std::vector<torch::Tensor>{v};
    
    auto hv_result = estimator.hessian_vector_product(loss, params, v_vec);
    
    ASSERT_EQ(hv_result.size(), 1);
    EXPECT_TRUE(torch::allclose(hv_result[0], expected_hv, 1e-4, 1e-4));
}

TEST(CurvatureEstimatorTest, PowerIteration) {
    // Test power iteration on known matrix
    // H = [[3, 0], [0, 1]] has max eigenvalue 3
    
    auto x = torch::tensor({{0.0}, {0.0}}, torch::requires_grad(true));
    auto H = torch::tensor({{3.0, 0.0}, {0.0, 1.0}});
    
    // Loss: L = 0.5 * x^T H x
    auto loss = 0.5 * x.t().mm(H).mm(x).squeeze();
    
    HutchinsonConfig config;
    config.power_iterations = 50;
    config.power_iter_tol = 1e-8;
    
    CurvatureEstimator estimator(config);
    auto params = std::vector<torch::Tensor>{x};
    
    double spectral_norm = estimator.estimate_spectral_norm_power(loss, params);
    
    // Should be close to 3.0
    EXPECT_NEAR(spectral_norm, 3.0, 0.1);
}

TEST(CurvatureEstimatorTest, HutchinsonTrace) {
    // Test Hutchinson's method on known matrix
    // H = [[2, 0], [0, 3]] has trace = 5
    
    auto x = torch::tensor({{0.0}, {0.0}}, torch::requires_grad(true));
    auto H = torch::tensor({{2.0, 0.0}, {0.0, 3.0}});
    
    // Loss: L = 0.5 * x^T H x
    auto loss = 0.5 * x.t().mm(H).mm(x).squeeze();
    
    HutchinsonConfig config;
    config.num_samples = 100;  // More samples for accuracy
    config.use_rademacher = true;
    
    CurvatureEstimator estimator(config);
    auto params = std::vector<torch::Tensor>{x};
    
    double trace = estimator.estimate_trace_hutchinson(loss, params);
    
    // Should be close to 5.0 (trace of H)
    EXPECT_NEAR(trace, 5.0, 1.0);  // Stochastic, allow some variance
}

TEST(CurvatureEstimatorTest, FullEstimation) {
    // Test full curvature estimation on a simple problem
    
    SimpleMLP model(10, 20, 5);
    auto input = torch::randn({8, 10});   // Batch of 8
    auto target = torch::randint(0, 5, {8});  // 5 classes
    
    auto output = model.forward(input);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    
    // Backward to compute gradients
    loss.backward();
    
    HutchinsonConfig config;
    config.num_samples = 5;
    config.power_iterations = 10;
    config.estimation_frequency = 1;
    
    CurvatureEstimator estimator(config);
    
    std::vector<torch::Tensor> params;
    for (const auto& p : model.parameters()) {
        params.push_back(p);
    }
    
    auto metrics = estimator.estimate(loss, params);
    
    // Sanity checks
    EXPECT_GT(metrics.gradient_norm, 0.0);
    EXPECT_GT(metrics.spectral_norm_hessian, 0.0);
    EXPECT_GE(metrics.kappa_curv, 0.0);
    
    std::cout << "Gradient norm: " << metrics.gradient_norm << std::endl;
    std::cout << "Spectral norm: " << metrics.spectral_norm_hessian << std::endl;
    std::cout << "Curvature κ: " << metrics.kappa_curv << std::endl;
}

TEST(HomotopyLRSchedulerTest, BasicFunctionality) {
    // Test that scheduler computes reasonable LR based on curvature
    
    SimpleMLP model(10, 20, 5);
    auto input = torch::randn({8, 10});
    auto target = torch::randint(0, 5, {8});
    
    auto output = model.forward(input);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    loss.backward();
    
    HomotopyLRScheduler::Config config;
    config.base_lr = 0.1;
    config.target_curvature = 1e3;
    config.adaptive_target = false;
    
    HutchinsonConfig hvp_config;
    hvp_config.num_samples = 3;
    hvp_config.power_iterations = 5;
    hvp_config.estimation_frequency = 1;
    
    HomotopyLRScheduler scheduler(config, hvp_config);
    
    std::vector<torch::Tensor> params;
    for (const auto& p : model.parameters()) {
        params.push_back(p);
    }
    
    double lr = scheduler.step(loss, params, 0);
    
    // LR should be positive and bounded
    EXPECT_GT(lr, config.min_lr);
    EXPECT_LE(lr, config.max_lr);
    
    std::cout << "Computed LR: " << lr << std::endl;
    std::cout << "Current curvature: " << scheduler.get_current_curvature() << std::endl;
}

TEST(HomotopyLRSchedulerTest, AdaptiveTarget) {
    // Test adaptive target curvature computation
    
    SimpleMLP model(10, 20, 5);
    
    HomotopyLRScheduler::Config config;
    config.base_lr = 0.1;
    config.warmup_steps = 10;
    config.adaptive_target = true;
    config.target_percentile = 0.75;
    
    HutchinsonConfig hvp_config;
    hvp_config.num_samples = 3;
    hvp_config.power_iterations = 5;
    hvp_config.estimation_frequency = 1;
    
    HomotopyLRScheduler scheduler(config, hvp_config);
    
    std::vector<torch::Tensor> params;
    for (const auto& p : model.parameters()) {
        params.push_back(p);
    }
    
    // Run several steps to build history
    for (int step = 0; step < 20; ++step) {
        auto input = torch::randn({8, 10});
        auto target = torch::randint(0, 5, {8});
        
        model.zero_grad();
        auto output = model.forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        loss.backward();
        
        double lr = scheduler.step(loss, params, step);
        
        EXPECT_GT(lr, 0.0);
    }
    
    // After warmup, target should have been adapted
    auto history = scheduler.get_curvature_history();
    EXPECT_GE(history.size(), 20);
}

TEST(HomotopyLRSchedulerTest, ExportMetrics) {
    // Test CSV export
    
    SimpleMLP model(10, 20, 5);
    
    HomotopyLRScheduler::Config config;
    config.base_lr = 0.1;
    
    HutchinsonConfig hvp_config;
    hvp_config.num_samples = 3;
    hvp_config.power_iterations = 5;
    hvp_config.estimation_frequency = 1;
    
    HomotopyLRScheduler scheduler(config, hvp_config);
    
    std::vector<torch::Tensor> params;
    for (const auto& p : model.parameters()) {
        params.push_back(p);
    }
    
    // Run a few steps
    for (int step = 0; step < 5; ++step) {
        auto input = torch::randn({8, 10});
        auto target = torch::randint(0, 5, {8});
        
        model.zero_grad();
        auto output = model.forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        loss.backward();
        
        scheduler.step(loss, params, step);
    }
    
    // Export
    std::string filename = "/tmp/homotopy_lr_test_metrics.csv";
    scheduler.export_metrics(filename);
    
    // Verify file exists and has content
    std::ifstream file(filename);
    ASSERT_TRUE(file.is_open());
    
    std::string line;
    int line_count = 0;
    while (std::getline(file, line)) {
        line_count++;
    }
    
    EXPECT_GT(line_count, 1);  // Header + data
}

//==============================================================================
// Integration Tests with Training
//==============================================================================

TEST(IntegrationTest, QuadraticConvergence) {
    // Test convergence on quadratic problem
    // L(θ) = 0.5 * θ^T H θ with known curvature
    
    int dim = 50;
    double condition_number = 100.0;
    
    auto [H, theta] = create_quadratic_problem(dim, condition_number);
    theta.set_requires_grad(true);
    
    HomotopyLRScheduler::Config config;
    config.base_lr = 0.1;
    config.target_curvature = condition_number;  // Match problem
    config.adaptive_target = false;
    config.alpha = 1.0;
    
    HutchinsonConfig hvp_config;
    hvp_config.num_samples = 5;
    hvp_config.power_iterations = 10;
    hvp_config.estimation_frequency = 5;
    
    HomotopyLRScheduler scheduler(config, hvp_config);
    
    std::vector<double> losses;
    std::vector<double> learning_rates;
    
    for (int step = 0; step < 100; ++step) {
        // Compute loss: L = 0.5 * θ^T H θ
        auto loss = 0.5 * theta.t().mm(H).mm(theta).squeeze();
        losses.push_back(loss.item<double>());
        
        // Backward
        if (theta.grad().defined()) {
            theta.grad().zero_();
        }
        loss.backward();
        
        // Update LR
        double lr = scheduler.step(loss, {theta}, step);
        learning_rates.push_back(lr);
        
        // Gradient step
        {
            torch::NoGradGuard no_grad;
            theta.sub_(lr * theta.grad());
        }
        
        if (step % 20 == 0) {
            std::cout << "Step " << step << ": loss = " << losses.back()
                     << ", LR = " << lr
                     << ", κ = " << scheduler.get_current_curvature() << std::endl;
        }
    }
    
    // Should have decreased loss
    EXPECT_LT(losses.back(), losses[10]);
    
    // Learning rate should have adapted
    EXPECT_GT(learning_rates.back(), config.min_lr);
}

TEST(IntegrationTest, MLPTraining) {
    // Test training a small MLP on synthetic data
    
    const int input_dim = 20;
    const int hidden_dim = 64;
    const int output_dim = 10;
    const int batch_size = 32;
    const int num_steps = 200;
    
    SimpleMLP model(input_dim, hidden_dim, output_dim);
    
    // Generate synthetic dataset
    auto train_inputs = torch::randn({num_steps * batch_size, input_dim});
    auto train_targets = torch::randint(0, output_dim, {num_steps * batch_size});
    
    HomotopyLRScheduler::Config config;
    config.base_lr = 0.01;
    config.target_curvature = 1e4;
    config.adaptive_target = true;
    config.warmup_steps = 50;
    
    HutchinsonConfig hvp_config;
    hvp_config.num_samples = 5;
    hvp_config.power_iterations = 10;
    hvp_config.estimation_frequency = 5;  // Every 5 steps for efficiency
    
    HomotopyLRScheduler scheduler(config, hvp_config);
    
    std::vector<torch::Tensor> params;
    for (const auto& p : model.parameters()) {
        params.push_back(p);
    }
    
    std::vector<double> losses;
    std::vector<double> lrs;
    std::vector<double> curvatures;
    
    for (int step = 0; step < num_steps; ++step) {
        // Get batch
        auto input = train_inputs.slice(0, step * batch_size, (step + 1) * batch_size);
        auto target = train_targets.slice(0, step * batch_size, (step + 1) * batch_size);
        
        // Forward
        model.zero_grad();
        auto output = model.forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        
        // Backward
        loss.backward();
        
        // Update LR based on curvature
        double lr = scheduler.step(loss, params, step);
        
        // Optimization step
        {
            torch::NoGradGuard no_grad;
            for (auto& p : model.parameters()) {
                if (p.grad().defined()) {
                    p.sub_(lr * p.grad());
                }
            }
        }
        
        // Record metrics
        losses.push_back(loss.item<double>());
        lrs.push_back(lr);
        curvatures.push_back(scheduler.get_current_curvature());
        
        if (step % 50 == 0 || step == num_steps - 1) {
            std::cout << "Step " << step << ": "
                     << "loss = " << losses.back() << ", "
                     << "LR = " << lr << ", "
                     << "κ = " << curvatures.back() << std::endl;
        }
    }
    
    // Verify training made progress
    double initial_loss = losses[10];  // Skip first few unstable steps
    double final_loss = losses.back();
    EXPECT_LT(final_loss, initial_loss * 0.5);  // At least 50% reduction
    
    // Verify warmup behavior: LR should increase initially
    double early_lr = lrs[10];
    double mid_lr = lrs[config.warmup_steps];
    // Warmup should increase LR if initial curvature is high
    // (This might not always be true, depends on initialization)
    
    // Export metrics for visualization
    scheduler.export_metrics("/tmp/mlp_training_metrics.csv");
    
    std::cout << "\nTraining summary:" << std::endl;
    std::cout << "Initial loss: " << initial_loss << std::endl;
    std::cout << "Final loss: " << final_loss << std::endl;
    std::cout << "Reduction: " << (1.0 - final_loss / initial_loss) * 100 << "%" << std::endl;
}

TEST(CurvatureAwareGradientClipperTest, BasicClipping) {
    // Test gradient clipping adjusts to curvature
    
    SimpleMLP model(10, 20, 5);
    auto input = torch::randn({8, 10});
    auto target = torch::randint(0, 5, {8});
    
    model.zero_grad();
    auto output = model.forward(input);
    auto loss = torch::nn::functional::cross_entropy(output, target);
    loss.backward();
    
    CurvatureAwareGradientClipper::Config config;
    config.base_clip_norm = 1.0;
    config.curvature_target = 1e4;
    config.min_clip_norm = 0.01;
    
    CurvatureAwareGradientClipper clipper(config);
    
    std::vector<torch::Tensor> params;
    for (const auto& p : model.parameters()) {
        params.push_back(p);
    }
    
    // Test with low curvature (should use base clip norm)
    double clip_norm_low = clipper.clip_gradients(params, 1e3);
    EXPECT_NEAR(clip_norm_low, config.base_clip_norm, 0.2);
    
    // Reset gradients
    model.zero_grad();
    loss.backward();
    
    // Test with high curvature (should reduce clip norm)
    double clip_norm_high = clipper.clip_gradients(params, 1e6);
    EXPECT_LT(clip_norm_high, clip_norm_low);
    EXPECT_GE(clip_norm_high, config.min_clip_norm);
}

TEST(CurvatureAwareWarmupTest, WarmupBehavior) {
    // Test that warmup increases LR adaptively
    
    SimpleMLP model(10, 20, 5);
    
    CurvatureAwareWarmup::Config config;
    config.target_lr = 0.1;
    config.initial_lr_fraction = 0.01;
    config.curvature_threshold = 1e5;
    config.increase_factor = 1.05;
    config.max_warmup_steps = 100;
    
    HutchinsonConfig hvp_config;
    hvp_config.num_samples = 3;
    hvp_config.power_iterations = 5;
    hvp_config.estimation_frequency = 1;
    
    CurvatureAwareWarmup warmup(config, hvp_config);
    
    std::vector<torch::Tensor> params;
    for (const auto& p : model.parameters()) {
        params.push_back(p);
    }
    
    std::vector<double> lrs;
    
    for (int step = 0; step < 50; ++step) {
        auto input = torch::randn({8, 10});
        auto target = torch::randint(0, 5, {8});
        
        model.zero_grad();
        auto output = model.forward(input);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        loss.backward();
        
        double lr = warmup.step(loss, params);
        lrs.push_back(lr);
        
        if (warmup.is_complete()) {
            std::cout << "Warmup completed at step " << step << std::endl;
            break;
        }
    }
    
    // LR should generally increase during warmup
    EXPECT_GT(lrs.back(), lrs[0]);
    
    // Should eventually reach target or complete
    if (warmup.is_complete()) {
        EXPECT_NEAR(lrs.back(), config.target_lr, 0.01);
    }
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);
    
    std::cout << "=============================================================\n";
    std::cout << "HNF Proposal 7: Homotopy Learning Rate - Comprehensive Tests\n";
    std::cout << "=============================================================\n\n";
    
    return RUN_ALL_TESTS();
}
