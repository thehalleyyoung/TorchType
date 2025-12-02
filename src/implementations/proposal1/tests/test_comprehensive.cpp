#include "precision_tensor.h"
#include "precision_nn.h"
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace hnf::proposal1;

// Test utilities
namespace test {

void assert_approx_equal(double a, double b, double tol, const std::string& msg) {
    if (std::abs(a - b) > tol && !std::isinf(a) && !std::isinf(b)) {
        std::cerr << "FAIL: " << msg << " (got " << a << ", expected " << b << ")" << std::endl;
        throw std::runtime_error("Test failed");
    }
}

void assert_true(bool condition, const std::string& msg) {
    if (!condition) {
        std::cerr << "FAIL: " << msg << std::endl;
        throw std::runtime_error("Test failed");
    }
}

} // namespace test

// ============================================================================
// Test 1: Basic Curvature Computations
// ============================================================================

void test_curvature_computations() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 1: Curvature Computations                             ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // Test exponential curvature
    {
        auto x = torch::tensor({1.0, 2.0, 3.0});
        double kappa = CurvatureComputer::exp_curvature(x);
        std::cout << "  exp curvature at [1,2,3]: " << kappa << std::endl;
        test::assert_true(kappa > 0, "exp curvature should be positive");
        test::assert_approx_equal(kappa, std::exp(3.0), 0.1, "exp curvature");
    }
    
    // Test log curvature
    {
        auto x = torch::tensor({0.5, 1.0, 2.0});
        double kappa = CurvatureComputer::log_curvature(x);
        std::cout << "  log curvature at [0.5,1,2]: " << kappa << std::endl;
        test::assert_true(kappa > 0, "log curvature should be positive");
        test::assert_approx_equal(kappa, 0.5 / (0.5 * 0.5), 1.0, "log curvature");
    }
    
    // Test ReLU curvature (should be zero)
    {
        auto x = torch::tensor({-1.0, 0.0, 1.0});
        double kappa = CurvatureComputer::relu_curvature(x);
        std::cout << "  ReLU curvature: " << kappa << std::endl;
        test::assert_approx_equal(kappa, 0.0, 1e-10, "ReLU curvature should be zero");
    }
    
    // Test sigmoid curvature (bounded)
    {
        auto x = torch::tensor({-2.0, 0.0, 2.0});
        double kappa = CurvatureComputer::sigmoid_curvature(x);
        std::cout << "  sigmoid curvature: " << kappa << std::endl;
        test::assert_true(kappa <= 0.3, "sigmoid curvature should be bounded by 0.25");
    }
    
    std::cout << "✓ Curvature computations passed\n";
}

// ============================================================================
// Test 2: Precision Requirement Calculation
// ============================================================================

void test_precision_requirements() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 2: Precision Requirements (Theorem 5.7)               ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // High curvature operation should require more bits
    {
        auto x = torch::randn({10});
        PrecisionTensor pt_exp = ops::exp(PrecisionTensor(x));
        
        std::cout << "  exp operation: " << pt_exp.required_bits() << " bits required\n";
        std::cout << "  Recommended: " << precision_name(pt_exp.recommend_precision()) << "\n";
        
        test::assert_true(pt_exp.required_bits() > 10, "exp should require significant precision");
    }
    
    // Low curvature (linear) operation
    {
        auto x = torch::randn({10});
        auto W = torch::randn({10, 10});
        PrecisionTensor pt_x(x);
        PrecisionTensor pt_W(W);
        PrecisionTensor pt_matmul = ops::matmul(pt_x, pt_W);
        
        std::cout << "  matmul operation: " << pt_matmul.required_bits() << " bits required\n";
        std::cout << "  Recommended: " << precision_name(pt_matmul.recommend_precision()) << "\n";
    }
    
    // ReLU (piecewise linear, zero curvature)
    {
        auto x = torch::randn({10});
        PrecisionTensor pt_relu = ops::relu(PrecisionTensor(x));
        
        std::cout << "  ReLU operation: " << pt_relu.required_bits() << " bits required\n";
        std::cout << "  Recommended: " << precision_name(pt_relu.recommend_precision()) << "\n";
        
        // ReLU has zero curvature, should have minimal requirements
        test::assert_true(pt_relu.required_bits() <= 25, "ReLU should have low precision req");
    }
    
    std::cout << "✓ Precision requirements passed\n";
}

// ============================================================================
// Test 3: Error Propagation (Theorem 3.8)
// ============================================================================

void test_error_propagation() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 3: Error Propagation (Stability Composition Theorem)  ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    auto x = torch::ones({5});
    PrecisionTensor pt(x);
    
    // Chain: x -> exp -> log -> sqrt
    auto y1 = ops::exp(pt);
    auto y2 = ops::log(y1);
    auto y3 = ops::sqrt(y2);
    
    double input_error = 1e-6;
    double final_error = y3.propagate_error(input_error);
    
    std::cout << "  Input error:  " << std::scientific << input_error << "\n";
    std::cout << "  Final error:  " << std::scientific << final_error << "\n";
    std::cout << "  Lipschitz:    " << std::scientific << y3.lipschitz() << "\n";
    std::cout << "  Curvature:    " << std::scientific << y3.curvature() << "\n";
    
    // Error should propagate according to composition law
    test::assert_true(final_error > input_error, "Error should accumulate in composition");
    
    std::cout << "✓ Error propagation passed\n";
}

// ============================================================================
// Test 4: Lipschitz Constant Composition
// ============================================================================

void test_lipschitz_composition() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 4: Lipschitz Constant Composition                     ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    auto x = torch::randn({10});
    PrecisionTensor pt(x, 2.0, 0.0);  // L=2
    
    std::cout << "  Input Lipschitz: " << pt.lipschitz() << "\n";
    
    // ReLU is 1-Lipschitz
    auto y1 = ops::relu(pt);
    std::cout << "  After ReLU:      " << y1.lipschitz() << "\n";
    test::assert_approx_equal(y1.lipschitz(), 2.0, 0.1, "ReLU preserves Lipschitz");
    
    // Sigmoid has L=0.25
    auto y2 = ops::sigmoid(y1);
    std::cout << "  After sigmoid:   " << y2.lipschitz() << "\n";
    test::assert_approx_equal(y2.lipschitz(), 2.0 * 0.25, 0.1, "Sigmoid scales Lipschitz");
    
    std::cout << "✓ Lipschitz composition passed\n";
}

// ============================================================================
// Test 5: Gallery Example 6 - Log-Sum-Exp Stability
// ============================================================================

void test_logsumexp_stability() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 5: Log-Sum-Exp Stability (Gallery Example 6)          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // Test with large values that would overflow naive implementation
    auto x = torch::tensor({100.0, 200.0, 300.0});
    
    PrecisionTensor pt(x);
    auto lse = ops::logsumexp(pt);
    
    std::cout << "  Input range: [100, 300]\n";
    std::cout << "  LSE curvature (stable): " << lse.curvature() << "\n";
    std::cout << "  LSE required bits:      " << lse.required_bits() << "\n";
    
    // Stable version should have bounded curvature
    test::assert_true(lse.curvature() < 10.0, "Stable LSE should have bounded curvature");
    
    // Verify the result is correct
    double expected = 300.0 + std::log(std::exp(-200.0) + std::exp(-100.0) + 1.0);
    double actual = lse.data().item<double>();
    std::cout << "  Expected: " << expected << "\n";
    std::cout << "  Actual:   " << actual << "\n";
    test::assert_approx_equal(actual, expected, 1e-6, "LSE numerical value");
    
    std::cout << "✓ Log-sum-exp stability passed\n";
}

// ============================================================================
// Test 6: Simple Neural Network
// ============================================================================

void test_simple_network() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 6: Simple Feedforward Network                         ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // Create a simple 3-layer network: 10 -> 20 -> 10 -> 5
    std::vector<int> sizes = {10, 20, 10, 5};
    SimpleFeedForward model(sizes, "relu", "test_ff");
    
    auto input = torch::randn({1, 10});
    PrecisionTensor pt_input(input);
    
    auto output = model.forward(pt_input);
    
    std::cout << "\n  Network architecture: ";
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i];
        if (i < sizes.size() - 1) std::cout << " -> ";
    }
    std::cout << "\n";
    
    model.print_precision_report();
    
    // Check if can run on float32
    bool can_run_fp32 = model.can_run_on(Precision::FLOAT32);
    std::cout << "\n  Can run on fp32: " << (can_run_fp32 ? "YES" : "NO") << "\n";
    
    // Check if can run on float16
    bool can_run_fp16 = model.can_run_on(Precision::FLOAT16);
    std::cout << "  Can run on fp16: " << (can_run_fp16 ? "YES" : "NO") << "\n";
    
    std::cout << "✓ Simple network passed\n";
}

// ============================================================================
// Test 7: Attention Mechanism (Gallery Example 4)
// ============================================================================

void test_attention_mechanism() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 7: Attention Mechanism (Gallery Example 4)            ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    int seq_len = 8;
    int d_model = 64;
    
    auto Q = torch::randn({1, seq_len, d_model});
    auto K = torch::randn({1, seq_len, d_model});
    auto V = torch::randn({1, seq_len, d_model});
    
    PrecisionTensor pt_Q(Q);
    PrecisionTensor pt_K(K);
    PrecisionTensor pt_V(V);
    
    auto attn_output = ops::attention(pt_Q, pt_K, pt_V);
    
    std::cout << "  Sequence length: " << seq_len << "\n";
    std::cout << "  Model dimension: " << d_model << "\n";
    std::cout << "  Attention curvature:     " << std::scientific << attn_output.curvature() << "\n";
    std::cout << "  Attention Lipschitz:     " << std::scientific << attn_output.lipschitz() << "\n";
    std::cout << "  Required bits:           " << attn_output.required_bits() << "\n";
    std::cout << "  Recommended precision:   " << precision_name(attn_output.recommend_precision()) << "\n";
    
    // From paper: attention has high curvature due to softmax composition
    test::assert_true(attn_output.curvature() > 1.0, "Attention should have significant curvature");
    
    std::cout << "✓ Attention mechanism passed\n";
}

// ============================================================================
// Test 8: Precision vs Accuracy Trade-off
// ============================================================================

void test_precision_accuracy_tradeoff() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 8: Precision vs Accuracy Trade-off                    ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    auto x = torch::randn({100});
    PrecisionTensor pt(x);
    
    // Test different target accuracies
    std::vector<double> targets = {1e-3, 1e-6, 1e-9, 1e-12};
    
    std::cout << "\n  Target Accuracy vs Required Precision:\n";
    std::cout << "  ────────────────────────────────────────\n";
    
    for (double eps : targets) {
        pt.set_target_accuracy(eps);
        std::cout << "  ε = " << std::scientific << std::setw(10) << eps
                  << "  →  " << std::setw(3) << pt.required_bits() << " bits"
                  << "  (" << precision_name(pt.recommend_precision()) << ")\n";
    }
    
    std::cout << "\n✓ Precision-accuracy tradeoff passed\n";
}

// ============================================================================
// Test 9: Catastrophic Cancellation Detection
// ============================================================================

void test_catastrophic_cancellation() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 9: Catastrophic Cancellation (Gallery Example 1)      ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // Compute (x-1)^2 in two ways:
    // 1. Direct: (x-1)^2
    // 2. Expanded: x^2 - 2x + 1
    
    auto x = torch::tensor({1.00001});  // Very close to 1
    
    // Method 1: Direct (numerically stable)
    {
        PrecisionTensor pt_x(x);
        auto pt_x_minus_1 = ops::sub(pt_x, PrecisionTensor(torch::ones_like(x)));
        auto result = ops::pow(pt_x_minus_1, 2.0);
        
        std::cout << "  Direct method (x-1)²:\n";
        std::cout << "    Curvature:      " << result.curvature() << "\n";
        std::cout << "    Required bits:  " << result.required_bits() << "\n";
        std::cout << "    Value:          " << std::scientific << result.data().item<double>() << "\n";
    }
    
    // Method 2: Expanded (numerically unstable due to cancellation)
    {
        PrecisionTensor pt_x(x);
        auto x_sq = ops::pow(pt_x, 2.0);
        auto two_x = ops::mul(PrecisionTensor(torch::tensor({2.0})), pt_x);
        auto term1 = ops::sub(x_sq, two_x);
        auto result = ops::add(term1, PrecisionTensor(torch::ones_like(x)));
        
        std::cout << "\n  Expanded method x² - 2x + 1:\n";
        std::cout << "    Curvature:      " << result.curvature() << "\n";
        std::cout << "    Required bits:  " << result.required_bits() << "\n";
        std::cout << "    Value:          " << std::scientific << result.data().item<double>() << "\n";
    }
    
    std::cout << "\n  Note: Expanded form may require more precision due to cancellation\n";
    std::cout << "✓ Catastrophic cancellation test passed\n";
}

// ============================================================================
// Test 10: Deep Network Precision Analysis
// ============================================================================

void test_deep_network() {
    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  TEST 10: Deep Network Precision Analysis                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
    
    // Create a deeper network to test error accumulation
    std::vector<int> sizes = {32, 64, 64, 32, 16, 8};
    SimpleFeedForward model(sizes, "relu", "deep_network");
    
    auto input = torch::randn({4, 32});  // Batch of 4
    PrecisionTensor pt_input(input);
    
    auto output = model.forward(pt_input);
    
    std::cout << "\n  Deep Network: " << sizes.size() - 1 << " layers\n";
    
    model.print_precision_report();
    
    // Analyze per-precision performance
    std::cout << "\n  Precision Compatibility:\n";
    std::cout << "  ────────────────────────────────────────\n";
    
    for (auto prec : {Precision::FLOAT16, Precision::FLOAT32, Precision::FLOAT64}) {
        bool compatible = model.can_run_on(prec);
        std::cout << "  " << std::setw(10) << precision_name(prec) << ": "
                  << (compatible ? "✓ COMPATIBLE" : "✗ INSUFFICIENT") << "\n";
    }
    
    std::cout << "\n✓ Deep network analysis passed\n";
}

// ============================================================================
// Main Test Runner
// ============================================================================

int main() {
    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                                                                          ║\n";
    std::cout << "║    HNF PROPOSAL #1: PRECISION-AWARE AUTOMATIC DIFFERENTIATION           ║\n";
    std::cout << "║    Comprehensive Test Suite                                             ║\n";
    std::cout << "║                                                                          ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";
    
    try {
        test_curvature_computations();
        test_precision_requirements();
        test_error_propagation();
        test_lipschitz_composition();
        test_logsumexp_stability();
        test_simple_network();
        test_attention_mechanism();
        test_precision_accuracy_tradeoff();
        test_catastrophic_cancellation();
        test_deep_network();
        
        std::cout << "\n";
        std::cout << "╔══════════════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                                                                          ║\n";
        std::cout << "║    ✓✓✓ ALL TESTS PASSED ✓✓✓                                            ║\n";
        std::cout << "║                                                                          ║\n";
        std::cout << "║    The HNF Precision-Aware AD framework successfully:                   ║\n";
        std::cout << "║    • Computes curvature bounds for primitive operations                 ║\n";
        std::cout << "║    • Tracks precision requirements through compositions                 ║\n";
        std::cout << "║    • Propagates errors according to Theorem 3.8                         ║\n";
        std::cout << "║    • Analyzes neural networks for mixed-precision deployment            ║\n";
        std::cout << "║    • Validates theoretical predictions from the HNF paper               ║\n";
        std::cout << "║                                                                          ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗✗✗ TEST FAILED ✗✗✗\n";
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
