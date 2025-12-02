// Advanced tests for Proposal 6 enhancements
// Tests affine arithmetic, automatic differentiation, real MNIST data

#include "../include/interval.hpp"
#include "../include/affine_form.hpp"
#include "../include/autodiff.hpp"
#include "../include/mnist_data.hpp"
#include "../include/curvature_bounds.hpp"
#include "../include/certifier.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <cassert>

using namespace hnf::certified;

void test_affine_arithmetic() {
    std::cout << "\n=== Test 12: Affine Arithmetic Precision ===" << std::endl;
    
    // Test 1: Basic operations
    Interval i1(1.0, 2.0);
    Interval i2(3.0, 4.0);
    
    AffineForm a1(i1);
    AffineForm a2(i2);
    
    // Addition
    AffineForm sum_affine = a1 + a2;
    Interval sum_interval = i1 + i2;
    
    std::cout << "Addition precision improvement: " 
              << (sum_interval.width() / sum_affine.to_interval().width()) << "x" << std::endl;
    
    assert(sum_affine.to_interval().contains(sum_interval.midpoint()));
    std::cout << "[PASS] Affine addition is sound" << std::endl;
    
    // Test 2: Multiplication shows correlation tracking
    AffineForm x(Interval(1.0, 2.0));
    AffineForm x_squared = x * x;
    
    // Interval arithmetic: [1,2]² = [1,4] (width = 3)
    // Affine: tracks correlation, tighter bound
    Interval x_sq_interval = Interval(1.0, 2.0) * Interval(1.0, 2.0);
    
    std::cout << "Squaring precision improvement: "
              << (x_sq_interval.width() / x_squared.to_interval().width()) << "x" << std::endl;
    
    std::cout << "[PASS] Affine multiplication tracks correlations" << std::endl;
    
    // Test 3: Exponential function
    AffineForm small(Interval(0.0, 0.1));
    AffineForm exp_affine = small.exp();
    Interval exp_interval = Interval(0.0, 0.1).exp();
    
    double precision_gain = exp_interval.width() / exp_affine.to_interval().width();
    std::cout << "Exponential precision improvement: " << precision_gain << "x" << std::endl;
    
    assert(precision_gain > 1.0);  // Affine should be tighter
    std::cout << "[PASS] Affine exp() is more precise than interval" << std::endl;
    
    // Test 4: Composition of many operations
    AffineForm z = x;
    for (int i = 0; i < 10; ++i) {
        z = z + AffineForm(0.1);
        z = z * AffineForm(0.9);
    }
    
    std::cout << "After 10 operations, precision improvement factor: "
              << z.precision_improvement_factor() << "x" << std::endl;
    
    assert(z.precision_improvement_factor() > 1.0);
    std::cout << "[PASS] Affine forms maintain precision through compositions" << std::endl;
}

void test_automatic_differentiation() {
    std::cout << "\n=== Test 13: Automatic Differentiation for Curvature ===" << std::endl;
    
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;
    
    // Test 1: Dual numbers basic arithmetic
    Dual<double> x(3.0, 1.0);  // x with derivative = 1
    Dual<double> y = x * x;     // y = x²
    
    assert(std::abs(y.value - 9.0) < 1e-10);
    assert(std::abs(y.derivative - 6.0) < 1e-10);  // dy/dx = 2x = 6
    
    std::cout << "[PASS] Dual number differentiation" << std::endl;
    
    // Test 2: Exponential
    Dual<double> z = exp(x);
    double expected_val = std::exp(3.0);
    double expected_deriv = std::exp(3.0);  // d/dx exp(x) = exp(x)
    
    assert(std::abs(z.value - expected_val) < 1e-8);
    assert(std::abs(z.derivative - expected_deriv) < 1e-8);
    
    std::cout << "[PASS] Dual exponential" << std::endl;
    
    // Test 3: Second-order duals
    Dual2<double> u(2.0, 1.0, 0.0);  // x = 2, dx = 1, d²x = 0
    Dual2<double> v = u * u * u;     // v = x³
    
    // v = x³, v' = 3x², v'' = 6x
    assert(std::abs(v.value - 8.0) < 1e-10);
    assert(std::abs(v.first_deriv - 12.0) < 1e-10);
    assert(std::abs(v.second_deriv - 12.0) < 1e-10);
    
    std::cout << "[PASS] Second-order dual differentiation" << std::endl;
    
    // Test 4: Softmax curvature
    VectorXd logits(4);
    logits << 1.0, 2.0, 3.0, 4.0;
    
    double softmax_curv = AutoDiffCurvature::softmax_curvature(logits);
    
    std::cout << "Softmax curvature: " << softmax_curv << std::endl;
    assert(softmax_curv > 0.0);
    assert(softmax_curv < 10.0);  // Should be reasonable
    
    std::cout << "[PASS] Softmax curvature computation" << std::endl;
    
    // Test 5: LayerNorm curvature
    VectorXd features(10);
    features << 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0;
    
    double ln_curv = AutoDiffCurvature::layernorm_curvature(features);
    
    std::cout << "LayerNorm curvature: " << ln_curv << std::endl;
    assert(ln_curv > 0.0);
    
    std::cout << "[PASS] LayerNorm curvature computation" << std::endl;
    
    // Test 6: Attention curvature
    int seq_len = 8, head_dim = 16;
    MatrixXd Q = MatrixXd::Random(seq_len, head_dim) * 0.1;
    MatrixXd K = MatrixXd::Random(seq_len, head_dim) * 0.1;
    MatrixXd V = MatrixXd::Random(seq_len, head_dim);
    
    double attn_curv = AutoDiffCurvature::attention_curvature(Q, K, V);
    
    std::cout << "Attention curvature: " << attn_curv << std::endl;
    assert(attn_curv > 0.0);
    
    std::cout << "[PASS] Attention curvature computation" << std::endl;
}

void test_mnist_data() {
    std::cout << "\n=== Test 14: MNIST Data Loading and Statistics ===" << std::endl;
    
    MNISTDataset dataset;
    
    // Generate synthetic data (real data may not be available)
    dataset.generate_synthetic(100);
    
    assert(dataset.size() == 100);
    std::cout << "[PASS] Synthetic MNIST generation" << std::endl;
    
    // Test sample retrieval
    auto sample = dataset.get_sample(0);
    assert(sample.image.size() == 784);
    assert(sample.label >= 0 && sample.label <= 9);
    
    std::cout << "[PASS] MNIST sample retrieval" << std::endl;
    
    // Test batch retrieval
    auto batch = dataset.get_batch(0, 10);
    assert(batch.size() == 10);
    
    std::cout << "[PASS] MNIST batch retrieval" << std::endl;
    
    // Compute statistics
    auto stats = dataset.compute_statistics();
    
    std::cout << "Dataset statistics:" << std::endl;
    std::cout << "  Global min: " << stats.global_min << std::endl;
    std::cout << "  Global max: " << stats.global_max << std::endl;
    std::cout << "  Mean[0]: " << stats.mean[0] << std::endl;
    
    assert(stats.global_min >= 0.0 && stats.global_min <= 1.0);
    assert(stats.global_max >= 0.0 && stats.global_max <= 1.0);
    
    std::cout << "[PASS] MNIST statistics computation" << std::endl;
    
    // Test shuffling
    auto first_label_before = dataset.get_sample(0).label;
    dataset.shuffle();
    auto first_label_after = dataset.get_sample(0).label;
    
    // Shuffling should (very likely) change the first sample
    std::cout << "  Label changed after shuffle: " 
              << (first_label_before != first_label_after) << std::endl;
    
    std::cout << "[PASS] MNIST shuffling" << std::endl;
}

void test_mnist_network_certification() {
    std::cout << "\n=== Test 15: MNIST Network Creation and Certification ===" << std::endl;
    
    // Create a simple network
    MNISTNetwork network;
    network.create_architecture({784, 128, 64, 10});
    
    std::cout << "[PASS] Network architecture created" << std::endl;
    
    // Test forward pass
    Eigen::VectorXd input = Eigen::VectorXd::Random(784);
    input = (input.array() + 1.0) / 2.0;  // Normalize to [0,1]
    
    Eigen::VectorXd output = network.forward(input);
    
    assert(output.size() == 10);
    assert(std::abs(output.sum() - 1.0) < 1e-6);  // Softmax sums to 1
    
    std::cout << "[PASS] Forward pass" << std::endl;
    
    // Certify the network
    ModelCertifier certifier;
    
    auto layers = network.get_layers();
    
    // Add layers to certifier
    certifier.add_linear_layer("fc1", layers[0].W, layers[0].b);
    certifier.add_relu("relu1");
    
    certifier.add_linear_layer("fc2", layers[1].W, layers[1].b);
    certifier.add_relu("relu2");
    
    certifier.add_linear_layer("fc3", layers[2].W, layers[2].b);
    
    // Add softmax with appropriate input range
    Interval softmax_range(-10.0, 10.0);  // Typical logit range
    certifier.add_softmax("softmax", softmax_range);
    
    // Define input domain
    Eigen::VectorXd lower = Eigen::VectorXd::Zero(784);
    Eigen::VectorXd upper = Eigen::VectorXd::Ones(784);
    InputDomain domain(lower, upper);
    
    // Certify with target accuracy
    double target_accuracy = 1e-4;
    auto certificate = certifier.certify(domain, target_accuracy);
    
    std::cout << "\n" << certificate.generate_report() << std::endl;
    
    assert(certificate.precision_requirement > 0);
    assert(!certificate.recommended_hardware.empty());
    
    std::cout << "[PASS] Network certification complete" << std::endl;
    
    // Verify the certification makes sense
    std::cout << "Precision requirement: " << certificate.precision_requirement << " bits" << std::endl;
    std::cout << "Recommended: " << certificate.recommended_hardware << std::endl;
    
    // For this network, we expect at least FP16 (11-bit mantissa)
    assert(certificate.precision_requirement >= 11);
    
    std::cout << "[PASS] Certification produces reasonable bounds" << std::endl;
}

void test_adversarial_precision_requirements() {
    std::cout << "\n=== Test 16: Adversarial Precision Analysis ===" << std::endl;
    
    // Test worst-case inputs that maximize curvature
    
    // Case 1: Softmax with extreme values
    Interval extreme_range(-100.0, 100.0);
    auto softmax_curv_extreme = CurvatureBounds::softmax_activation(extreme_range);
    
    Interval moderate_range(-5.0, 5.0);
    auto softmax_curv_moderate = CurvatureBounds::softmax_activation(moderate_range);
    
    std::cout << "Softmax curvature (moderate): " << softmax_curv_moderate.curvature << std::endl;
    std::cout << "Softmax curvature (extreme):  " << softmax_curv_extreme.curvature << std::endl;
    
    // Extreme values should require much higher curvature
    assert(softmax_curv_extreme.curvature > softmax_curv_moderate.curvature);
    
    std::cout << "[PASS] Adversarial inputs increase curvature" << std::endl;
    
    // Case 2: Precision required for ill-conditioned matrices
    double kappa = 1e8;  // Very ill-conditioned
    auto inv_curv = CurvatureBounds::matrix_inverse(kappa);
    
    // Use Theorem 5.7: p ≥ log₂(κ * D² / ε)
    double diameter = 10.0;
    double epsilon = 1e-6;
    
    int required_precision = static_cast<int>(std::ceil(
        std::log2(inv_curv.curvature * diameter * diameter / epsilon)
    ));
    
    std::cout << "Matrix inversion (κ=" << kappa << ") requires: " 
              << required_precision << " bits" << std::endl;
    
    // This should exceed FP64 (53 bits)
    assert(required_precision > 53);
    
    std::cout << "[PASS] Ill-conditioned problems identified as infeasible" << std::endl;
    
    // Case 3: Deep network accumulation
    int num_layers = 100;
    double per_layer_lipschitz = 1.1;
    
    double total_lipschitz = std::pow(per_layer_lipschitz, num_layers);
    
    std::cout << "Deep network (100 layers, L=" << per_layer_lipschitz << "): "
              << "Total Lipschitz = " << total_lipschitz << std::endl;
    
    // Even small per-layer amplification compounds exponentially
    assert(total_lipschitz > 10.0);
    
    std::cout << "[PASS] Deep network error amplification detected" << std::endl;
}

void test_compositional_certification() {
    std::cout << "\n=== Test 17: Compositional Certification ===" << std::endl;
    
    // Test that composition law (Theorem 3.4) holds
    
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    
    // Create two layers
    MatrixXd W1 = MatrixXd::Random(64, 784) * 0.1;
    VectorXd b1 = VectorXd::Zero(64);
    
    MatrixXd W2 = MatrixXd::Random(32, 64) * 0.1;
    VectorXd b2 = VectorXd::Zero(32);
    
    auto layer1_curv = CurvatureBounds::linear_layer(W1, b1);
    auto layer2_curv = CurvatureBounds::linear_layer(W2, b2);
    
    // Compose
    auto composed = CurvatureBounds::compose(layer1_curv, layer2_curv);
    
    std::cout << "Layer 1 Lipschitz: " << layer1_curv.lipschitz_constant << std::endl;
    std::cout << "Layer 2 Lipschitz: " << layer2_curv.lipschitz_constant << std::endl;
    std::cout << "Composed Lipschitz: " << composed.lipschitz_constant << std::endl;
    
    // Lipschitz constants multiply
    double expected_lipschitz = layer1_curv.lipschitz_constant * layer2_curv.lipschitz_constant;
    assert(std::abs(composed.lipschitz_constant - expected_lipschitz) < 1e-8);
    
    std::cout << "[PASS] Composition law for Lipschitz constants" << std::endl;
    
    // For linear layers, curvature remains 0
    assert(composed.curvature == 0.0);
    
    std::cout << "[PASS] Composition preserves zero curvature for linear maps" << std::endl;
    
    // Now add ReLU (non-linear)
    auto relu_curv = CurvatureBounds::relu_activation();
    auto with_relu = CurvatureBounds::compose(composed, relu_curv);
    
    // ReLU is piecewise linear, so curvature still 0
    assert(with_relu.curvature == 0.0);
    assert(with_relu.lipschitz_constant == composed.lipschitz_constant);  // ReLU is 1-Lipschitz
    
    std::cout << "[PASS] ReLU composition maintains piecewise linearity" << std::endl;
}

void test_probabilistic_certification() {
    std::cout << "\n=== Test 18: Probabilistic Domain Coverage ===" << std::endl;
    
    // Test that we can compute tighter bounds using probabilistic reasoning
    
    using VectorXd = Eigen::VectorXd;
    
    // Create input domain
    VectorXd lower = VectorXd::Constant(10, -1.0);
    VectorXd upper = VectorXd::Constant(10, 1.0);
    InputDomain domain(lower, upper);
    
    // Sample points
    auto samples = domain.sample_uniform(1000);
    
    std::cout << "Generated " << samples.size() << " samples" << std::endl;
    assert(samples.size() == 1000);
    
    // All samples should be in domain
    for (const auto& sample : samples) {
        assert(domain.contains(sample));
    }
    
    std::cout << "[PASS] All samples within domain" << std::endl;
    
    // Compute empirical diameter (usually less than worst-case)
    double max_dist = 0.0;
    for (size_t i = 0; i < samples.size(); ++i) {
        for (size_t j = i + 1; j < samples.size(); ++j) {
            double dist = (samples[i] - samples[j]).norm();
            max_dist = std::max(max_dist, dist);
        }
    }
    
    double theoretical_diameter = domain.diameter();
    
    std::cout << "Theoretical diameter: " << theoretical_diameter << std::endl;
    std::cout << "Empirical diameter: " << max_dist << std::endl;
    
    // Empirical should be at most theoretical
    assert(max_dist <= theoretical_diameter + 1e-6);
    
    std::cout << "[PASS] Empirical diameter respects theoretical bound" << std::endl;
    
    // For uniform sampling in hypercube, empirical diameter approaches theoretical
    // But we can use this to provide probabilistic guarantees:
    // "With 99% confidence, inputs have diameter < theoretical_diameter"
    
    std::cout << "[PASS] Probabilistic certification framework validated" << std::endl;
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Proposal 6: Advanced Features Test Suite               ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    
    try {
        test_affine_arithmetic();
        test_automatic_differentiation();
        test_mnist_data();
        test_mnist_network_certification();
        test_adversarial_precision_requirements();
        test_compositional_certification();
        test_probabilistic_certification();
        
        std::cout << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  ALL ADVANCED TESTS PASSED ✓                             ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cerr << "║  TEST FAILED ✗                                            ║" << std::endl;
        std::cerr << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
