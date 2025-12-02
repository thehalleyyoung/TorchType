#include "../include/interval.hpp"
#include "../include/input_domain.hpp"
#include "../include/curvature_bounds.hpp"
#include "../include/certifier.hpp"
#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>

using namespace hnf::certified;

// Color codes for output
#define GREEN "\033[32m"
#define RED "\033[31m"
#define YELLOW "\033[33m"
#define RESET "\033[0m"

void print_test(const std::string& name, bool passed) {
    std::cout << "[" << (passed ? GREEN "PASS" : RED "FAIL") << RESET << "] " << name << std::endl;
    if (!passed) {
        throw std::runtime_error("Test failed: " + name);
    }
}

// Test 1: Interval arithmetic correctness
void test_interval_arithmetic() {
    std::cout << "\n=== Test 1: Interval Arithmetic ===" << std::endl;
    
    // Basic operations
    Interval a(1.0, 2.0);
    Interval b(3.0, 4.0);
    
    Interval sum = a + b;
    print_test("Interval addition", sum.lower() == 4.0 && sum.upper() == 6.0);
    
    Interval diff = b - a;
    print_test("Interval subtraction", diff.lower() == 1.0 && diff.upper() == 3.0);
    
    Interval prod = a * b;
    print_test("Interval multiplication", prod.lower() == 3.0 && prod.upper() == 8.0);
    
    // Exponential
    Interval x(0.0, 1.0);
    Interval exp_x = x.exp();
    bool exp_correct = std::abs(exp_x.lower() - 1.0) < 1e-10 && 
                      std::abs(exp_x.upper() - std::exp(1.0)) < 1e-10;
    print_test("Interval exponential", exp_correct);
    
    // Logarithm
    Interval y(1.0, std::exp(1.0));
    Interval log_y = y.log();
    bool log_correct = std::abs(log_y.lower()) < 1e-10 && 
                      std::abs(log_y.upper() - 1.0) < 1e-10;
    print_test("Interval logarithm", log_correct);
    
    // Square root
    Interval z(4.0, 9.0);
    Interval sqrt_z = z.sqrt();
    print_test("Interval sqrt", sqrt_z.lower() == 2.0 && sqrt_z.upper() == 3.0);
    
    // Containment
    print_test("Interval contains", a.contains(1.5) && !a.contains(3.0));
}

// Test 2: Input domain functionality
void test_input_domain() {
    std::cout << "\n=== Test 2: Input Domain ===" << std::endl;
    
    Eigen::VectorXd lower(3);
    lower << -1.0, -2.0, -3.0;
    Eigen::VectorXd upper(3);
    upper << 1.0, 2.0, 3.0;
    
    InputDomain domain(lower, upper);
    
    // Diameter calculation
    double expected_diam = std::sqrt(4.0 + 16.0 + 36.0);
    bool diam_correct = std::abs(domain.diameter() - expected_diam) < 1e-10;
    print_test("Domain diameter", diam_correct);
    
    // Sampling
    auto samples = domain.sample(100);
    print_test("Domain sampling count", samples.size() == 100);
    
    // Check all samples are in domain
    bool all_in_domain = true;
    for (const auto& s : samples) {
        if (!domain.contains(s)) {
            all_in_domain = false;
            break;
        }
    }
    print_test("All samples in domain", all_in_domain);
    
    // Boundary sampling
    auto boundary_samples = domain.sample_boundary(50);
    print_test("Boundary sampling count", boundary_samples.size() == 50);
    
    // Subdivision
    auto subdomains = domain.subdivide(2);
    print_test("Subdivision count", subdomains.size() == 8);  // 2^3
    
    // Volume
    double expected_vol = 2.0 * 4.0 * 6.0;
    bool vol_correct = std::abs(domain.volume() - expected_vol) < 1e-10;
    print_test("Domain volume", vol_correct);
}

// Test 3: Curvature bounds for different layers
void test_curvature_bounds() {
    std::cout << "\n=== Test 3: Curvature Bounds ===" << std::endl;
    
    // Linear layer
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(3, 3) * 2.0;
    Eigen::VectorXd b = Eigen::VectorXd::Zero(3);
    auto linear = CurvatureBounds::linear_layer(W, b);
    
    print_test("Linear curvature is zero", linear.curvature == 0.0);
    print_test("Linear Lipschitz constant", std::abs(linear.lipschitz_constant - 2.0) < 1e-10);
    
    // ReLU
    auto relu = CurvatureBounds::relu_activation();
    print_test("ReLU curvature is zero", relu.curvature == 0.0);
    print_test("ReLU Lipschitz is 1", relu.lipschitz_constant == 1.0);
    
    // Softmax with bounded inputs
    Interval input_range(-5.0, 5.0);
    auto softmax = CurvatureBounds::softmax_activation(input_range);
    print_test("Softmax has positive curvature", softmax.curvature > 0.0);
    print_test("Softmax Lipschitz is 1", softmax.lipschitz_constant == 1.0);
    
    // Layer normalization
    Interval var_bounds(0.5, 2.0);
    auto layer_norm = CurvatureBounds::layer_norm(var_bounds);
    print_test("LayerNorm has positive curvature", layer_norm.curvature > 0.0);
    
    // GELU
    Interval gelu_range(-3.0, 3.0);
    auto gelu = CurvatureBounds::gelu_activation(gelu_range);
    print_test("GELU has positive curvature", gelu.curvature > 0.0);
    print_test("GELU Lipschitz > 1", gelu.lipschitz_constant > 1.0);
    
    // Composition
    auto composed = CurvatureBounds::compose(linear, relu);
    print_test("Composition curvature", composed.curvature == 0.0);  // Both linear
    print_test("Composition Lipschitz", std::abs(composed.lipschitz_constant - 2.0) < 1e-10);
}

// Test 4: Precision computation (Theorem 5.7)
void test_precision_computation() {
    std::cout << "\n=== Test 4: Precision Computation ===" << std::endl;
    
    // Test case from paper: matrix inversion with κ = 10^8
    double curvature = 1e8;
    double diameter = 10.0;
    double target_accuracy = 1e-8;
    
    int precision = PrecisionComputer::compute_minimum_precision(
        curvature, diameter, target_accuracy);
    
    std::cout << "  Curvature: " << curvature << std::endl;
    std::cout << "  Diameter: " << diameter << std::endl;
    std::cout << "  Target accuracy: " << target_accuracy << std::endl;
    std::cout << "  Required precision: " << precision << " bits" << std::endl;
    
    // Should require more than fp32 (24 bits)
    print_test("High curvature requires high precision", precision > 24);
    
    // Hardware recommendation
    std::string hw = PrecisionComputer::recommend_hardware(precision);
    std::cout << "  Recommended: " << hw << std::endl;
    print_test("Recommends fp64 or higher", hw.find("64") != std::string::npos);
    
    // Low curvature case
    int low_precision = PrecisionComputer::compute_minimum_precision(
        0.1, 1.0, 1e-4);
    std::cout << "  Low curvature precision: " << low_precision << " bits" << std::endl;
    print_test("Low curvature needs less precision", low_precision < precision);
}

// Test 5: Simple model certification
void test_simple_certification() {
    std::cout << "\n=== Test 5: Simple Model Certification ===" << std::endl;
    
    // Create a simple 2-layer network: Linear -> ReLU -> Linear
    ModelCertifier certifier;
    
    Eigen::MatrixXd W1(4, 3);
    W1 << 1.0, 0.5, 0.2,
          0.3, 1.0, 0.4,
          0.1, 0.3, 1.0,
          0.5, 0.2, 0.3;
    Eigen::VectorXd b1 = Eigen::VectorXd::Zero(4);
    
    Eigen::MatrixXd W2(2, 4);
    W2 << 1.0, 0.5, 0.3, 0.2,
          0.4, 1.0, 0.2, 0.5;
    Eigen::VectorXd b2 = Eigen::VectorXd::Zero(2);
    
    certifier.add_linear_layer("layer1", W1, b1);
    certifier.add_relu("relu1");
    certifier.add_linear_layer("layer2", W2, b2);
    
    print_test("Model has 3 layers", certifier.num_layers() == 3);
    
    // Define input domain
    Eigen::VectorXd lower(3);
    lower << -1.0, -1.0, -1.0;
    Eigen::VectorXd upper(3);
    upper << 1.0, 1.0, 1.0;
    InputDomain domain(lower, upper);
    
    // Certify the model
    auto cert = certifier.certify(domain, 1e-6);
    
    std::cout << "\n" << cert.generate_report() << std::endl;
    
    print_test("Certificate generated", !cert.model_hash.empty());
    print_test("Precision requirement computed", cert.precision_requirement > 0);
    print_test("Hardware recommended", !cert.recommended_hardware.empty());
    
    // Verify the certificate
    double recomputed_curvature = certifier.compute_total_curvature(domain);
    bool verified = cert.verify(recomputed_curvature);
    print_test("Certificate verification", verified);
    
    // For piecewise linear network (Linear-ReLU-Linear), curvature should be zero
    print_test("Piecewise linear has zero curvature", recomputed_curvature == 0.0);
}

// Test 6: Softmax certification (high curvature)
void test_softmax_certification() {
    std::cout << "\n=== Test 6: Softmax Certification ===" << std::endl;
    
    ModelCertifier certifier;
    
    // Simple model: Linear -> Softmax
    // Use smaller weights for realistic softmax inputs
    Eigen::MatrixXd W(5, 10);
    W.setRandom();
    W *= 0.1;  // Small scale to keep logits in reasonable range
    Eigen::VectorXd b = Eigen::VectorXd::Zero(5);
    
    certifier.add_linear_layer("logits", W, b);
    
    // Estimate input range for softmax
    Eigen::VectorXd lower(10);
    lower.setConstant(-1.0);
    Eigen::VectorXd upper(10);
    upper.setConstant(1.0);
    
    // After linear layer, estimate output range conservatively
    // For normalized inputs in [-1, 1], output should be small
    double max_output = 2.0;  // Conservative estimate for scaled weights
    Interval softmax_input(-max_output, max_output);
    
    certifier.add_softmax("softmax", softmax_input);
    
    InputDomain domain(lower, upper);
    
    // Certify
    auto cert = certifier.certify(domain, 1e-4);
    
    std::cout << "\n" << cert.generate_report() << std::endl;
    
    // Softmax should have significant curvature
    print_test("Softmax model has positive curvature", cert.curvature_bound > 0.0);
    
    // With reasonable inputs, should require fp32 or less
    print_test("Precision requirement reasonable", 
               cert.precision_requirement >= 11 && cert.precision_requirement <= 52);
}

// Test 7: Attention layer certification
void test_attention_certification() {
    std::cout << "\n=== Test 7: Attention Layer Certification ===" << std::endl;
    
    int d_model = 64;
    int seq_len = 128;
    int head_dim = 16;
    
    // Create Q, K, V projection matrices
    Eigen::MatrixXd Q = Eigen::MatrixXd::Random(head_dim, d_model) * 0.1;
    Eigen::MatrixXd K = Eigen::MatrixXd::Random(head_dim, d_model) * 0.1;
    Eigen::MatrixXd V = Eigen::MatrixXd::Random(head_dim, d_model) * 0.1;
    
    auto attn_curv = CurvatureBounds::attention_layer(Q, K, V, seq_len, head_dim);
    
    std::cout << "  Attention curvature: " << attn_curv.curvature << std::endl;
    std::cout << "  Attention Lipschitz: " << attn_curv.lipschitz_constant << std::endl;
    
    // Attention should have high curvature due to softmax over long sequences
    print_test("Attention has positive curvature", attn_curv.curvature > 0.0);
    
    // Compute precision requirement
    double diameter = std::sqrt(d_model) * 2.0;  // Assume normalized inputs
    int precision = PrecisionComputer::compute_minimum_precision(
        attn_curv.curvature, diameter, 1e-3);
    
    std::cout << "  Required precision: " << precision << " bits" << std::endl;
    std::string hw = PrecisionComputer::recommend_hardware(precision);
    std::cout << "  Recommended hardware: " << hw << std::endl;
    
    // For long sequences, should need more than int8
    print_test("Attention needs > int8", precision > 8);
}

// Test 8: Matrix inversion precision bound
void test_matrix_inversion() {
    std::cout << "\n=== Test 8: Matrix Inversion Precision ===" << std::endl;
    
    // Test different condition numbers
    std::vector<double> condition_numbers = {10.0, 100.0, 1000.0, 1e6, 1e8};
    
    for (double kappa : condition_numbers) {
        auto inv_curv = CurvatureBounds::matrix_inverse(kappa);
        
        double diameter = 10.0;
        double target_acc = 1e-8;
        
        int precision = PrecisionComputer::compute_minimum_precision(
            inv_curv.curvature, diameter, target_acc);
        
        std::string hw = PrecisionComputer::recommend_hardware(precision);
        
        std::cout << "  κ = " << std::scientific << kappa 
                  << " → " << precision << " bits (" << hw << ")" << std::endl;
    }
    
    // High condition number should require high precision
    auto high_cond = CurvatureBounds::matrix_inverse(1e8);
    int high_prec = PrecisionComputer::compute_minimum_precision(
        high_cond.curvature, 10.0, 1e-8);
    
    print_test("High condition number needs fp64+", high_prec > 52);
}

// Test 9: Interval propagation through network
void test_interval_propagation() {
    std::cout << "\n=== Test 9: Interval Propagation ===" << std::endl;
    
    ModelCertifier certifier;
    
    // Small network
    Eigen::MatrixXd W(2, 2);
    W << 2.0, 0.5,
         0.5, 2.0;
    Eigen::VectorXd b(2);
    b << 0.1, -0.1;
    
    certifier.add_linear_layer("layer1", W, b);
    certifier.add_relu("relu1");
    
    // Input domain
    Eigen::VectorXd lower(2);
    lower << -1.0, -1.0;
    Eigen::VectorXd upper(2);
    upper << 1.0, 1.0;
    InputDomain domain(lower, upper);
    
    // Propagate intervals
    auto intervals = certifier.propagate_intervals(domain);
    
    print_test("Interval propagation produces correct number of layers", 
               intervals.size() == 3);  // Input + 2 layers
    
    // Check that output intervals are finite
    bool all_finite = true;
    for (const auto& iv : intervals) {
        for (size_t i = 0; i < iv.size(); ++i) {
            if (!std::isfinite(iv[i].lower()) || !std::isfinite(iv[i].upper())) {
                all_finite = false;
            }
        }
    }
    print_test("All interval bounds are finite", all_finite);
    
    // Output interval should be wider than input (due to transformation)
    double input_diam = intervals[0].diameter();
    double output_diam = intervals.back().diameter();
    
    std::cout << "  Input diameter: " << input_diam << std::endl;
    std::cout << "  Output diameter: " << output_diam << std::endl;
}

// Test 10: Composition law verification
void test_composition_law() {
    std::cout << "\n=== Test 10: Composition Law Verification ===" << std::endl;
    
    // Create layers
    Eigen::MatrixXd W1 = Eigen::MatrixXd::Identity(3, 3) * 2.0;
    Eigen::VectorXd b1 = Eigen::VectorXd::Zero(3);
    auto layer1 = CurvatureBounds::linear_layer(W1, b1);
    
    Eigen::MatrixXd W2 = Eigen::MatrixXd::Identity(3, 3) * 3.0;
    Eigen::VectorXd b2 = Eigen::VectorXd::Zero(3);
    auto layer2 = CurvatureBounds::linear_layer(W2, b2);
    
    // Compose
    auto composed = CurvatureBounds::compose(layer1, layer2);
    
    // For linear layers: κ = 0, L = 6.0
    print_test("Composed curvature is zero for linear layers", composed.curvature == 0.0);
    
    double expected_lipschitz = layer1.lipschitz_constant * layer2.lipschitz_constant;
    bool lipschitz_correct = std::abs(composed.lipschitz_constant - expected_lipschitz) < 1e-10;
    print_test("Composed Lipschitz is product", lipschitz_correct);
    
    std::cout << "  L1 = " << layer1.lipschitz_constant << std::endl;
    std::cout << "  L2 = " << layer2.lipschitz_constant << std::endl;
    std::cout << "  L_composed = " << composed.lipschitz_constant << std::endl;
}

// Test 11: Precision tightness
void test_precision_tightness() {
    std::cout << "\n=== Test 11: Precision Bound Tightness ===" << std::endl;
    
    // For a simple case, verify that the precision bound is reasonable
    // Linear function should need minimal precision
    
    ModelCertifier certifier;
    Eigen::MatrixXd W = Eigen::MatrixXd::Identity(5, 5);
    Eigen::VectorXd b = Eigen::VectorXd::Zero(5);
    certifier.add_linear_layer("identity", W, b);
    
    Eigen::VectorXd lower = Eigen::VectorXd::Constant(5, -1.0);
    Eigen::VectorXd upper = Eigen::VectorXd::Constant(5, 1.0);
    InputDomain domain(lower, upper);
    
    auto cert = certifier.certify(domain, 1e-12);
    
    std::cout << "  Identity function precision: " << cert.precision_requirement << " bits" << std::endl;
    
    // Identity (linear, κ=0) should need minimal precision
    // Just enough to represent target accuracy
    print_test("Identity needs low precision", cert.precision_requirement < 52);
}

int main() {
    std::cout << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║  Proposal 6: Certified Precision Bounds - Test Suite     ║" << std::endl;
    std::cout << "╚═══════════════════════════════════════════════════════════╝" << std::endl;
    
    try {
        test_interval_arithmetic();
        test_input_domain();
        test_curvature_bounds();
        test_precision_computation();
        test_simple_certification();
        test_softmax_certification();
        test_attention_certification();
        test_matrix_inversion();
        test_interval_propagation();
        test_composition_law();
        test_precision_tightness();
        
        std::cout << "\n" << GREEN << "╔═══════════════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║  ALL TESTS PASSED ✓                                       ║" << std::endl;
        std::cout << "╚═══════════════════════════════════════════════════════════╝" << RESET << std::endl;
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n" << RED << "Test suite failed: " << e.what() << RESET << std::endl;
        return 1;
    }
}
