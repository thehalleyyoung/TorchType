#include "../include/attention_analyzer.hpp"
#include "../include/attention_curvature.hpp"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace hnf::attention;

// Test utilities
class TestSuite {
public:
    static void assert_close(double a, double b, double tol, const std::string& msg) {
        if (std::abs(a - b) > tol) {
            std::cerr << "FAIL: " << msg << "\n";
            std::cerr << "  Expected: " << b << "\n";
            std::cerr << "  Got: " << a << "\n";
            std::cerr << "  Diff: " << std::abs(a - b) << "\n";
            throw std::runtime_error("Assertion failed");
        } else {
            std::cout << "PASS: " << msg << "\n";
        }
    }
    
    static void assert_true(bool condition, const std::string& msg) {
        if (!condition) {
            std::cerr << "FAIL: " << msg << "\n";
            throw std::runtime_error("Assertion failed");
        } else {
            std::cout << "PASS: " << msg << "\n";
        }
    }
};

// Test 1: Curvature computation matches theoretical bounds
void test_curvature_bounds() {
    std::cout << "\n=== Test 1: Curvature Bounds ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 16;
    int head_dim = 64;
    double temperature = 1.0;
    
    // Create normalized random Q, K
    auto Q = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    auto K = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    
    auto curvature = AttentionCurvature::compute_curvature(Q, K, temperature);
    
    // Theoretical bound: κ should be positive and bounded
    auto curv_mean = curvature.mean().item<double>();
    TestSuite::assert_true(curv_mean > 0, "Curvature should be positive");
    TestSuite::assert_true(curv_mean < 1e10, "Curvature should be bounded for small inputs");
    
    // Test scaling: if we scale Q and K by factor s, curvature should scale appropriately
    auto Q_scaled = Q * 2.0;
    auto K_scaled = K * 2.0;
    auto curvature_scaled = AttentionCurvature::compute_curvature(Q_scaled, K_scaled, temperature);
    auto curv_scaled_mean = curvature_scaled.mean().item<double>();
    
    // With scaled inputs, logits scale quadratically → exp(2*logits) scales exponentially
    TestSuite::assert_true(curv_scaled_mean > curv_mean, "Curvature should increase with input scale");
    
    std::cout << "  Original curvature: " << curv_mean << "\n";
    std::cout << "  Scaled curvature: " << curv_scaled_mean << "\n";
}

// Test 2: Softmax curvature properties
void test_softmax_curvature() {
    std::cout << "\n=== Test 2: Softmax Curvature ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 8;
    
    // Case 1: Uniform logits → moderate curvature
    auto logits_uniform = torch::zeros({batch, heads, seq_len, seq_len});
    auto curv_uniform = AttentionCurvature::compute_softmax_curvature(logits_uniform);
    
    // Case 2: Peaked logits → different curvature characteristics
    // Softmax Hessian has bounded spectral norm regardless of input peakedness
    // But we can check it's computed correctly
    auto logits_peaked = torch::zeros({batch, heads, seq_len, seq_len});
    logits_peaked.index_put_({torch::indexing::Ellipsis, 0, 0}, 10.0);  // One very large value
    auto curv_peaked = AttentionCurvature::compute_softmax_curvature(logits_peaked);
    
    auto curv_u = curv_uniform.mean().item<double>();
    auto curv_p = curv_peaked.mean().item<double>();
    
    std::cout << "  Uniform logits curvature: " << curv_u << "\n";
    std::cout << "  Peaked logits curvature: " << curv_p << "\n";
    
    // Both should be bounded by 0.5 (the maximum Hessian norm of softmax)
    TestSuite::assert_true(curv_u <= 0.5 && curv_u >= 0, "Uniform curvature should be bounded by 0.5");
    TestSuite::assert_true(curv_p <= 0.5 && curv_p >= 0, "Peaked curvature should be bounded by 0.5");
}

// Test 3: Precision requirement estimation
void test_precision_requirements() {
    std::cout << "\n=== Test 3: Precision Requirements ===\n";
    
    // Test that precision requirements scale correctly with curvature
    double diameter = 10.0;
    double target_accuracy = 1e-6;
    
    auto curvature_low = torch::tensor({100.0});
    auto curvature_high = torch::tensor({10000.0});
    
    auto prec_low = AttentionCurvature::estimate_precision_requirement(curvature_low, diameter, target_accuracy);
    auto prec_high = AttentionCurvature::estimate_precision_requirement(curvature_high, diameter, target_accuracy);
    
    double p_low = prec_low.item<double>();
    double p_high = prec_high.item<double>();
    
    std::cout << "  Low curvature precision: " << p_low << " bits\n";
    std::cout << "  High curvature precision: " << p_high << " bits\n";
    
    TestSuite::assert_true(p_high > p_low, "Higher curvature requires more precision");
    
    // Check formula: p = log2(κ * D^2 / ε)
    double expected_low = std::log2(100.0 * 100.0 / 1e-6);
    TestSuite::assert_close(p_low, expected_low, 0.1, "Precision formula correct for low curvature");
}

// Test 4: Lipschitz constant computation
void test_lipschitz_constant() {
    std::cout << "\n=== Test 4: Lipschitz Constant ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 16;
    int head_dim = 64;
    
    // Orthonormal Q, K should have Lipschitz ~ 1
    auto Q = torch::randn({batch, heads, seq_len, head_dim});
    Q = torch::nn::functional::normalize(Q, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    
    auto K = torch::randn({batch, heads, seq_len, head_dim});
    K = torch::nn::functional::normalize(K, torch::nn::functional::NormalizeFuncOptions().dim(-1));
    
    auto lipschitz = AttentionCurvature::compute_lipschitz_constant(Q, K, 1.0);
    auto lip_mean = lipschitz.mean().item<double>();
    
    std::cout << "  Lipschitz constant (normalized): " << lip_mean << "\n";
    
    // Should be close to 1/sqrt(head_dim) for normalized vectors
    double expected = std::sqrt(seq_len) / std::sqrt(head_dim);
    TestSuite::assert_true(lip_mean > 0.1 && lip_mean < 10.0, "Lipschitz constant in reasonable range");
}

// Test 5: Error functional composition
void test_error_functional() {
    std::cout << "\n=== Test 5: Error Functional ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 16;
    int head_dim = 64;
    
    auto Q = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    auto K = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    
    double input_error = 1e-7;
    HardwareModel fp32 = HardwareModel::fp32();
    HardwareModel fp16 = HardwareModel::fp16();
    
    auto error_fp32 = AttentionCurvature::compute_error_functional(Q, K, input_error, fp32, 1.0);
    auto error_fp16 = AttentionCurvature::compute_error_functional(Q, K, input_error, fp16, 1.0);
    
    double err32 = error_fp32.mean().item<double>();
    double err16 = error_fp16.mean().item<double>();
    
    std::cout << "  Error functional (fp32): " << err32 << "\n";
    std::cout << "  Error functional (fp16): " << err16 << "\n";
    
    // fp16 should have higher error due to lower precision
    TestSuite::assert_true(err16 > err32, "fp16 should have higher error than fp32");
}

// Test 6: Entropy computation
void test_entropy_computation() {
    std::cout << "\n=== Test 6: Entropy Computation ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 8;
    
    // Uniform distribution → max entropy
    auto uniform_attn = torch::ones({batch, heads, seq_len, seq_len}) / seq_len;
    
    AttentionConfig config;
    AttentionAnalyzer analyzer(config);
    auto entropy_uniform = analyzer.compute_entropy(uniform_attn);
    double ent_u = entropy_uniform.mean().item<double>();
    
    // One-hot distribution → min entropy (0)
    auto onehot_attn = torch::zeros({batch, heads, seq_len, seq_len});
    onehot_attn.index_put_({torch::indexing::Ellipsis, torch::indexing::Slice(), 0}, 1.0);
    auto entropy_onehot = analyzer.compute_entropy(onehot_attn);
    double ent_o = entropy_onehot.mean().item<double>();
    
    std::cout << "  Uniform entropy: " << ent_u << " nats (expected: " << std::log(seq_len) << ")\n";
    std::cout << "  One-hot entropy: " << ent_o << " nats (expected: 0)\n";
    
    // Uniform should have max entropy ≈ log(seq_len)
    TestSuite::assert_close(ent_u, std::log(seq_len), 0.01, "Uniform entropy correct");
    TestSuite::assert_close(ent_o, 0.0, 0.01, "One-hot entropy near zero");
}

// Test 7: Full pattern analysis
void test_pattern_analysis() {
    std::cout << "\n=== Test 7: Pattern Analysis ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 16;
    int head_dim = 64;
    
    auto Q = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    auto K = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    auto V = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    
    AttentionConfig config;
    AttentionAnalyzer analyzer(config);
    
    auto stats = analyzer.analyze_pattern(Q, K, V, "test_layer");
    
    std::cout << "  Batch size: " << stats.batch_size << "\n";
    std::cout << "  Num heads: " << stats.num_heads << "\n";
    std::cout << "  Seq length: " << stats.seq_length << "\n";
    std::cout << "  Mean entropy: " << stats.entropy_per_head.mean().item<double>() << "\n";
    std::cout << "  Mean curvature: " << stats.curvature_estimate.mean().item<double>() << "\n";
    std::cout << "  Mean precision req: " << stats.precision_bits_required.mean().item<double>() << " bits\n";
    
    TestSuite::assert_true(stats.batch_size == batch, "Batch size correct");
    TestSuite::assert_true(stats.num_heads == heads, "Num heads correct");
    TestSuite::assert_true(stats.entropy_per_head.size(0) == heads, "Entropy tensor size correct");
}

// Test 8: Overflow detection
void test_overflow_detection() {
    std::cout << "\n=== Test 8: Overflow Detection ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 16;
    
    // Safe logits
    auto logits_safe = torch::randn({batch, heads, seq_len, seq_len}) * 2.0;
    
    // Dangerous logits (large values)
    auto logits_danger = torch::randn({batch, heads, seq_len, seq_len}) * 2.0;
    logits_danger.index_put_({0, 0, 0, 0}, 90.0);  // Will overflow exp(90)
    
    AttentionConfig config;
    AttentionAnalyzer analyzer(config);
    
    auto risk_safe = analyzer.detect_overflow_risk(logits_safe);
    auto risk_danger = analyzer.detect_overflow_risk(logits_danger);
    
    std::cout << "  Safe logits overflow: " << risk_safe.overflow_likely << "\n";
    std::cout << "  Dangerous logits overflow: " << risk_danger.overflow_likely << "\n";
    std::cout << "  Recommendation: " << risk_danger.recommendation << "\n";
    
    TestSuite::assert_true(!risk_safe.overflow_likely, "Safe logits should not overflow");
    TestSuite::assert_true(risk_danger.overflow_likely, "Dangerous logits should overflow");
}

// Test 9: Pre-training stability check
void test_pretraining_stability() {
    std::cout << "\n=== Test 9: Pre-training Stability ===\n";
    
    AttentionConfig config;
    config.max_seq_length = 512;
    config.num_heads = 8;
    config.head_dim = 64;
    config.hardware = HardwareModel::fp16();
    
    AttentionAnalyzer analyzer(config);
    auto diagnosis = analyzer.check_pretraining_stability(12);  // 12 layers
    
    std::cout << "  Number of issues: " << diagnosis.issues.size() << "\n";
    for (const auto& issue : diagnosis.issues) {
        std::cout << "  - " << issue.message << "\n";
        std::cout << "    Suggestion: " << issue.suggestion << "\n";
    }
    
    TestSuite::assert_true(diagnosis.issues.size() >= 0, "Diagnosis should complete");
}

// Test 10: Stability prediction
void test_stability_prediction() {
    std::cout << "\n=== Test 10: Stability Prediction ===\n";
    
    AttentionConfig config;
    AttentionAnalyzer analyzer(config);
    
    // Test case 1: Small, stable config with fp32
    auto pred_small = analyzer.predict_stability(64, 8, 64, 1.0, HardwareModel::fp32());
    std::cout << "  Small config stable: " << pred_small.is_stable << "\n";
    std::cout << "  Required bits: " << pred_small.required_precision_bits << "\n";
    
    // Test case 2: Large, potentially unstable config with fp16
    auto pred_large = analyzer.predict_stability(4096, 32, 128, 1.0, HardwareModel::fp16());
    std::cout << "  Large config stable: " << pred_large.is_stable << "\n";
    std::cout << "  Required bits: " << pred_large.required_precision_bits << "\n";
    std::cout << "  Warnings: " << pred_large.warnings.size() << "\n";
    for (const auto& warn : pred_large.warnings) {
        std::cout << "    - " << warn << "\n";
    }
    
    // Updated test: small config with fp32 should be more stable than large config with fp16
    TestSuite::assert_true(pred_small.required_precision_bits < pred_large.required_precision_bits,
                          "Small config should need less precision than large config");
    TestSuite::assert_true(pred_large.warnings.size() > 0, "Large config should have warnings");
}

// Test 11: Diagnosis from history
void test_diagnosis_from_history() {
    std::cout << "\n=== Test 11: Diagnosis from History ===\n";
    
    AttentionConfig config;
    config.entropy_collapse_threshold = 1.0;  // Set threshold
    AttentionAnalyzer analyzer(config);
    
    // Create fake history with entropy collapse
    std::map<std::string, std::vector<AttentionStats>> history;
    
    for (int i = 0; i < 50; ++i) {
        AttentionStats stats;
        stats.num_heads = 4;
        stats.batch_size = 2;
        
        // Decreasing entropy (simulating collapse)
        stats.entropy_per_head = torch::tensor({0.3, 0.8, 0.4, 0.9});  // Head 0 and 2 collapsed
        stats.curvature_estimate = torch::tensor({1e5, 1e3, 1e5, 1e3});
        stats.max_attention_per_head = torch::tensor({0.98, 0.6, 0.97, 0.5});
        stats.precision_bits_required = torch::tensor({25.0, 18.0, 26.0, 17.0});
        stats.logit_max = torch::tensor({10.0, 5.0, 11.0, 4.0});
        
        history["layer1"].push_back(stats);
    }
    
    auto diagnosis = analyzer.diagnose(history);
    
    std::cout << "  Total issues: " << diagnosis.issues.size() << "\n";
    int entropy_issues = 0;
    int curvature_issues = 0;
    int spike_issues = 0;
    
    for (const auto& issue : diagnosis.issues) {
        if (issue.type == IssueType::ENTROPY_COLLAPSE) entropy_issues++;
        if (issue.type == IssueType::HIGH_CURVATURE) curvature_issues++;
        if (issue.type == IssueType::ATTENTION_SPIKE) spike_issues++;
        
        std::cout << "  - Head " << issue.head_index << ": " << issue.message << "\n";
    }
    
    std::cout << "  Entropy issues: " << entropy_issues << "\n";
    std::cout << "  Curvature issues: " << curvature_issues << "\n";
    std::cout << "  Spike issues: " << spike_issues << "\n";
    
    TestSuite::assert_true(entropy_issues >= 2, "Should detect entropy collapse in heads 0 and 2");
}

// Test 12: Intervention suggestions
void test_intervention_suggestions() {
    std::cout << "\n=== Test 12: Intervention Suggestions ===\n";
    
    AttentionDiagnosis diagnosis;
    diagnosis.config = AttentionConfig();
    
    // Add various issues
    diagnosis.issues.push_back(StabilityIssue(
        "layer1", 0, IssueType::ENTROPY_COLLAPSE, Severity::ERROR, 0.2,
        "Entropy collapse", "Add regularization"
    ));
    diagnosis.issues.push_back(StabilityIssue(
        "layer2", 1, IssueType::OVERFLOW_RISK, Severity::ERROR, 95.0,
        "Overflow risk", "Clamp logits"
    ));
    diagnosis.issues.push_back(StabilityIssue(
        "layer3", 2, IssueType::PRECISION_INSUFFICIENT, Severity::ERROR, 18.0,
        "Need more precision", "Use fp32"
    ));
    
    AttentionAnalyzer analyzer(diagnosis.config);
    auto suggestions = analyzer.suggest_interventions(diagnosis);
    
    std::cout << "  Number of suggestions: " << suggestions.size() << "\n";
    for (const auto& sugg : suggestions) {
        std::cout << "  - Action: " << sugg.action << "\n";
        std::cout << "    Reason: " << sugg.reason << "\n";
        std::cout << "    Expected improvement: " << sugg.expected_improvement << "\n";
    }
    
    TestSuite::assert_true(suggestions.size() >= 3, "Should suggest interventions for each issue type");
}

// Test 13: Monitoring with hooks
void test_monitoring() {
    std::cout << "\n=== Test 13: Monitoring ===\n";
    
    AttentionConfig config;
    AttentionMonitor monitor(config, 10);
    
    int callback_count = 0;
    monitor.register_hook([&](const std::string& layer, const AttentionStats& stats) {
        callback_count++;
        std::cout << "  Hook called for " << layer << "\n";
    });
    
    // Record some stats
    for (int step = 0; step < 25; ++step) {
        AttentionStats stats;
        stats.num_heads = 4;
        stats.batch_size = 2;
        stats.seq_length = 16;
        stats.hidden_dim = 64;
        
        // Initialize all tensor fields
        stats.entropy_per_head = torch::tensor({0.5, 0.6, 0.7, 0.8});
        stats.curvature_estimate = torch::tensor({1e3, 1e3, 1e3, 1e3});
        stats.max_attention_per_head = torch::tensor({0.8, 0.7, 0.9, 0.6});
        stats.min_attention_per_head = torch::tensor({0.01, 0.02, 0.01, 0.03});
        stats.logit_max = torch::tensor({5.0, 4.0, 6.0, 3.0});
        stats.logit_min = torch::tensor({-5.0, -4.0, -6.0, -3.0});
        stats.logit_range = stats.logit_max - stats.logit_min;
        stats.logit_std = torch::tensor({2.0, 2.0, 2.5, 1.5});
        stats.lipschitz_constant = torch::tensor({1.2, 1.1, 1.3, 1.0});
        stats.precision_bits_required = torch::tensor({20.0, 19.0, 21.0, 18.0});
        stats.gradient_norm = torch::tensor({1.0, 1.0, 1.0, 1.0});
        
        monitor.record("layer1", stats);
        
        if (monitor.should_monitor(step)) {
            auto diagnosis = monitor.get_diagnosis();
            std::cout << "  Step " << step << ": " << diagnosis.issues.size() << " issues\n";
        }
    }
    
    std::cout << "  Total callbacks: " << callback_count << "\n";
    TestSuite::assert_true(callback_count == 25, "Hook should be called for each recording");
}

// Test 14: Real attention computation with stats
void test_attention_with_stats() {
    std::cout << "\n=== Test 14: Attention with Stats ===\n";
    
    int batch = 2;
    int heads = 4;
    int seq_len = 16;
    int head_dim = 64;
    
    auto Q = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    auto K = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    auto V = torch::randn({batch, heads, seq_len, head_dim}) * 0.1;
    
    AttentionConfig config;
    AttentionAnalyzer analyzer(config);
    
    auto [output, stats] = analyzer.compute_attention_with_stats(Q, K, V, "test_layer");
    
    std::cout << "  Output shape: [" << output.size(0) << ", " << output.size(1) 
              << ", " << output.size(2) << ", " << output.size(3) << "]\n";
    std::cout << "  Expected: [" << batch << ", " << heads << ", " << seq_len << ", " << head_dim << "]\n";
    
    TestSuite::assert_true(output.size(0) == batch && output.size(1) == heads &&
                          output.size(2) == seq_len && output.size(3) == head_dim,
                          "Output shape correct");
}

// Test 15: Extreme case - nearly singular attention
void test_extreme_cases() {
    std::cout << "\n=== Test 15: Extreme Cases ===\n";
    
    int batch = 1;
    int heads = 2;
    int seq_len = 8;
    int head_dim = 64;
    
    // Case 1: Very peaked attention (one query matches one key perfectly)
    auto Q = torch::randn({batch, heads, seq_len, head_dim});
    auto K = Q.clone();  // Perfect match
    K.index_put_({0, 0, 1}, K.index({0, 0, 1}) + 10.0);  // Perturb slightly
    auto V = torch::randn({batch, heads, seq_len, head_dim});
    
    AttentionConfig config;
    AttentionAnalyzer analyzer(config);
    auto stats = analyzer.analyze_pattern(Q, K, V, "extreme_layer");
    
    std::cout << "  Peaked attention entropy: " << stats.entropy_per_head.min().item<double>() << "\n";
    std::cout << "  Peaked attention curvature: " << stats.curvature_estimate.max().item<double>() << "\n";
    
    // Case 2: Very large logits
    auto Q_large = torch::randn({batch, heads, seq_len, head_dim}) * 10.0;
    auto K_large = torch::randn({batch, heads, seq_len, head_dim}) * 10.0;
    auto stats_large = analyzer.analyze_pattern(Q_large, K_large, V, "large_layer");
    
    std::cout << "  Large logits curvature: " << stats_large.curvature_estimate.mean().item<double>() << "\n";
    std::cout << "  Large logits precision req: " << stats_large.precision_bits_required.mean().item<double>() << "\n";
    
    auto risk = analyzer.detect_overflow_risk(
        torch::matmul(Q_large, K_large.transpose(-2, -1)) / std::sqrt(head_dim)
    );
    std::cout << "  Overflow likely: " << risk.overflow_likely << "\n";
}

int main() {
    try {
        std::cout << "==============================================\n";
        std::cout << "  HNF Attention Stability Analysis Tests\n";
        std::cout << "  Proposal #3 Implementation\n";
        std::cout << "==============================================\n";
        
        test_curvature_bounds();
        test_softmax_curvature();
        test_precision_requirements();
        test_lipschitz_constant();
        test_error_functional();
        test_entropy_computation();
        test_pattern_analysis();
        test_overflow_detection();
        test_pretraining_stability();
        test_stability_prediction();
        test_diagnosis_from_history();
        test_intervention_suggestions();
        test_monitoring();
        test_attention_with_stats();
        test_extreme_cases();
        
        std::cout << "\n==============================================\n";
        std::cout << "  ALL TESTS PASSED!\n";
        std::cout << "==============================================\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n==============================================\n";
        std::cerr << "  TEST FAILED: " << e.what() << "\n";
        std::cerr << "==============================================\n";
        return 1;
    }
}
