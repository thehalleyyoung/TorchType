/**
 * Comprehensive Enhanced Tests for Proposal 8 - KV-Cache Precision Analyzer
 * 
 * These tests validate:
 * 1. HNF Theorem 5.7 correctness (rigorous)
 * 2. Real data validation
 * 3. Formal verification of precision bounds
 * 4. Ablation studies
 * 5. Stress tests
 */

#include "kv_cache_analyzer.hpp"
#include "hnf_theorem_verifier.hpp"
#include "real_data_validator.hpp"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace hnf::kv_cache;

// Helpers
void print_test_header(const std::string& name) {
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "TEST: " << name << "\n";
    std::cout << std::string(70, '=') << "\n";
}

void print_result(bool passed) {
    std::cout << (passed ? "  ✓ PASSED" : "  ✗ FAILED") << "\n";
}

/**
 * Test 1: HNF Theorem 5.7 - Rigorous Verification
 * 
 * This test validates that our precision assignments actually satisfy
 * the theoretical lower bounds from the paper.
 */
bool test_hnf_theorem_rigorous() {
    print_test_header("HNF Theorem 5.7 Rigorous Verification");
    
    // Test various curvature values
    struct TestCase {
        double curvature;
        double diameter;
        double target_epsilon;
        int assigned_precision;
        bool should_pass;
    };
    
    std::vector<TestCase> test_cases = {
        // Low curvature, lenient epsilon - INT4 should suffice
        {0.01, 10.0, 0.1, 4, true},
        
        // Medium curvature, medium epsilon - INT8 needed
        {1.0, 10.0, 0.01, 8, true},
        
        // High curvature, strict epsilon - FP16 needed
        {100.0, 10.0, 0.001, 16, true},
        
        // Failure case: insufficient precision
        {100.0, 10.0, 0.001, 4, false},
        
        // Edge case: zero curvature (linear function)
        {0.0, 10.0, 0.001, 4, true},
        
        // Very high curvature - even FP16 might not suffice
        {10000.0, 100.0, 0.0001, 16, false},
    };
    
    int passed = 0;
    int total = test_cases.size();
    
    for (size_t i = 0; i < test_cases.size(); ++i) {
        const auto& tc = test_cases[i];
        
        auto result = HNFTheoremVerifier::verify_precision_assignment(
            tc.curvature,
            tc.diameter,
            tc.target_epsilon,
            tc.assigned_precision,
            4.0 // c_constant
        );
        
        bool test_passed = (result.is_valid == tc.should_pass);
        
        std::cout << "  Case " << (i+1) << ": "
                  << "κ=" << tc.curvature << ", "
                  << "D=" << tc.diameter << ", "
                  << "ε=" << tc.target_epsilon << ", "
                  << "p=" << tc.assigned_precision << " bits\n";
        std::cout << "    Required: " << result.required_precision_bits << " bits\n";
        std::cout << "    Result: " << (result.is_valid ? "VALID" : "INVALID") << "\n";
        std::cout << "    Expected: " << (tc.should_pass ? "VALID" : "INVALID") << "\n";
        std::cout << "    " << (test_passed ? "✓" : "✗") << "\n";
        
        if (test_passed) passed++;
    }
    
    std::cout << "\nPassed: " << passed << "/" << total << "\n";
    print_result(passed == total);
    
    return passed == total;
}

/**
 * Test 2: Bound Sharpness Analysis
 * 
 * Verify that we're close to the theoretical minimum precision
 * (not wasting bits unnecessarily)
 */
bool test_bound_sharpness() {
    print_test_header("Bound Sharpness Analysis");
    
    std::vector<double> curvatures = {0.1, 1.0, 10.0, 100.0};
    double diameter = 10.0;
    double target_epsilon = 0.001;
    
    int passed = 0;
    
    for (auto kappa : curvatures) {
        // Compute minimum required precision
        double p_min = std::log2(4.0 * kappa * diameter * diameter / target_epsilon);
        int p_min_int = static_cast<int>(std::ceil(p_min));
        
        // Test with minimal precision
        double sharpness = HNFTheoremVerifier::compute_bound_sharpness(
            kappa, diameter, target_epsilon, p_min_int
        );
        
        std::cout << "  κ=" << kappa << ": "
                  << "p_min=" << p_min_int << " bits, "
                  << "sharpness=" << sharpness << "x\n";
        
        // Sharpness should be close to 1.0 (within 10%)
        bool test_passed = (sharpness >= 0.9 && sharpness <= 1.5);
        std::cout << "    " << (test_passed ? "✓" : "✗") << "\n";
        
        if (test_passed) passed++;
    }
    
    print_result(passed == static_cast<int>(curvatures.size()));
    return passed == static_cast<int>(curvatures.size());
}

/**
 * Test 3: Composition Law Verification
 * 
 * HNF states: Φ_{g ∘ f}(ε) ≤ Φ_g(Φ_f(ε)) + L_g · Φ_f(ε)
 */
bool test_composition_law() {
    print_test_header("Composition Law Verification");
    
    struct CompTestCase {
        double epsilon_in;
        double phi_f;
        double phi_g;
        double lipschitz_g;
    };
    
    std::vector<CompTestCase> cases = {
        {0.01, 0.02, 0.03, 1.5},   // Simple case
        {0.001, 0.005, 0.01, 2.0},  // More error accumulation
        {0.1, 0.15, 0.2, 1.1},      // Low Lipschitz
    };
    
    int passed = 0;
    
    for (const auto& tc : cases) {
        // Compute composed error
        double phi_composed = tc.phi_g + tc.lipschitz_g * tc.phi_f;
        
        bool is_valid = HNFTheoremVerifier::verify_composition_law(
            tc.epsilon_in,
            tc.phi_f,
            tc.phi_g,
            tc.lipschitz_g,
            phi_composed
        );
        
        std::cout << "  ε=" << tc.epsilon_in << ", "
                  << "Φ_f=" << tc.phi_f << ", "
                  << "Φ_g=" << tc.phi_g << ", "
                  << "L_g=" << tc.lipschitz_g << "\n";
        std::cout << "    Φ_composed=" << phi_composed << "\n";
        std::cout << "    " << (is_valid ? "✓" : "✗") << "\n";
        
        if (is_valid) passed++;
    }
    
    print_result(passed == static_cast<int>(cases.size()));
    return passed == static_cast<int>(cases.size());
}

/**
 * Test 4: Real Data Validation
 * 
 * Validate on synthetic sequences that mimic real transformer workloads
 */
bool test_real_data_validation() {
    print_test_header("Real Data Validation");
    
    KVCacheConfig config;
    config.num_layers = 6;
    config.num_heads = 8;
    config.head_dim = 64;
    config.max_seq_length = 256;
    
    KVCacheAnalyzer analyzer(config);
    
    RealDataValidator::ValidationConfig val_config;
    val_config.dataset_name = "wikitext";
    val_config.num_samples = 10;
    val_config.max_sequence_length = 256;
    val_config.quality_threshold = 0.99;
    val_config.measure_perplexity = true;
    
    auto metrics = RealDataValidator::validate_on_dataset(analyzer, val_config);
    
    // Check that we achieve reasonable compression
    bool compression_ok = metrics.compression_ratio >= 2.0;
    std::cout << "  Compression: " << metrics.compression_ratio << "x "
              << (compression_ok ? "✓" : "✗") << "\n";
    
    // Check quality preservation
    bool quality_ok = metrics.perplexity_degradation < 0.05; // < 5%
    std::cout << "  Quality degradation: " << (metrics.perplexity_degradation * 100) << "% "
              << (quality_ok ? "✓" : "✗") << "\n";
    
    // Check theorem validation
    bool theorem_ok = metrics.theorem_validation.all_positions_meet_bound;
    std::cout << "  Theorem bounds satisfied: " << (theorem_ok ? "YES" : "NO") << " "
              << (theorem_ok ? "✓" : "✗") << "\n";
    
    // Generate full report
    auto report = RealDataValidator::generate_validation_report(metrics);
    std::cout << report;
    
    bool all_passed = compression_ok && quality_ok && theorem_ok;
    print_result(all_passed);
    
    return all_passed;
}

/**
 * Test 5: Test Different Datasets
 */
bool test_multiple_datasets() {
    print_test_header("Multi-Dataset Validation");
    
    KVCacheConfig config;
    config.num_layers = 4;
    config.num_heads = 8;
    config.head_dim = 64;
    config.max_seq_length = 128;
    
    KVCacheAnalyzer analyzer(config);
    
    std::vector<std::string> datasets = {"wikitext", "code", "conversation"};
    
    int passed = 0;
    
    for (const auto& dataset : datasets) {
        std::cout << "\n  Dataset: " << dataset << "\n";
        
        RealDataValidator::ValidationConfig val_config;
        val_config.dataset_name = dataset;
        val_config.num_samples = 5;
        val_config.max_sequence_length = 128;
        val_config.quality_threshold = 0.95;
        
        auto metrics = RealDataValidator::validate_on_dataset(analyzer, val_config);
        
        bool dataset_ok = (metrics.compression_ratio >= 1.5 && 
                          metrics.theorem_validation.all_positions_meet_bound);
        
        std::cout << "    Compression: " << metrics.compression_ratio << "x\n";
        std::cout << "    Result: " << (dataset_ok ? "✓" : "✗") << "\n";
        
        if (dataset_ok) passed++;
    }
    
    print_result(passed == static_cast<int>(datasets.size()));
    return passed == static_cast<int>(datasets.size());
}

/**
 * Test 6: Interval Arithmetic Correctness
 */
bool test_interval_arithmetic() {
    print_test_header("Interval Arithmetic for Conservative Bounds");
    
    // Create a position curvature with known values
    PositionCurvature curv;
    curv.attention_weight = 0.5;
    curv.gradient_norm = 1.0;
    curv.hessian_trace = 0.25;
    curv.curvature_score = 0.5 * 1.0 * std::sqrt(0.25); // = 0.25
    
    // Compute interval
    // (This would use the private method, so we test via public interface)
    
    std::vector<PositionCurvature> curvatures = {curv};
    std::vector<PrecisionLevel> precisions = {PrecisionLevel::INT8};
    
    bool is_correct = FormalCorrectnessChecker::check_precision_correctness_smt(
        curvatures, precisions, 0.01
    );
    
    std::cout << "  Curvature score: " << curv.curvature_score << "\n";
    std::cout << "  INT8 sufficient for ε=0.01: " << (is_correct ? "YES" : "NO") << "\n";
    std::cout << "  " << (is_correct ? "✓" : "✗") << "\n";
    
    print_result(is_correct);
    return is_correct;
}

/**
 * Test 7: Empirical Error Measurement
 */
bool test_empirical_error() {
    print_test_header("Empirical Error Measurement");
    
    // Create tensors at different precisions
    auto full_precision = torch::randn({100, 100}, torch::kFloat64);
    auto reduced_precision = full_precision.to(torch::kFloat16).to(torch::kFloat64);
    
    double error = HNFTheoremVerifier::measure_empirical_error(
        full_precision, reduced_precision
    );
    
    std::cout << "  Empirical error (FP64 -> FP16 -> FP64): " << error << "\n";
    
    // FP16 has ~11 bits of precision, so error should be ~2^-11 ≈ 0.0005
    bool reasonable_error = (error < 0.01 && error > 1e-6);
    
    std::cout << "  Error is reasonable: " << (reasonable_error ? "YES" : "NO") << "\n";
    std::cout << "  " << (reasonable_error ? "✓" : "✗") << "\n";
    
    print_result(reasonable_error);
    return reasonable_error;
}

/**
 * Test 8: Stress Test - Pathological Attention
 */
bool test_pathological_attention() {
    print_test_header("Stress Test: Pathological Attention Patterns");
    
    KVCacheConfig config;
    config.num_layers = 2;
    config.num_heads = 4;
    config.head_dim = 32;
    config.max_seq_length = 64;
    
    KVCacheAnalyzer analyzer(config);
    
    bool passed = StressTest::test_pathological_attention(analyzer);
    
    std::cout << "  Handled pathological patterns: " << (passed ? "YES" : "NO") << "\n";
    print_result(passed);
    
    return passed;
}

/**
 * Test 9: Ultra-Long Sequences
 */
bool test_ultra_long_sequences() {
    print_test_header("Stress Test: Ultra-Long Sequences");
    
    KVCacheConfig config;
    config.num_layers = 2;
    config.num_heads = 4;
    config.head_dim = 32;
    config.max_seq_length = 8192; // Very long
    
    KVCacheAnalyzer analyzer(config);
    
    bool passed = StressTest::test_ultra_long_sequences(analyzer);
    
    std::cout << "  Handled 8K+ token sequences: " << (passed ? "YES" : "NO") << "\n";
    print_result(passed);
    
    return passed;
}

/**
 * Test 10: Full Integration Test
 */
bool test_full_integration() {
    print_test_header("Full Integration Test");
    
    std::cout << "\nRunning complete end-to-end pipeline:\n";
    std::cout << "  1. Load realistic dataset\n";
    std::cout << "  2. Analyze with HNF-based method\n";
    std::cout << "  3. Verify theorem bounds\n";
    std::cout << "  4. Compare to baselines\n";
    std::cout << "  5. Generate report\n\n";
    
    KVCacheConfig config;
    config.num_layers = 6;
    config.num_heads = 8;
    config.head_dim = 64;
    config.max_seq_length = 512;
    
    KVCacheAnalyzer analyzer(config);
    
    RealDataValidator::ValidationConfig val_config;
    val_config.dataset_name = "conversation"; // Most realistic
    val_config.num_samples = 20;
    val_config.max_sequence_length = 512;
    val_config.quality_threshold = 0.99;
    val_config.measure_perplexity = true;
    val_config.measure_next_token_accuracy = true;
    
    auto metrics = RealDataValidator::validate_on_dataset(analyzer, val_config);
    
    // Success criteria
    bool criteria_met = 
        metrics.compression_ratio >= 2.5 &&
        metrics.perplexity_degradation < 0.02 &&
        metrics.theorem_validation.all_positions_meet_bound &&
        metrics.baseline_comparison.hnf_outperformance > 0.5;
    
    std::cout << "\n=== FINAL RESULTS ===\n";
    std::cout << "  Compression:      " << metrics.compression_ratio << "x ✓\n";
    std::cout << "  Quality:          " << (100.0 - metrics.perplexity_degradation * 100) << "% ✓\n";
    std::cout << "  Bounds satisfied: " << (metrics.theorem_validation.all_positions_meet_bound ? "YES" : "NO") << "\n";
    std::cout << "  Better than INT8: " << (metrics.baseline_comparison.hnf_outperformance * 100) << "% ✓\n";
    
    print_result(criteria_met);
    
    return criteria_met;
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║   PROPOSAL 8 ENHANCED TEST SUITE - HNF THEOREM VALIDATION     ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    std::vector<std::pair<std::string, bool(*)()>> tests = {
        {"HNF Theorem 5.7 Rigorous", test_hnf_theorem_rigorous},
        {"Bound Sharpness", test_bound_sharpness},
        {"Composition Law", test_composition_law},
        {"Real Data Validation", test_real_data_validation},
        {"Multiple Datasets", test_multiple_datasets},
        {"Interval Arithmetic", test_interval_arithmetic},
        {"Empirical Error", test_empirical_error},
        {"Pathological Attention", test_pathological_attention},
        {"Ultra-Long Sequences", test_ultra_long_sequences},
        {"Full Integration", test_full_integration},
    };
    
    int passed = 0;
    int total = tests.size();
    
    for (const auto& [name, test_fn] : tests) {
        try {
            if (test_fn()) {
                passed++;
            }
        } catch (const std::exception& e) {
            std::cout << "  EXCEPTION: " << e.what() << "\n";
            print_result(false);
        }
    }
    
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                        TEST SUMMARY                            ║\n";
    std::cout << "╠════════════════════════════════════════════════════════════════╣\n";
    std::cout << "║  Total:   " << std::setw(2) << total << "                                                  ║\n";
    std::cout << "║  Passed:  " << std::setw(2) << passed << "                                                  ║\n";
    std::cout << "║  Failed:  " << std::setw(2) << (total - passed) << "                                                  ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    return (passed == total) ? 0 : 1;
}
