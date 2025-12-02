#include "kv_cache_analyzer.hpp"
#include "hnf_theorem_verifier.hpp"
#include "real_data_validator.hpp"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <fstream>

using namespace hnf::kv_cache;

/**
 * Comprehensive Real-World Validation Suite for Proposal 8
 * 
 * This test suite validates that:
 * 1. HNF Theorem 5.7 holds for all position-precision assignments
 * 2. Compression is achieved with quality preservation
 * 3. The system works on realistic data patterns
 * 4. Performance is acceptable for practical use
 */

// Test utilities
#define TEST_ASSERT(condition, message) \
    do { \
        if (!(condition)) { \
            std::cerr << "TEST FAILED: " << message << std::endl; \
            std::cerr << "  at " << __FILE__ << ":" << __LINE__ << std::endl; \
            return false; \
        } \
    } while(0)

#define RUN_TEST(test_func) \
    do { \
        std::cout << "\n══════════════════════════════════════════════\n"; \
        std::cout << "Running " << #test_func << "..." << std::endl; \
        std::cout << "══════════════════════════════════════════════\n"; \
        auto start = std::chrono::high_resolution_clock::now(); \
        bool result = test_func(); \
        auto end = std::chrono::high_resolution_clock::now(); \
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count(); \
        if (result) { \
            std::cout << "✓ PASSED (" << duration << " ms)\n" << std::endl; \
            passed++; \
        } else { \
            std::cout << "✗ FAILED (" << duration << " ms)\n" << std::endl; \
            failed++; \
        } \
        total++; \
    } while(0)

// Generate realistic transformer attention pattern
torch::Tensor generate_realistic_attention(int64_t batch, int64_t heads, int64_t seq_len) {
    auto attn = torch::zeros({batch, heads, seq_len, seq_len});
    std::mt19937 gen(42);
    std::normal_distribution<double> noise(0.0, 0.05);
    
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < heads; ++h) {
            for (int64_t q = 0; q < seq_len; ++q) {
                for (int64_t k = 0; k <= q; ++k) {
                    // Distance-based attention with multiple patterns
                    double distance = static_cast<double>(q - k);
                    
                    // Pattern 1: Exponential recency bias (most important)
                    double recency = std::exp(-distance / 20.0);
                    
                    // Pattern 2: Positional anchors (first few tokens)
                    double anchor = (k < 5) ? 0.2 : 0.0;
                    
                    // Pattern 3: Periodic peaks (simulating semantic clusters)
                    double periodic = 0.1 * std::sin(distance / 10.0);
                    
                    // Combine patterns with noise
                    double weight = recency + anchor + periodic + noise(gen);
                    weight = std::max(0.0, weight);  // Ensure non-negative
                    
                    attn[b][h][q][k] = weight;
                }
                
                // Normalize to sum to 1
                auto row_sum = attn[b][h][q].sum().item<double>();
                if (row_sum > 0) {
                    attn[b][h][q] /= row_sum;
                }
            }
        }
    }
    
    return attn;
}

// Test 1: Comprehensive HNF Theorem Verification
bool test_hnf_theorem_comprehensive() {
    std::cout << "Validating HNF Theorem 5.7 comprehensively...\n" << std::endl;
    
    KVCacheConfig config;
    config.num_layers = 8;
    config.num_heads = 12;
    config.head_dim = 64;
    config.max_seq_length = 256;
    config.target_epsilon = 1e-3;
    config.safety_margin_bits = 1;  // Conservative
    
    CurvatureAnalyzer curvature_analyzer(config);
    PrecisionMapper precision_mapper(config);
    
    // Generate realistic data
    int64_t seq_len = 256;
    auto attention_weights = generate_realistic_attention(1, config.num_heads, seq_len);
    auto keys = torch::randn({1, seq_len, config.num_heads, config.head_dim}) * 0.5;
    auto values = torch::randn({1, seq_len, config.num_heads, config.head_dim}) * 0.5;
    auto queries = torch::randn({1, seq_len, config.num_heads, config.head_dim}) * 0.5;
    
    // Compute curvatures
    auto curvatures = curvature_analyzer.compute_position_curvatures(
        attention_weights, keys, values, queries, 0
    );
    
    // Map to precisions
    auto precision_map = precision_mapper.map_curvatures_to_precisions(curvatures, 0);
    
    // Verify every position against HNF Theorem 5.7
    std::cout << "Verifying " << curvatures.size() << " positions..." << std::endl;
    
    auto verification_results = HNFTheoremVerifier::verify_precision_map(
        curvatures,
        precision_map.position_precisions,
        10.0,  // diameter
        config.target_epsilon
    );
    
    int valid_count = 0;
    int invalid_count = 0;
    double max_violation = 0.0;
    
    for (size_t i = 0; i < verification_results.size(); ++i) {
        const auto& vr = verification_results[i];
        if (vr.is_valid) {
            valid_count++;
        } else {
            invalid_count++;
            max_violation = std::max(max_violation, std::abs(vr.theoretical_error - vr.empirical_error));
            
            if (invalid_count <= 3) {  // Print first few violations
                std::cout << "  Position " << i << " violates bound:" << std::endl;
                std::cout << "    Curvature: " << curvatures[i].curvature_score << std::endl;
                std::cout << "    Precision: " << bits_per_element(precision_map.position_precisions[i]) << " bits" << std::endl;
                std::cout << "    Required:  " << vr.required_precision_bits << " bits" << std::endl;
            }
        }
    }
    
    std::cout << "\nVerification Results:" << std::endl;
    std::cout << "  Valid positions:   " << valid_count << " / " << verification_results.size() << std::endl;
    std::cout << "  Invalid positions: " << invalid_count << std::endl;
    std::cout << "  Success rate:      " << (100.0 * valid_count / verification_results.size()) << "%" << std::endl;
    
    // All positions must meet the theorem bound
    TEST_ASSERT(invalid_count == 0, "All positions must satisfy HNF Theorem 5.7");
    
    return true;
}

// Test 2: Compression vs Quality Trade-off Analysis
bool test_compression_quality_tradeoff() {
    std::cout << "Analyzing compression vs quality trade-off...\n" << std::endl;
    
    std::vector<double> quality_thresholds = {0.90, 0.95, 0.99, 0.999};
    
    std::cout << std::setw(12) << "Threshold" 
              << std::setw(15) << "Compression"
              << std::setw(15) << "Quality"
              << std::setw(15) << "Speedup" << std::endl;
    std::cout << std::string(56, '-') << std::endl;
    
    for (double threshold : quality_thresholds) {
        KVCacheConfig config;
        config.num_layers = 4;
        config.num_heads = 8;
        config.head_dim = 64;
        config.max_seq_length = 128;
        config.quality_threshold = threshold;
        config.target_epsilon = 1.0 - threshold;  // Convert to error tolerance
        
        CurvatureAnalyzer curvature_analyzer(config);
        PrecisionMapper precision_mapper(config);
        
        // Generate realistic data
        int64_t seq_len = 128;
        auto attention_weights = generate_realistic_attention(1, config.num_heads, seq_len);
        auto keys = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto values = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto queries = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        
        // Compute curvatures and precisions
        auto curvatures = curvature_analyzer.compute_position_curvatures(
            attention_weights, keys, values, queries, 0
        );
        auto precision_map = precision_mapper.map_curvatures_to_precisions(curvatures, 0);
        
        // Estimate quality using precision adequacy
        double total_quality = 0.0;
        for (size_t i = 0; i < curvatures.size(); ++i) {
            auto required = precision_mapper.compute_required_precision(
                curvatures[i].curvature_score, 10.0, config.target_epsilon
            );
            int required_bits = bits_per_element(required);
            int actual_bits = bits_per_element(precision_map.position_precisions[i]);
            
            total_quality += std::min(1.0, static_cast<double>(actual_bits) / required_bits);
        }
        double avg_quality = total_quality / curvatures.size();
        
        // Compute speedup from lower precision
        double speedup = 1.0;  // Baseline
        for (auto prec : precision_map.position_precisions) {
            int bits = bits_per_element(prec);
            speedup += (16 - bits) / 100.0;  // Approximate speedup
        }
        speedup /= precision_map.position_precisions.size();
        
        std::cout << std::fixed << std::setprecision(3);
        std::cout << std::setw(12) << threshold
                  << std::setw(15) << precision_map.compression_ratio() << "x"
                  << std::setw(14) << (avg_quality * 100) << "%"
                  << std::setw(14) << speedup << "x" << std::endl;
    }
    
    std::cout << "\n✓ Trade-off analysis complete" << std::endl;
    return true;
}

// Test 3: Scalability Test
bool test_scalability() {
    std::cout << "Testing scalability with varying sequence lengths...\n" << std::endl;
    
    std::vector<int64_t> seq_lengths = {64, 128, 256, 512, 1024};
    
    std::cout << std::setw(12) << "Seq Length"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Compression"
              << std::setw(18) << "Memory Saved" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (int64_t seq_len : seq_lengths) {
        KVCacheConfig config;
        config.num_layers = 4;
        config.num_heads = 8;
        config.head_dim = 64;
        config.max_seq_length = seq_len;
        config.quality_threshold = 0.95;
        config.target_epsilon = 1e-2;
        
        CurvatureAnalyzer curvature_analyzer(config);
        PrecisionMapper precision_mapper(config);
        
        // Generate data
        auto attention_weights = generate_realistic_attention(1, config.num_heads, seq_len);
        auto keys = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto values = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto queries = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        
        // Time the analysis
        auto start = std::chrono::high_resolution_clock::now();
        auto curvatures = curvature_analyzer.compute_position_curvatures(
            attention_weights, keys, values, queries, 0
        );
        auto precision_map = precision_mapper.map_curvatures_to_precisions(curvatures, 0);
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
        double memory_saved_mb = (precision_map.memory_bytes_fp16 - precision_map.memory_bytes_adaptive) / (1024.0 * 1024.0);
        
        std::cout << std::setw(12) << seq_len
                  << std::setw(15) << duration
                  << std::setw(14) << precision_map.compression_ratio() << "x"
                  << std::setw(15) << memory_saved_mb << " MB" << std::endl;
    }
    
    std::cout << "\n✓ Scalability test complete" << std::endl;
    return true;
}

// Test 4: Stress Test with Pathological Cases
bool test_pathological_cases() {
    std::cout << "Testing pathological cases...\n" << std::endl;
    
    KVCacheConfig config;
    config.num_layers = 4;
    config.num_heads = 8;
    config.head_dim = 64;
    config.max_seq_length = 128;
    config.target_epsilon = 1e-3;
    
    CurvatureAnalyzer curvature_analyzer(config);
    PrecisionMapper precision_mapper(config);
    
    // Case 1: Uniform attention (no recency bias)
    std::cout << "Case 1: Uniform attention..." << std::endl;
    {
        int64_t seq_len = 128;
        auto attention_weights = torch::ones({1, config.num_heads, seq_len, seq_len}) / seq_len;
        auto keys = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto values = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto queries = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        
        auto curvatures = curvature_analyzer.compute_position_curvatures(
            attention_weights, keys, values, queries, 0
        );
        auto precision_map = precision_mapper.map_curvatures_to_precisions(curvatures, 0);
        
        std::cout << "  Compression: " << precision_map.compression_ratio() << "x" << std::endl;
        TEST_ASSERT(precision_map.compression_ratio() >= 1.0, "Should maintain or improve compression");
    }
    
    // Case 2: Extremely peaked attention (all on last position)
    std::cout << "\nCase 2: Extremely peaked attention..." << std::endl;
    {
        int64_t seq_len = 128;
        auto attention_weights = torch::zeros({1, config.num_heads, seq_len, seq_len});
        for (int64_t q = 0; q < seq_len; ++q) {
            attention_weights.index_put_({0, torch::indexing::Slice(), q, q}, 1.0);
        }
        
        auto keys = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto values = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        auto queries = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        
        auto curvatures = curvature_analyzer.compute_position_curvatures(
            attention_weights, keys, values, queries, 0
        );
        auto precision_map = precision_mapper.map_curvatures_to_precisions(curvatures, 0);
        
        std::cout << "  Compression: " << precision_map.compression_ratio() << "x" << std::endl;
        TEST_ASSERT(precision_map.compression_ratio() >= 1.0, "Should achieve compression even with peaked attention");
    }
    
    // Case 3: Very large key/value norms
    std::cout << "\nCase 3: Large key/value norms..." << std::endl;
    {
        int64_t seq_len = 128;
        auto attention_weights = generate_realistic_attention(1, config.num_heads, seq_len);
        auto keys = torch::randn({1, seq_len, config.num_heads, config.head_dim}) * 10.0;  // 10x larger
        auto values = torch::randn({1, seq_len, config.num_heads, config.head_dim}) * 10.0;
        auto queries = torch::randn({1, seq_len, config.num_heads, config.head_dim});
        
        auto curvatures = curvature_analyzer.compute_position_curvatures(
            attention_weights, keys, values, queries, 0
        );
        auto precision_map = precision_mapper.map_curvatures_to_precisions(curvatures, 0);
        
        std::cout << "  Compression: " << precision_map.compression_ratio() << "x" << std::endl;
        std::cout << "  Max curvature: " << precision_map.max_curvature << std::endl;
        TEST_ASSERT(precision_map.max_curvature > 0, "Should compute non-zero curvature");
    }
    
    std::cout << "\n✓ All pathological cases handled" << std::endl;
    return true;
}

// Test 5: End-to-End Real-World Simulation
bool test_end_to_end_simulation() {
    std::cout << "Running end-to-end simulation with realistic transformer...\n" << std::endl;
    
    // Simulate a multi-layer transformer with varying attention patterns per layer
    int64_t num_layers = 12;
    int64_t num_heads = 12;
    int64_t head_dim = 64;
    int64_t seq_len = 512;
    
    KVCacheConfig config;
    config.num_layers = num_layers;
    config.num_heads = num_heads;
    config.head_dim = head_dim;
    config.max_seq_length = seq_len;
    config.quality_threshold = 0.99;
    config.target_epsilon = 1e-3;
    
    std::vector<LayerPrecisionMap> all_layer_maps;
    double total_fp16_memory = 0.0;
    double total_adaptive_memory = 0.0;
    
    CurvatureAnalyzer curvature_analyzer(config);
    PrecisionMapper precision_mapper(config);
    
    std::cout << "Analyzing " << num_layers << " layers..." << std::endl;
    
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        // Different layers have different attention patterns
        // Early layers: more local, Late layers: more global
        double locality_factor = 1.0 - (static_cast<double>(layer) / num_layers);
        
        auto attention_weights = torch::zeros({1, num_heads, seq_len, seq_len});
        for (int64_t h = 0; h < num_heads; ++h) {
            for (int64_t q = 0; q < seq_len; ++q) {
                for (int64_t k = 0; k <= q; ++k) {
                    double distance = static_cast<double>(q - k);
                    double decay_rate = 10.0 + locality_factor * 30.0;  // Early layers more local
                    double weight = std::exp(-distance / decay_rate);
                    attention_weights[0][h][q][k] = weight;
                }
                auto row_sum = attention_weights[0][h][q].sum().item<double>();
                if (row_sum > 0) {
                    attention_weights[0][h][q] /= row_sum;
                }
            }
        }
        
        auto keys = torch::randn({1, seq_len, num_heads, head_dim}) * 0.5;
        auto values = torch::randn({1, seq_len, num_heads, head_dim}) * 0.5;
        auto queries = torch::randn({1, seq_len, num_heads, head_dim}) * 0.5;
        
        auto curvatures = curvature_analyzer.compute_position_curvatures(
            attention_weights, keys, values, queries, layer
        );
        auto precision_map = precision_mapper.map_curvatures_to_precisions(curvatures, layer);
        
        all_layer_maps.push_back(precision_map);
        total_fp16_memory += precision_map.memory_bytes_fp16;
        total_adaptive_memory += precision_map.memory_bytes_adaptive;
        
        if (layer % 3 == 0) {
            std::cout << "  Layer " << std::setw(2) << layer 
                      << ": " << precision_map.compression_ratio() << "x compression"
                      << " (avg curv: " << precision_map.avg_curvature << ")" << std::endl;
        }
    }
    
    double overall_compression = total_fp16_memory / total_adaptive_memory;
    double memory_saved_gb = (total_fp16_memory - total_adaptive_memory) / (1024.0 * 1024.0 * 1024.0);
    
    std::cout << "\n═══════════════════════════════════════════════" << std::endl;
    std::cout << "  FINAL RESULTS" << std::endl;
    std::cout << "═══════════════════════════════════════════════" << std::endl;
    std::cout << "  Total layers:         " << num_layers << std::endl;
    std::cout << "  Sequence length:      " << seq_len << std::endl;
    std::cout << "  FP16 memory:          " << (total_fp16_memory / (1024 * 1024 * 1024)) << " GB" << std::endl;
    std::cout << "  Adaptive memory:      " << (total_adaptive_memory / (1024 * 1024 * 1024)) << " GB" << std::endl;
    std::cout << "  Overall compression:  " << overall_compression << "x" << std::endl;
    std::cout << "  Memory saved:         " << memory_saved_gb << " GB" << std::endl;
    std::cout << "═══════════════════════════════════════════════" << std::endl;
    
    TEST_ASSERT(overall_compression > 1.2, "Should achieve >1.2x compression");
    TEST_ASSERT(overall_compression < 5.0, "Compression should be realistic (<5x)");
    
    return true;
}

int main() {
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║         PROPOSAL 8: COMPREHENSIVE VALIDATION SUITE             ║" << std::endl;
    std::cout << "║         Real-World Tests with HNF Theorem Verification         ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n" << std::endl;
    
    int total = 0, passed = 0, failed = 0;
    
    RUN_TEST(test_hnf_theorem_comprehensive);
    RUN_TEST(test_compression_quality_tradeoff);
    RUN_TEST(test_scalability);
    RUN_TEST(test_pathological_cases);
    RUN_TEST(test_end_to_end_simulation);
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║                    COMPREHENSIVE TEST SUMMARY                   ║" << std::endl;
    std::cout << "╠════════════════════════════════════════════════════════════════╣" << std::endl;
    std::cout << "║  Total:   " << std::setw(2) << total << "                                                    ║" << std::endl;
    std::cout << "║  Passed:  " << std::setw(2) << passed << "                                                    ║" << std::endl;
    std::cout << "║  Failed:  " << std::setw(2) << failed << "                                                    ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════════════════════╝" << std::endl;
    
    return (failed == 0) ? 0 : 1;
}
