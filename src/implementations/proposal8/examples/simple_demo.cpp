#include "kv_cache_analyzer.hpp"
#include <torch/torch.h>
#include <iostream>

using namespace hnf::kv_cache;

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║         KV-Cache Precision Analyzer - Simple Demo              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Configure analyzer
    KVCacheConfig config;
    config.num_layers = 6;
    config.num_heads = 8;
    config.head_dim = 64;
    config.max_seq_length = 512;
    config.quality_threshold = 0.99;
    config.target_epsilon = 1e-3;
    config.curvature_method = KVCacheConfig::CurvatureMethod::ATTENTION_BASED;
    
    std::cout << "Configuration:\n";
    std::cout << "  Layers: " << config.num_layers << "\n";
    std::cout << "  Heads: " << config.num_heads << "\n";
    std::cout << "  Head dimension: " << config.head_dim << "\n";
    std::cout << "  Max sequence length: " << config.max_seq_length << "\n";
    std::cout << "  Quality threshold: " << config.quality_threshold << "\n\n";
    
    // Create analyzer
    KVCacheAnalyzer analyzer(config);
    
    // Create synthetic calibration data
    std::cout << "Generating calibration data...\n";
    std::vector<torch::Tensor> calibration_data;
    for (int i = 0; i < 5; ++i) {
        calibration_data.push_back(torch::randn({1, 128}));
    }
    
    // Mock forward function with realistic attention patterns
    auto forward_fn = [&](const torch::Tensor& input) -> std::pair<torch::Tensor, std::vector<torch::Tensor>> {
        auto seq_len = input.size(1);
        auto output = torch::randn({1, seq_len, config.num_heads * config.head_dim});
        
        std::vector<torch::Tensor> attention_weights;
        
        for (int64_t layer = 0; layer < config.num_layers; ++layer) {
            auto attn = torch::zeros({1, config.num_heads, seq_len, seq_len});
            
            // Create realistic attention pattern
            // - Recency bias
            // - Positional anchors (BOS token)
            // - Some long-range dependencies
            
            for (int64_t h = 0; h < config.num_heads; ++h) {
                for (int64_t q = 0; q < seq_len; ++q) {
                    // Local attention (recency bias)
                    for (int64_t k = 0; k <= q; ++k) {
                        double distance = static_cast<double>(q - k + 1);
                        double local_weight = 1.0 / (distance * distance);
                        attn[0][h][q][k] = local_weight;
                    }
                    
                    // Boost first token (positional anchor)
                    if (q > 0) {
                        attn[0][h][q][0] = attn[0][h][q][0] * 3.0;
                    }
                    
                    // Add some long-range connections
                    if (q > 20 && q % 10 == 0) {
                        for (int64_t k = 0; k < q - 20; k += 10) {
                            attn[0][h][q][k] = attn[0][h][q][k] * 2.0;
                        }
                    }
                    
                    // Normalize
                    auto row_sum = attn[0][h][q].sum().item<double>();
                    if (row_sum > 0) {
                        attn[0][h][q] /= row_sum;
                    }
                }
            }
            
            attention_weights.push_back(attn);
        }
        
        return {output, attention_weights};
    };
    
    // Run analysis
    std::cout << "\nRunning analysis...\n";
    auto result = analyzer.analyze(calibration_data, forward_fn);
    
    // Print detailed report
    std::cout << "\n";
    analyzer.print_analysis_report(result);
    
    // Create adaptive cache
    std::cout << "\nCreating adaptive precision KV cache...\n";
    auto cache = analyzer.create_adaptive_cache(result);
    
    // Demonstrate usage
    std::cout << "\nDemonstrating cache usage:\n";
    
    // Simulate adding entries
    for (int64_t pos = 0; pos < 64; ++pos) {
        auto key = torch::randn({config.num_heads * config.head_dim});
        auto value = torch::randn({config.num_heads * config.head_dim});
        
        cache->update(0, pos, key, value);
    }
    
    std::cout << "  Added 64 KV pairs to layer 0\n";
    std::cout << "  Current memory usage: " << cache->total_memory_usage_gb() << " GB\n";
    std::cout << "  Compression ratio: " << cache->compression_ratio() << "x\n";
    
    // Read back some entries
    std::vector<int64_t> positions = {0, 10, 20, 30, 40};
    auto [keys, values] = cache->get(0, positions);
    std::cout << "  Retrieved " << keys.size(0) << " KV pairs\n";
    
    // Show per-layer precision distribution
    std::cout << "\nPer-layer precision distribution:\n";
    for (size_t i = 0; i < result.layer_maps.size(); ++i) {
        const auto& map = result.layer_maps[i];
        
        std::map<PrecisionLevel, int> counts;
        for (auto prec : map.position_precisions) {
            counts[prec]++;
        }
        
        std::cout << "  Layer " << map.layer_idx << ":\n";
        for (const auto& [prec, count] : counts) {
            double percentage = 100.0 * count / map.position_precisions.size();
            std::cout << "    " << precision_to_string(prec) << ": " 
                     << count << " positions (" << std::fixed << std::setprecision(1) 
                     << percentage << "%)\n";
        }
    }
    
    // Save analysis to file
    std::cout << "\nSaving analysis to file...\n";
    analyzer.save_analysis(result, "kv_cache_analysis.txt");
    std::cout << "  Saved to kv_cache_analysis.txt\n";
    
    std::cout << "\n✓ Demo complete!\n";
    
    return 0;
}
