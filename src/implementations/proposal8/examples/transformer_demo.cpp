#include "kv_cache_analyzer.hpp"
#include <torch/torch.h>
#include <iostream>
#include <vector>
#include <memory>

using namespace hnf::kv_cache;

// Simple transformer attention layer for demonstration
class TransformerAttention : public torch::nn::Module {
public:
    TransformerAttention(int64_t d_model, int64_t num_heads)
        : d_model_(d_model), num_heads_(num_heads) {
        
        head_dim_ = d_model / num_heads;
        
        q_proj = register_module("q_proj", torch::nn::Linear(d_model, d_model));
        k_proj = register_module("k_proj", torch::nn::Linear(d_model, d_model));
        v_proj = register_module("v_proj", torch::nn::Linear(d_model, d_model));
        out_proj = register_module("out_proj", torch::nn::Linear(d_model, d_model));
    }
    
    std::pair<torch::Tensor, torch::Tensor> forward(const torch::Tensor& x) {
        auto batch_size = x.size(0);
        auto seq_len = x.size(1);
        
        // Project to Q, K, V
        auto q = q_proj->forward(x);
        auto k = k_proj->forward(x);
        auto v = v_proj->forward(x);
        
        // Reshape to [batch, num_heads, seq_len, head_dim]
        q = q.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
        k = k.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
        v = v.view({batch_size, seq_len, num_heads_, head_dim_}).transpose(1, 2);
        
        // Scaled dot-product attention
        auto scores = torch::matmul(q, k.transpose(-2, -1)) / std::sqrt(static_cast<double>(head_dim_));
        
        // Apply causal mask
        auto mask = torch::triu(torch::ones({seq_len, seq_len}), 1).to(x.device());
        scores = scores.masked_fill(mask.to(torch::kBool), -1e9);
        
        auto attn_weights = torch::softmax(scores, -1);
        
        // Apply attention to values
        auto attn_output = torch::matmul(attn_weights, v);
        
        // Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view({batch_size, seq_len, d_model_});
        auto output = out_proj->forward(attn_output);
        
        return {output, attn_weights};
    }
    
private:
    int64_t d_model_;
    int64_t num_heads_;
    int64_t head_dim_;
    
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, out_proj{nullptr};
};

// Simple transformer model
class SimpleTransformer : public torch::nn::Module {
public:
    SimpleTransformer(int64_t d_model, int64_t num_heads, int64_t num_layers)
        : d_model_(d_model), num_heads_(num_heads), num_layers_(num_layers) {
        
        for (int64_t i = 0; i < num_layers; ++i) {
            auto attn = std::make_shared<TransformerAttention>(d_model, num_heads);
            layers_.push_back(register_module("attn_" + std::to_string(i), attn));
        }
    }
    
    std::pair<torch::Tensor, std::vector<torch::Tensor>> forward(const torch::Tensor& x) {
        auto hidden = x;
        std::vector<torch::Tensor> all_attn_weights;
        
        for (auto& layer : layers_) {
            auto [output, attn_weights] = layer->forward(hidden);
            hidden = output;
            all_attn_weights.push_back(attn_weights);
        }
        
        return {hidden, all_attn_weights};
    }
    
private:
    int64_t d_model_;
    int64_t num_heads_;
    int64_t num_layers_;
    std::vector<std::shared_ptr<TransformerAttention>> layers_;
};

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║      KV-Cache Precision Analyzer - Transformer Demo            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    // Model configuration
    int64_t d_model = 512;
    int64_t num_heads = 8;
    int64_t num_layers = 6;
    int64_t head_dim = d_model / num_heads;
    
    std::cout << "Building transformer model:\n";
    std::cout << "  Model dimension: " << d_model << "\n";
    std::cout << "  Number of heads: " << num_heads << "\n";
    std::cout << "  Number of layers: " << num_layers << "\n";
    std::cout << "  Head dimension: " << head_dim << "\n\n";
    
    // Create model
    auto model = std::make_shared<SimpleTransformer>(d_model, num_heads, num_layers);
    model->eval();
    
    // Configure KV cache analyzer
    KVCacheConfig config;
    config.num_layers = num_layers;
    config.num_heads = num_heads;
    config.head_dim = head_dim;
    config.max_seq_length = 1024;
    config.quality_threshold = 0.99;
    config.target_epsilon = 1e-3;
    config.curvature_method = KVCacheConfig::CurvatureMethod::HYBRID;
    
    KVCacheAnalyzer analyzer(config);
    
    // Generate calibration data (various sequence lengths)
    std::cout << "Generating calibration data...\n";
    std::vector<torch::Tensor> calibration_data;
    std::vector<int64_t> seq_lengths = {64, 128, 256, 512};
    
    for (auto seq_len : seq_lengths) {
        auto input = torch::randn({1, seq_len, d_model});
        calibration_data.push_back(input);
        std::cout << "  Generated sequence of length " << seq_len << "\n";
    }
    
    // Forward function that uses our model
    auto forward_fn = [&](const torch::Tensor& input) -> std::pair<torch::Tensor, std::vector<torch::Tensor>> {
        torch::NoGradGuard no_grad;
        return model->forward(input);
    };
    
    // Run analysis
    std::cout << "\nAnalyzing KV cache precision requirements...\n";
    auto result = analyzer.analyze(calibration_data, forward_fn);
    
    // Print report
    std::cout << "\n";
    analyzer.print_analysis_report(result);
    
    // Detailed layer analysis
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                   DETAILED LAYER ANALYSIS                       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    for (size_t layer_idx = 0; layer_idx < result.layer_maps.size(); ++layer_idx) {
        const auto& map = result.layer_maps[layer_idx];
        
        std::cout << "Layer " << layer_idx << ":\n";
        std::cout << "  Average curvature: " << map.avg_curvature << "\n";
        std::cout << "  Maximum curvature: " << map.max_curvature << "\n";
        std::cout << "  Compression ratio: " << map.compression_ratio() << "x\n";
        
        // Analyze position-wise precision pattern
        std::cout << "  Precision pattern (first 64 positions):\n    ";
        for (size_t pos = 0; pos < std::min(size_t(64), map.position_precisions.size()); ++pos) {
            auto prec = map.position_precisions[pos];
            char symbol;
            switch(prec) {
                case PrecisionLevel::FP32: symbol = '#'; break;
                case PrecisionLevel::FP16: symbol = '+'; break;
                case PrecisionLevel::INT8: symbol = '.'; break;
                case PrecisionLevel::INT4: symbol = ' '; break;
                default: symbol = '?';
            }
            std::cout << symbol;
            if ((pos + 1) % 32 == 0) std::cout << "\n    ";
        }
        std::cout << "\n  Legend: #=FP32  +=FP16  .=INT8  (space)=INT4\n\n";
    }
    
    // Create and test adaptive cache
    std::cout << "Creating adaptive KV cache...\n";
    auto cache = analyzer.create_adaptive_cache(result);
    
    // Simulate inference with the cache
    std::cout << "\nSimulating inference with adaptive cache:\n";
    
    int64_t test_seq_len = 256;
    auto test_input = torch::randn({1, test_seq_len, d_model});
    
    // Run forward pass and cache KV pairs
    auto [output, attention_weights] = model->forward(test_input);
    
    std::cout << "  Processed sequence of length " << test_seq_len << "\n";
    
    // In practice, we would extract K,V from the model
    // For demo, we'll populate the cache with synthetic data
    for (int64_t layer = 0; layer < num_layers; ++layer) {
        for (int64_t pos = 0; pos < test_seq_len; ++pos) {
            auto key = torch::randn({num_heads * head_dim});
            auto value = torch::randn({num_heads * head_dim});
            cache->update(layer, pos, key, value);
        }
    }
    
    std::cout << "  Cached " << (num_layers * test_seq_len) << " KV pairs\n";
    std::cout << "  Total memory: " << cache->total_memory_usage_gb() << " GB\n";
    std::cout << "  Compression: " << cache->compression_ratio() << "x\n";
    
    // Calculate memory savings
    double uniform_fp16_gb = static_cast<double>(num_layers * test_seq_len * num_heads * head_dim * 2 * 2) / (1024.0 * 1024.0 * 1024.0);
    double saved_gb = uniform_fp16_gb - cache->total_memory_usage_gb();
    
    std::cout << "\nMemory comparison:\n";
    std::cout << "  Uniform FP16:       " << uniform_fp16_gb << " GB\n";
    std::cout << "  Adaptive precision: " << cache->total_memory_usage_gb() << " GB\n";
    std::cout << "  Saved:              " << saved_gb << " GB (" 
              << (saved_gb / uniform_fp16_gb * 100.0) << "%)\n";
    
    // Demonstrate dynamic precision adjustment
    std::cout << "\nDemonstrating dynamic precision adjustment:\n";
    DynamicPrecisionAdjuster adjuster(config, cache);
    adjuster.set_upgrade_threshold(0.1);
    adjuster.set_downgrade_threshold(0.01);
    
    // Simulate attention pattern changes
    for (int64_t layer = 0; layer < std::min(int64_t(3), num_layers); ++layer) {
        adjuster.update_importance(layer, attention_weights[layer]);
        std::cout << "  Updated importance for layer " << layer << "\n";
    }
    
    std::cout << "\n✓ Transformer demo complete!\n";
    std::cout << "\nKey insights:\n";
    for (const auto& rec : result.recommendations) {
        std::cout << "  • " << rec << "\n";
    }
    
    return 0;
}
