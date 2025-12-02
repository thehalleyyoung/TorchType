#pragma once

#include "stability_linter.hpp"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>

namespace hnf {
namespace stability_linter {

// Real transformer architecture analysis
class TransformerAnalyzer {
public:
    struct AttentionConfig {
        int64_t d_model = 512;
        int64_t n_heads = 8;
        int64_t d_k = 64;
        int64_t d_v = 64;
        int64_t seq_len = 128;
        bool scaled = true;
        float dropout = 0.1f;
    };
    
    struct AnalysisResult {
        std::map<std::string, double> layer_curvatures;
        std::map<std::string, int> precision_requirements;  // bits
        std::map<std::string, std::string> recommendations;
        double total_composition_curvature;
        int min_safe_precision;  // minimum bits for entire transformer
        std::vector<LintResult> issues;
    };
    
    TransformerAnalyzer(const AttentionConfig& config = AttentionConfig());
    
    // Analyze multi-head attention mechanism
    AnalysisResult analyze_attention_block();
    
    // Analyze full transformer layer (attention + FFN)
    AnalysisResult analyze_transformer_layer(int ffn_hidden_dim = 2048);
    
    // Analyze stacked transformer (BERT/GPT style)
    AnalysisResult analyze_stacked_transformer(int num_layers);
    
    // Build computation graph for scaled dot-product attention
    std::shared_ptr<ComputationGraph> build_attention_graph();
    
    // Build computation graph for feed-forward network
    std::shared_ptr<ComputationGraph> build_ffn_graph(int hidden_dim);
    
    // Analyze impact of quantization on transformer
    struct QuantizationAnalysis {
        std::map<std::string, bool> can_use_int8;
        std::map<std::string, bool> can_use_fp16;
        std::map<std::string, bool> requires_fp32;
        double accuracy_degradation_estimate;
        std::string summary;
    };
    
    QuantizationAnalysis analyze_quantization_safety(double target_accuracy = 1e-3);
    
private:
    AttentionConfig config_;
    
    // Helper: compute attention scores graph
    void add_attention_score_nodes(ComputationGraph& graph, const std::string& prefix);
    
    // Helper: compute curvature for attention mechanism
    double compute_attention_curvature();
    
    // Helper: estimate Lipschitz constant for layer
    double estimate_layer_lipschitz(const std::string& layer_type);
};

// Analyze common transformer variants
class ModelVariantAnalyzer {
public:
    enum class ModelType {
        BERT,
        GPT2,
        LLAMA,
        VISION_TRANSFORMER
    };
    
    struct ModelSpec {
        ModelType type;
        int num_layers;
        int d_model;
        int n_heads;
        int ffn_hidden_dim;
        int vocab_size;
        int max_seq_len;
    };
    
    static ModelSpec get_bert_base();
    static ModelSpec get_gpt2_small();
    static ModelSpec get_llama2_7b();
    static ModelSpec get_vit_base();
    
    static TransformerAnalyzer::AnalysisResult analyze_model(const ModelSpec& spec);
    
    // Identify which layers are precision-critical
    static std::vector<std::string> identify_critical_layers(const ModelSpec& spec);
};

} // namespace stability_linter
} // namespace hnf
