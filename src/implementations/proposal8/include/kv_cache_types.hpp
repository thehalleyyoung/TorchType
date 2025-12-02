#pragma once

#include <torch/torch.h>
#include <vector>
#include <map>
#include <memory>
#include <cmath>

namespace hnf {
namespace kv_cache {

// Hardware precision levels
enum class PrecisionLevel {
    FP32 = 32,
    FP16 = 16,
    INT8 = 8,
    INT4 = 4
};

// Convert precision level to string
inline std::string precision_to_string(PrecisionLevel p) {
    switch(p) {
        case PrecisionLevel::FP32: return "FP32";
        case PrecisionLevel::FP16: return "FP16";
        case PrecisionLevel::INT8: return "INT8";
        case PrecisionLevel::INT4: return "INT4";
        default: return "UNKNOWN";
    }
}

// Bits per element for each precision
inline int bits_per_element(PrecisionLevel p) {
    return static_cast<int>(p);
}

// Position-wise curvature information
struct PositionCurvature {
    int64_t position;
    int64_t layer_idx;
    double curvature_score;  // Higher = more important
    double attention_weight;  // Average attention to this position
    double gradient_norm;     // Norm of gradient w.r.t. this position
    double hessian_trace;     // Trace of Hessian (curvature measure)
    
    // Computed precision requirement based on HNF Theorem 5.7
    // p >= log2(kappa * D^2 / epsilon)
    PrecisionLevel required_precision;
    
    PositionCurvature() 
        : position(0), layer_idx(0), curvature_score(0.0), 
          attention_weight(0.0), gradient_norm(0.0), hessian_trace(0.0),
          required_precision(PrecisionLevel::FP16) {}
};

// Layer-wise precision map
struct LayerPrecisionMap {
    int64_t layer_idx;
    int64_t num_positions;
    std::vector<PrecisionLevel> position_precisions;
    
    // Statistics
    double avg_curvature;
    double max_curvature;
    double memory_bytes_fp16;  // Baseline FP16 memory
    double memory_bytes_adaptive;  // Adaptive precision memory
    
    LayerPrecisionMap() 
        : layer_idx(0), num_positions(0), avg_curvature(0.0), 
          max_curvature(0.0), memory_bytes_fp16(0.0), 
          memory_bytes_adaptive(0.0) {}
    
    double compression_ratio() const {
        if (memory_bytes_adaptive == 0.0) return 1.0;
        return memory_bytes_fp16 / memory_bytes_adaptive;
    }
};

// Complete precision analysis result
struct PrecisionAnalysisResult {
    int64_t num_layers;
    int64_t max_seq_length;
    int64_t hidden_dim;
    
    std::vector<LayerPrecisionMap> layer_maps;
    std::map<int64_t, std::vector<PositionCurvature>> layer_curvatures;
    
    // Global statistics
    double total_memory_fp16_gb;
    double total_memory_adaptive_gb;
    double overall_compression_ratio;
    double quality_preserved;  // Estimated quality preservation [0, 1]
    
    // Recommendations
    std::vector<std::string> recommendations;
    
    PrecisionAnalysisResult() 
        : num_layers(0), max_seq_length(0), hidden_dim(0),
          total_memory_fp16_gb(0.0), total_memory_adaptive_gb(0.0),
          overall_compression_ratio(1.0), quality_preserved(1.0) {}
};

// Quantization parameters for each precision level
struct QuantizationParams {
    PrecisionLevel precision;
    torch::Tensor scale;  // Per-channel or per-tensor scale
    torch::Tensor zero_point;  // For asymmetric quantization
    double clip_min;
    double clip_max;
    
    QuantizationParams() : precision(PrecisionLevel::FP16), 
                          clip_min(-1e6), clip_max(1e6) {}
};

// Mixed-precision buffer entry
struct MixedPrecisionEntry {
    PrecisionLevel precision;
    torch::Tensor data;  // Quantized data
    QuantizationParams params;  // Dequantization parameters
    
    MixedPrecisionEntry() : precision(PrecisionLevel::FP16) {}
};

// Attention pattern statistics
struct AttentionPattern {
    int64_t layer_idx;
    torch::Tensor attention_weights;  // [num_heads, seq_len, seq_len]
    torch::Tensor position_importance;  // [seq_len] - averaged importance
    
    // Locality patterns
    double recency_bias;  // Correlation with distance
    double positional_anchor_strength;  // Importance of first tokens
    double semantic_clustering;  // Clustering coefficient
    
    AttentionPattern() 
        : layer_idx(0), recency_bias(0.0), 
          positional_anchor_strength(0.0), semantic_clustering(0.0) {}
};

// Calibration sample for analysis
struct CalibrationSample {
    std::vector<torch::Tensor> attention_patterns;  // [batch, heads, seq, seq]
    std::vector<torch::Tensor> keys;    // [batch, seq, heads, head_dim]
    std::vector<torch::Tensor> values;  // [batch, seq, heads, head_dim]
    std::vector<torch::Tensor> queries; // [batch, seq, heads, head_dim]
    
    CalibrationSample() {}
};

// Configuration for KV cache precision analysis
struct KVCacheConfig {
    // Analysis parameters
    int64_t num_layers;
    int64_t num_heads;
    int64_t head_dim;
    int64_t max_seq_length;
    
    // Quality threshold (0 to 1)
    double quality_threshold;
    
    // Memory budget (in GB, 0 = no constraint)
    double memory_budget_gb;
    
    // Curvature computation method
    enum class CurvatureMethod {
        ATTENTION_BASED,  // Based on attention weights
        GRADIENT_BASED,   // Based on gradients
        HESSIAN_BASED,    // Based on Hessian diagonal
        HYBRID            // Combination of all
    };
    CurvatureMethod curvature_method;
    
    // Target accuracy epsilon for HNF bounds
    double target_epsilon;
    
    // Safety margin for precision (add extra bits)
    int safety_margin_bits;
    
    KVCacheConfig() 
        : num_layers(12), num_heads(12), head_dim(64), 
          max_seq_length(2048), quality_threshold(0.99),
          memory_budget_gb(0.0), curvature_method(CurvatureMethod::HYBRID),
          target_epsilon(1e-3), safety_margin_bits(2) {}
};

} // namespace kv_cache
} // namespace hnf
