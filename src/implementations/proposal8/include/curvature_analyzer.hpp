#pragma once

#include "kv_cache_types.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace hnf {
namespace kv_cache {

/**
 * CurvatureAnalyzer: Computes position-wise curvature scores for KV cache
 * 
 * Based on HNF Theorem 5.7 (Precision Obstruction Theorem):
 * For a morphism f with curvature κ_f on domain of diameter D,
 * achieving ε-accuracy requires p >= log_2(c * κ_f * D^2 / ε) bits
 * 
 * For KV cache, the curvature at position t is:
 * κ_t^{KV} = α_t * ||∂output/∂K_t|| * ||∂²output/∂K_t²||
 * 
 * where α_t is the attention weight to position t.
 */
class CurvatureAnalyzer {
public:
    explicit CurvatureAnalyzer(const KVCacheConfig& config);
    
    /**
     * Compute position-wise curvature from attention patterns
     * 
     * @param attention_weights: [batch, num_heads, seq_len, seq_len]
     * @param keys: [batch, seq_len, num_heads, head_dim]
     * @param values: [batch, seq_len, num_heads, head_dim]
     * @param queries: [batch, seq_len, num_heads, head_dim]
     * @param layer_idx: Which layer this is
     * @return: Vector of PositionCurvature for each position
     */
    std::vector<PositionCurvature> compute_position_curvatures(
        const torch::Tensor& attention_weights,
        const torch::Tensor& keys,
        const torch::Tensor& values,
        const torch::Tensor& queries,
        int64_t layer_idx
    );
    
    /**
     * Compute curvature based on gradients
     * Requires model outputs and targets
     */
    std::vector<PositionCurvature> compute_gradient_based_curvature(
        const torch::Tensor& keys,
        const torch::Tensor& values,
        const torch::Tensor& output,
        const torch::Tensor& target,
        int64_t layer_idx
    );
    
    /**
     * Compute Hessian-based curvature (second-order information)
     * This is the most accurate but most expensive
     */
    std::vector<PositionCurvature> compute_hessian_based_curvature(
        const torch::Tensor& keys,
        const torch::Tensor& values,
        const torch::Tensor& output,
        int64_t layer_idx
    );
    
    /**
     * Hybrid method: combines attention, gradient, and Hessian info
     */
    std::vector<PositionCurvature> compute_hybrid_curvature(
        const torch::Tensor& attention_weights,
        const torch::Tensor& keys,
        const torch::Tensor& values,
        const torch::Tensor& queries,
        const torch::Tensor& output,
        int64_t layer_idx
    );
    
    /**
     * Analyze attention patterns to identify locality structure
     */
    AttentionPattern analyze_attention_pattern(
        const torch::Tensor& attention_weights,
        int64_t layer_idx
    );
    
private:
    KVCacheConfig config_;
    
    // Compute attention-based importance scores
    torch::Tensor compute_attention_importance(
        const torch::Tensor& attention_weights
    );
    
    // Compute gradient norm w.r.t. each position
    torch::Tensor compute_gradient_norms(
        const torch::Tensor& keys,
        const torch::Tensor& values,
        const torch::Tensor& output
    );
    
    // Compute Hessian trace approximation (Hutchinson's estimator)
    torch::Tensor compute_hessian_trace_approx(
        const torch::Tensor& keys,
        const torch::Tensor& values,
        const torch::Tensor& output,
        int num_samples = 10
    );
    
    // Detect recency bias in attention
    double compute_recency_bias(const torch::Tensor& attention_weights);
    
    // Detect positional anchor strength (importance of first tokens)
    double compute_positional_anchor_strength(const torch::Tensor& attention_weights);
    
    // Compute semantic clustering coefficient
    double compute_semantic_clustering(const torch::Tensor& attention_weights);
};

} // namespace kv_cache
} // namespace hnf
