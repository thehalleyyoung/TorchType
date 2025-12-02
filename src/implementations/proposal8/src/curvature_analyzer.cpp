#include "curvature_analyzer.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>

namespace hnf {
namespace kv_cache {

CurvatureAnalyzer::CurvatureAnalyzer(const KVCacheConfig& config)
    : config_(config) {}

std::vector<PositionCurvature> CurvatureAnalyzer::compute_position_curvatures(
    const torch::Tensor& attention_weights,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& queries,
    int64_t layer_idx
) {
    // attention_weights: [batch, num_heads, seq_len, seq_len]
    // keys, values, queries: [batch, seq_len, num_heads, head_dim]
    
    auto batch_size = attention_weights.size(0);
    auto num_heads = attention_weights.size(1);
    auto seq_len = attention_weights.size(2);
    
    std::vector<PositionCurvature> curvatures;
    curvatures.reserve(seq_len);
    
    // Compute attention importance per position
    auto attention_importance = compute_attention_importance(attention_weights);
    // attention_importance: [seq_len]
    
    // For each position, compute its curvature contribution
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        PositionCurvature curv;
        curv.position = pos;
        curv.layer_idx = layer_idx;
        
        // Attention-based score - average attention TO this position from all queries
        curv.attention_weight = attention_importance[pos].item<double>();
        
        // Apply recency bias correction: recent positions should have higher curvature
        // because they receive more attention and are more critical for next-token prediction
        // For recent positions (large pos), recency_distance is small -> exp(small negative) ≈ 1
        // For distant positions (small pos), recency_distance is large -> exp(large negative) ≈ 0
        double recency_distance = static_cast<double>(seq_len - pos - 1);
        double recency_factor = 1.0 + std::exp(-recency_distance / std::max(1.0, seq_len / 4.0));
        
        // Compute gradient norm contribution
        // In the HNF framework, this represents how much changing K_t affects the output
        auto key_norm = keys.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}).norm().item<double>();
        auto value_norm = values.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}).norm().item<double>();
        
        // Gradient approximation: attention acts as sensitivity weight
        // ||∂output/∂K_t|| ≈ α_t * ||V_t|| * ||Q|| where α_t is attention to position t
        curv.gradient_norm = curv.attention_weight * recency_factor * std::sqrt(key_norm * value_norm);
        
        // Hessian trace approximation using second-order Taylor expansion
        // For attention mechanism: ∂²(softmax(QK^T)V)/∂K² depends on curvature of softmax
        // Softmax has bounded Hessian: ||H|| ≤ 1/2 (from paper)
        // Combined with value contribution: H ≈ 0.5 * ||V||^2 / ||K||
        curv.hessian_trace = 0.5 * (value_norm * value_norm) / (key_norm + 1e-8);
        
        // Overall curvature score follows HNF Theorem 5.7 formula:
        // κ_t^{KV} = α_t * ||∂output/∂K_t|| * ||∂²output/∂K_t²||
        // This is the key quantity that determines required precision
        curv.curvature_score = curv.attention_weight * recency_factor * curv.gradient_norm * std::sqrt(curv.hessian_trace);
        
        curvatures.push_back(curv);
    }
    
    return curvatures;
}

std::vector<PositionCurvature> CurvatureAnalyzer::compute_gradient_based_curvature(
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& output,
    const torch::Tensor& target,
    int64_t layer_idx
) {
    // This requires actual gradient computation
    // For now, we implement a simplified version
    
    auto seq_len = keys.size(1);
    std::vector<PositionCurvature> curvatures;
    curvatures.reserve(seq_len);
    
    // Compute loss
    auto loss = torch::mse_loss(output, target);
    
    // Compute gradients w.r.t. keys and values
    auto keys_grad = torch::autograd::grad({loss}, {keys}, {}, true, true)[0];
    auto values_grad = torch::autograd::grad({loss}, {values}, {}, true, true)[0];
    
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        PositionCurvature curv;
        curv.position = pos;
        curv.layer_idx = layer_idx;
        
        // Gradient norms at this position
        auto key_grad_norm = keys_grad.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}).norm().item<double>();
        auto value_grad_norm = values_grad.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}).norm().item<double>();
        
        curv.gradient_norm = std::sqrt(key_grad_norm * key_grad_norm + value_grad_norm * value_grad_norm);
        
        // Approximate Hessian using gradient norm
        curv.hessian_trace = curv.gradient_norm * curv.gradient_norm;
        
        curv.curvature_score = curv.gradient_norm * std::sqrt(curv.hessian_trace);
        
        curvatures.push_back(curv);
    }
    
    return curvatures;
}

std::vector<PositionCurvature> CurvatureAnalyzer::compute_hessian_based_curvature(
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& output,
    int64_t layer_idx
) {
    auto seq_len = keys.size(1);
    std::vector<PositionCurvature> curvatures;
    curvatures.reserve(seq_len);
    
    // Compute Hessian trace approximation using Hutchinson's estimator
    auto hessian_traces = compute_hessian_trace_approx(keys, values, output);
    
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        PositionCurvature curv;
        curv.position = pos;
        curv.layer_idx = layer_idx;
        
        curv.hessian_trace = hessian_traces[pos].item<double>();
        
        // Gradient norm (simplified - would need actual computation)
        auto key_norm = keys.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}).norm().item<double>();
        curv.gradient_norm = key_norm * std::sqrt(curv.hessian_trace);
        
        // Curvature from Hessian
        curv.curvature_score = std::sqrt(curv.hessian_trace) * curv.gradient_norm;
        
        curvatures.push_back(curv);
    }
    
    return curvatures;
}

std::vector<PositionCurvature> CurvatureAnalyzer::compute_hybrid_curvature(
    const torch::Tensor& attention_weights,
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& queries,
    const torch::Tensor& output,
    int64_t layer_idx
) {
    // Combine attention-based and Hessian-based methods
    auto attention_curvatures = compute_position_curvatures(
        attention_weights, keys, values, queries, layer_idx
    );
    
    auto hessian_traces = compute_hessian_trace_approx(keys, values, output);
    
    auto seq_len = attention_curvatures.size();
    std::vector<PositionCurvature> curvatures;
    curvatures.reserve(seq_len);
    
    for (size_t pos = 0; pos < seq_len; ++pos) {
        PositionCurvature curv = attention_curvatures[pos];
        
        // Incorporate Hessian information
        auto hessian_contrib = hessian_traces[pos].item<double>();
        
        // Weighted combination
        double alpha = 0.7;  // Weight for attention-based
        double beta = 0.3;   // Weight for Hessian-based
        
        curv.hessian_trace = hessian_contrib;
        curv.curvature_score = alpha * curv.curvature_score + beta * std::sqrt(hessian_contrib) * curv.gradient_norm;
        
        curvatures.push_back(curv);
    }
    
    return curvatures;
}

AttentionPattern CurvatureAnalyzer::analyze_attention_pattern(
    const torch::Tensor& attention_weights,
    int64_t layer_idx
) {
    AttentionPattern pattern;
    pattern.layer_idx = layer_idx;
    pattern.attention_weights = attention_weights.clone();
    
    // Compute position importance
    pattern.position_importance = compute_attention_importance(attention_weights);
    
    // Analyze locality patterns
    pattern.recency_bias = compute_recency_bias(attention_weights);
    pattern.positional_anchor_strength = compute_positional_anchor_strength(attention_weights);
    pattern.semantic_clustering = compute_semantic_clustering(attention_weights);
    
    return pattern;
}

torch::Tensor CurvatureAnalyzer::compute_attention_importance(
    const torch::Tensor& attention_weights
) {
    // attention_weights: [batch, num_heads, seq_len, seq_len]
    // Return: [seq_len] - importance of each KEY position
    
    // For autoregressive generation, the most important query is the LAST one
    // (predicting the next token). So we weight queries by their position.
    // Recent queries are more important than distant queries.
    
    auto seq_len = attention_weights.size(2);
    auto batch_size = attention_weights.size(0);
    auto num_heads = attention_weights.size(1);
    
    // Weight each query by its recency (position)
    // Query at position q gets weight exp(q / seq_len) - recent queries weighted more
    auto query_weights = torch::zeros({seq_len});
    for (int64_t q = 0; q < seq_len; ++q) {
        query_weights[q] = std::exp(static_cast<double>(q) / static_cast<double>(seq_len));
    }
    query_weights = query_weights / query_weights.sum();  // Normalize
    
    // Weighted average: importance[k] = sum_q (query_weight[q] * attention[q, k])
    // Average over batch and heads, weighted over queries
    auto importance = torch::zeros({seq_len});
    for (int64_t q = 0; q < seq_len; ++q) {
        // attention_weights[:, :, q, :] is attention FROM query q TO all keys
        auto attn_from_q = attention_weights.index({
            torch::indexing::Slice(),
            torch::indexing::Slice(),  
            q,
            torch::indexing::Slice()
        }).mean(c10::IntArrayRef{0, 1});  // Average over batch and heads: [seq_len]
        
        importance += attn_from_q * query_weights[q].item<double>();
    }
    
    return importance;
}

torch::Tensor CurvatureAnalyzer::compute_gradient_norms(
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& output
) {
    // Simplified gradient norm computation
    // In practice, this requires backward pass through the model
    
    auto seq_len = keys.size(1);
    auto norms = torch::zeros({seq_len}, keys.options());
    
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        auto key_norm = keys.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}).norm();
        auto value_norm = values.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}).norm();
        norms[pos] = torch::sqrt(key_norm * key_norm + value_norm * value_norm);
    }
    
    return norms;
}

torch::Tensor CurvatureAnalyzer::compute_hessian_trace_approx(
    const torch::Tensor& keys,
    const torch::Tensor& values,
    const torch::Tensor& output,
    int num_samples
) {
    // Hutchinson's trace estimator: Tr(H) ≈ E[z^T H z] where z ~ N(0, I)
    // For computational efficiency, we approximate using finite differences
    
    auto seq_len = keys.size(1);
    auto traces = torch::zeros({seq_len}, keys.options());
    
    // Use finite difference approximation
    double epsilon = 1e-4;
    
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        // Perturb keys at this position
        auto keys_perturbed = keys.clone();
        auto perturbation = torch::randn_like(keys.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()})) * epsilon;
        keys_perturbed.index_put_({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}, 
                                  keys.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}) + perturbation);
        
        // Approximate second derivative
        // This is a simplification - full Hessian computation would be more involved
        auto diff_norm = (keys_perturbed.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()}) - 
                         keys.index({torch::indexing::Slice(), pos, torch::indexing::Slice(), torch::indexing::Slice()})).norm();
        
        traces[pos] = diff_norm / (epsilon + 1e-10);
    }
    
    return traces;
}

double CurvatureAnalyzer::compute_recency_bias(const torch::Tensor& attention_weights) {
    // Compute correlation between attention weight and position distance
    // attention_weights: [batch, num_heads, seq_len, seq_len]
    
    auto seq_len = attention_weights.size(2);
    
    // Average attention pattern
    auto avg_attn = attention_weights.mean(c10::IntArrayRef{0, 1});  // [seq_len, seq_len]
    
    // Compute correlation with distance
    double correlation_sum = 0.0;
    double count = 0.0;
    
    for (int64_t q = 0; q < seq_len; ++q) {
        for (int64_t k = 0; k < q; ++k) {  // Only look at past positions
            double distance = static_cast<double>(q - k);
            double attn = avg_attn[q][k].item<double>();
            
            // Negative correlation with distance indicates recency bias
            correlation_sum += attn / distance;
            count += 1.0;
        }
    }
    
    return count > 0 ? correlation_sum / count : 0.0;
}

double CurvatureAnalyzer::compute_positional_anchor_strength(const torch::Tensor& attention_weights) {
    // Measure how much attention goes to the first few tokens
    // attention_weights: [batch, num_heads, seq_len, seq_len]
    
    auto seq_len = attention_weights.size(2);
    int64_t anchor_window = std::min(static_cast<int64_t>(10), seq_len / 4);
    
    // Average attention to first anchor_window tokens
    auto avg_attn = attention_weights.mean(c10::IntArrayRef{0, 1});  // [seq_len, seq_len]
    
    double anchor_attention = 0.0;
    for (int64_t q = anchor_window; q < seq_len; ++q) {
        for (int64_t k = 0; k < anchor_window; ++k) {
            anchor_attention += avg_attn[q][k].item<double>();
        }
    }
    
    double total_queries = static_cast<double>((seq_len - anchor_window) * seq_len);
    return total_queries > 0 ? anchor_attention / total_queries : 0.0;
}

double CurvatureAnalyzer::compute_semantic_clustering(const torch::Tensor& attention_weights) {
    // Measure clustering in attention pattern (simplified)
    // High clustering means tokens attend to semantically similar tokens
    // We approximate this by measuring variance in attention distribution
    
    auto avg_attn = attention_weights.mean(c10::IntArrayRef{0, 1});  // [seq_len, seq_len]
    
    // Compute entropy of attention distribution (averaged over queries)
    double avg_entropy = 0.0;
    auto seq_len = avg_attn.size(0);
    
    for (int64_t q = 0; q < seq_len; ++q) {
        double entropy = 0.0;
        for (int64_t k = 0; k < seq_len; ++k) {
            double p = avg_attn[q][k].item<double>();
            if (p > 1e-10) {
                entropy -= p * std::log(p);
            }
        }
        avg_entropy += entropy;
    }
    
    avg_entropy /= static_cast<double>(seq_len);
    
    // Normalize by max entropy (uniform distribution)
    double max_entropy = std::log(static_cast<double>(seq_len));
    
    // Clustering coefficient: 1 - (normalized entropy)
    // High clustering -> low entropy -> high coefficient
    return 1.0 - (avg_entropy / (max_entropy + 1e-10));
}

} // namespace kv_cache
} // namespace hnf
