#include "../include/attention_curvature.hpp"
#include <torch/torch.h>
#include <cmath>
#include <algorithm>

namespace hnf {
namespace attention {

// Helper: compute spectral norm (largest singular value)
torch::Tensor AttentionCurvature::spectral_norm(const torch::Tensor& matrix) {
    // For a matrix, spectral norm = largest singular value
    // We use SVD: ||A||_2 = σ_max(A)
    auto svd_result = torch::svd(matrix);
    return std::get<1>(svd_result).index({torch::indexing::Ellipsis, 0});
}

// Helper: compute Frobenius norm
torch::Tensor AttentionCurvature::frobenius_norm(const torch::Tensor& matrix) {
    return matrix.norm(2.0);  // Frobenius norm
}

// Compute the curvature bound for attention mechanism
torch::Tensor AttentionCurvature::compute_curvature(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    double temperature
) {
    // Q, K: [batch, heads, seq, head_dim]
    const auto batch_size = Q.size(0);
    const auto num_heads = Q.size(1);
    const auto seq_len = Q.size(2);
    const auto head_dim = Q.size(3);
    
    // Compute QK^T / sqrt(head_dim)
    auto QK = torch::matmul(Q, K.transpose(-2, -1));  // [batch, heads, seq, seq]
    QK = QK / (std::sqrt(static_cast<double>(head_dim)) * temperature);
    
    // Compute spectral norms of Q and K per head
    // Reshape to [batch * heads, seq, head_dim] for batched spectral norm
    auto Q_reshaped = Q.reshape({-1, seq_len, head_dim});
    auto K_reshaped = K.reshape({-1, seq_len, head_dim});
    
    auto Q_norms = torch::zeros({batch_size, num_heads}, Q.options());
    auto K_norms = torch::zeros({batch_size, num_heads}, K.options());
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            int idx = b * num_heads + h;
            auto Q_slice = Q_reshaped[idx];
            auto K_slice = K_reshaped[idx];
            
            // Compute spectral norm via SVD
            auto Q_svd = torch::svd(Q_slice);
            auto K_svd = torch::svd(K_slice);
            auto Q_sv = std::get<1>(Q_svd);  // singular values
            auto K_sv = std::get<1>(K_svd);
            
            Q_norms[b][h] = Q_sv[0];
            K_norms[b][h] = K_sv[0];
        }
    }
    
    // Maximum logit value per head
    auto QK_max = QK.abs().amax({-1, -2});  // [batch, heads]
    
    // HNF curvature formula (from paper Example 4):
    // κ_attn = O(||Q|| * ||K|| / d * exp(2 * ||QK^T||_∞ / sqrt(d)))
    // We use the explicit formula:
    // κ = (1/2) * ||Q|| * ||K|| / sqrt(d) * exp(2 * max(QK))
    
    auto curvature = 0.5 * Q_norms * K_norms / std::sqrt(head_dim) 
                   * torch::exp(2.0 * QK_max);
    
    return curvature;  // [batch, heads]
}

// Compute Hessian-based curvature estimate for softmax
torch::Tensor AttentionCurvature::compute_softmax_curvature(
    const torch::Tensor& logits
) {
    // logits: [batch, heads, seq, seq]
    // For softmax(x), Hessian H_ij = δ_ij * s_i - s_i * s_j
    // where s = softmax(x), δ_ij is Kronecker delta
    // ||H||_op = ||diag(s) - s*s^T||_op
    
    auto attention = torch::softmax(logits, /*dim=*/-1);
    const auto batch_size = attention.size(0);
    const auto num_heads = attention.size(1);
    const auto seq_len = attention.size(2);
    
    auto curvature = torch::zeros({batch_size, num_heads}, attention.options());
    
    // For each batch and head, compute Hessian operator norm
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            auto attn_slice = attention[b][h];  // [seq, seq]
            
            // For each query position
            double max_norm = 0.0;
            for (int i = 0; i < seq_len; ++i) {
                auto s = attn_slice[i];  // [seq] - softmax distribution
                
                // Construct Hessian: H = diag(s) - s*s^T
                auto diag_s = torch::diag(s);
                auto outer = torch::outer(s, s);
                auto H = diag_s - outer;
                
                // Compute operator norm (spectral norm)
                auto H_svd = torch::svd(H);
                auto H_sv = std::get<1>(H_svd);
                double norm = H_sv[0].item<double>();
                max_norm = std::max(max_norm, norm);
            }
            
            curvature[b][h] = 0.5 * max_norm;
        }
    }
    
    return curvature;
}

// Estimate required precision bits from curvature
torch::Tensor AttentionCurvature::estimate_precision_requirement(
    const torch::Tensor& curvature,
    double diameter,
    double target_accuracy
) {
    // HNF Precision Obstruction Theorem (Theorem 4.1):
    // p_min = log2(c * κ * D^2 / ε)
    // where c is an explicit constant (we use c ≈ 1 for simplicity)
    
    const double c = 1.0;
    auto precision_bits = torch::log2(
        c * curvature * diameter * diameter / target_accuracy
    );
    
    return precision_bits;
}

// Compute Lipschitz constant of attention operation
torch::Tensor AttentionCurvature::compute_lipschitz_constant(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    double temperature
) {
    // The Lipschitz constant of softmax is 1 (contractive)
    // The Lipschitz constant of matmul(Q, K^T) is ||Q|| * ||K||
    // Combined: L_attn ≈ ||Q|| * ||K|| / (sqrt(d) * temp)
    
    const auto batch_size = Q.size(0);
    const auto num_heads = Q.size(1);
    const auto seq_len = Q.size(2);
    const auto head_dim = Q.size(3);
    
    auto Q_reshaped = Q.reshape({-1, seq_len, head_dim});
    auto K_reshaped = K.reshape({-1, seq_len, head_dim});
    
    auto lipschitz = torch::zeros({batch_size, num_heads}, Q.options());
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            int idx = b * num_heads + h;
            auto Q_slice = Q_reshaped[idx];
            auto K_slice = K_reshaped[idx];
            
            auto Q_svd = torch::svd(Q_slice);
            auto K_svd = torch::svd(K_slice);
            auto Q_sv = std::get<1>(Q_svd);
            auto K_sv = std::get<1>(K_svd);
            
            double Q_norm = Q_sv[0].item<double>();
            double K_norm = K_sv[0].item<double>();
            
            // Softmax has Lipschitz constant 1
            // Total: L_QK * L_softmax * L_V
            // For QK^T: L = ||Q|| * ||K|| / sqrt(d)
            lipschitz[b][h] = Q_norm * K_norm / (std::sqrt(head_dim) * temperature);
        }
    }
    
    return lipschitz;
}

// Compute error functional
torch::Tensor AttentionCurvature::compute_error_functional(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    double input_error,
    const HardwareModel& hardware,
    double temperature
) {
    // HNF Stability Composition Theorem (Theorem 3.1):
    // Φ_f(ε, H) = L_f * ε + Δ_f(H)
    // where Δ_f(H) is hardware-dependent roundoff
    
    auto lipschitz = compute_lipschitz_constant(Q, K, temperature);
    
    // Hardware roundoff error
    double machine_eps = hardware.machine_epsilon();
    const auto seq_len = Q.size(2);
    const auto head_dim = Q.size(3);
    
    // Roundoff accumulates over operations:
    // QK^T: seq_len * head_dim multiplications + additions
    // softmax: 1 exp + seq_len additions + 1 division
    // attention: seq_len * head_dim multiplications
    double operations = 2.0 * seq_len * head_dim + seq_len + head_dim;
    double roundoff_term = operations * machine_eps;
    
    // Total error: input propagation + roundoff
    auto error_functional = lipschitz * input_error + roundoff_term;
    
    return error_functional;
}

// Analyze gradient flow through attention
torch::Tensor AttentionCurvature::analyze_gradient_flow(
    const torch::Tensor& attention_weights,
    const torch::Tensor& values
) {
    // Gradient norm estimate: ||∂L/∂Q|| ≈ ||attn|| * ||V||
    // If attention is too peaked (near one-hot), gradients become sparse
    
    const auto batch_size = attention_weights.size(0);
    const auto num_heads = attention_weights.size(1);
    const auto seq_len = attention_weights.size(2);
    
    auto gradient_norms = torch::zeros({batch_size, num_heads}, attention_weights.options());
    
    // Compute gradient norm indicator
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            auto attn_slice = attention_weights[b][h];  // [seq, seq]
            auto V_slice = values[b][h];  // [seq, head_dim]
            
            // Gradient flow indicator: sum of attention weights * value norms
            auto attn_max_tuple = attn_slice.max(-1);
            auto attn_max = std::get<0>(attn_max_tuple);  // [seq]
            auto attn_entropy = -(attn_slice * torch::log(attn_slice + 1e-10)).sum(-1);  // [seq]
            
            // Low entropy → sparse gradients
            // High max → peaked attention → unstable gradients
            auto gradient_indicator = attn_entropy.mean() * (1.0 - attn_max.mean());
            
            gradient_norms[b][h] = gradient_indicator;
        }
    }
    
    return gradient_norms;
}

// Compute condition number
torch::Tensor AttentionCurvature::compute_condition_number(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    double temperature
) {
    // Condition number: κ = ||A|| * ||A^{-1}||
    // For attention: approximate as product of Lipschitz constants
    
    auto lipschitz_QK = compute_lipschitz_constant(Q, K, temperature);
    
    const auto batch_size = V.size(0);
    const auto num_heads = V.size(1);
    const auto seq_len = V.size(2);
    const auto head_dim = V.size(3);
    
    auto V_reshaped = V.reshape({-1, seq_len, head_dim});
    auto V_norms = torch::zeros({batch_size, num_heads}, V.options());
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            int idx = b * num_heads + h;
            auto V_slice = V_reshaped[idx];
            auto V_svd = torch::svd(V_slice);
            auto V_sv = std::get<1>(V_svd);
            V_norms[b][h] = V_sv[0];
        }
    }
    
    // Condition number ≈ L_QK * L_V
    auto condition_number = lipschitz_QK * V_norms;
    
    return condition_number;
}

// Estimate domain diameter
double AttentionCurvature::estimate_domain_diameter(
    const torch::Tensor& Q,
    const torch::Tensor& K
) {
    // Domain diameter: max distance between inputs
    // D ≈ sqrt(seq_len * head_dim) * (max norm)
    
    auto Q_flat = Q.flatten();
    auto K_flat = K.flatten();
    auto Q_norm = Q_flat.norm(2.0);
    auto K_norm = K_flat.norm(2.0);
    auto max_norm = torch::max(Q_norm, K_norm);
    
    const auto seq_len = Q.size(2);
    const auto head_dim = Q.size(3);
    
    double diameter = std::sqrt(seq_len * head_dim) * max_norm.item<double>();
    
    return diameter;
}

// Compose curvature for compositional analysis
torch::Tensor AttentionCurvature::compose_curvature(
    double curvature_f,
    double lipschitz_f,
    double curvature_g,
    double lipschitz_g
) {
    // HNF compositional curvature (from paper):
    // κ_{f∘g} = κ_f * L_g^2 + L_f * κ_g
    
    double composed = curvature_f * lipschitz_g * lipschitz_g + 
                     lipschitz_f * curvature_g;
    
    return torch::tensor(composed);
}

// Estimate Hessian norm via finite differences
torch::Tensor AttentionCurvature::estimate_hessian_norm(
    const torch::Tensor& logits,
    double epsilon
) {
    // Approximate Hessian using finite differences
    // H ≈ (∇f(x+εe) - ∇f(x)) / ε
    
    const auto batch_size = logits.size(0);
    const auto num_heads = logits.size(1);
    const auto seq_len = logits.size(2);
    
    auto hessian_norms = torch::zeros({batch_size, num_heads}, logits.options());
    
    for (int b = 0; b < batch_size; ++b) {
        for (int h = 0; h < num_heads; ++h) {
            auto logits_slice = logits[b][h].clone();  // [seq, seq]
            
            // Compute softmax and its gradient
            auto s0 = torch::softmax(logits_slice, -1);
            
            double max_hessian = 0.0;
            
            // Sample a few directions for Hessian estimation
            for (int trial = 0; trial < std::min(10, static_cast<int>(seq_len)); ++trial) {
                auto perturbation = torch::randn_like(logits_slice) * epsilon;
                auto s_perturbed = torch::softmax(logits_slice + perturbation, -1);
                
                auto diff = (s_perturbed - s0) / epsilon;
                double diff_norm = diff.flatten().norm(2.0).item<double>();
                max_hessian = std::max(max_hessian, diff_norm);
            }
            
            hessian_norms[b][h] = max_hessian;
        }
    }
    
    return hessian_norms;
}

} // namespace attention
} // namespace hnf
