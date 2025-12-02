#pragma once

#include "attention_types.hpp"
#include <torch/torch.h>
#include <cmath>

namespace hnf {
namespace attention {

/**
 * Curvature analysis for attention mechanisms based on HNF theory.
 * 
 * From the paper (Section on Attention, Example 4):
 * For attention A = softmax(QK^T / sqrt(d)), the curvature is:
 * κ_attn^curv = (1/2) * ||D^2 softmax|| 
 *             = O(||Q|| * ||K|| / d * exp(2 * ||QK^T||_∞ / sqrt(d)))
 * 
 * This predicts:
 * - Overflow when ||QK^T||_∞ is large
 * - High precision requirements when curvature is high
 * - Gradient vanishing when attention is too peaked
 */
class AttentionCurvature {
public:
    /**
     * Compute the curvature bound for attention mechanism.
     * 
     * Based on Theorem 4.1 (Precision Obstruction Theorem):
     * p >= log2(c * κ * D^2 / ε)
     * where κ is curvature, D is domain diameter, ε is target accuracy.
     */
    static torch::Tensor compute_curvature(
        const torch::Tensor& Q,  // [batch, heads, seq, head_dim]
        const torch::Tensor& K,  // [batch, heads, seq, head_dim]
        double temperature = 1.0
    );
    
    /**
     * Compute the Hessian-based curvature estimate.
     * 
     * For softmax(x), the Hessian is:
     * H_ij = diag(s) - s*s^T
     * where s = softmax(x).
     * 
     * The curvature invariant is κ = (1/2) * sup_x ||H_x||_op
     */
    static torch::Tensor compute_softmax_curvature(
        const torch::Tensor& logits  // [batch, heads, seq, seq]
    );
    
    /**
     * Estimate required precision bits from curvature.
     * 
     * From HNF Theorem (Precision Lower Bound):
     * p_min = log2(c * κ * D^2 / ε)
     */
    static torch::Tensor estimate_precision_requirement(
        const torch::Tensor& curvature,
        double diameter,
        double target_accuracy
    );
    
    /**
     * Compute Lipschitz constant of attention operation.
     * 
     * L_attn = max(||∂A/∂Q||, ||∂A/∂K||)
     */
    static torch::Tensor compute_lipschitz_constant(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        double temperature = 1.0
    );
    
    /**
     * Compute the error functional Φ for attention.
     * 
     * From HNF Stability Composition Theorem:
     * Φ_attn(ε, H) = L_attn * ε + Δ_attn(H)
     * where Δ_attn is hardware-dependent roundoff error.
     */
    static torch::Tensor compute_error_functional(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        double input_error,
        const HardwareModel& hardware,
        double temperature = 1.0
    );
    
    /**
     * Analyze gradient flow through attention.
     * 
     * Checks for gradient vanishing/explosion based on:
     * - Attention spikiness (near one-hot → sparse gradients)
     * - Curvature (high curvature → unstable gradients)
     */
    static torch::Tensor analyze_gradient_flow(
        const torch::Tensor& attention_weights,  // [batch, heads, seq, seq]
        const torch::Tensor& values              // [batch, heads, seq, head_dim]
    );
    
    /**
     * Compute the condition number of the attention operation.
     * 
     * κ(attn) = ||attn|| * ||attn^{-1}||
     * Approximated using operator norm bounds.
     */
    static torch::Tensor compute_condition_number(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& V,
        double temperature = 1.0
    );
    
    /**
     * Estimate domain diameter for attention.
     * 
     * D = diameter of input space, used in precision bounds.
     */
    static double estimate_domain_diameter(
        const torch::Tensor& Q,
        const torch::Tensor& K
    );
    
    /**
     * Compute second-order curvature for compositional analysis.
     * 
     * For composed operations f ∘ g:
     * κ_{f∘g} = κ_f * L_g^2 + L_f * κ_g
     */
    static torch::Tensor compose_curvature(
        double curvature_f,
        double lipschitz_f,
        double curvature_g,
        double lipschitz_g
    );

private:
    // Helper: compute spectral norm (largest singular value)
    static torch::Tensor spectral_norm(const torch::Tensor& matrix);
    
    // Helper: compute Frobenius norm
    static torch::Tensor frobenius_norm(const torch::Tensor& matrix);
    
    // Helper: estimate Hessian norm via finite differences
    static torch::Tensor estimate_hessian_norm(
        const torch::Tensor& logits,
        double epsilon = 1e-5
    );
};

} // namespace attention
} // namespace hnf
