#pragma once

#include "interval.hpp"
#include "input_domain.hpp"
#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <limits>

namespace hnf {
namespace certified {

// Curvature bounds for different layer types
// Based on HNF paper Theorem 5.7 and Section 4
class CurvatureBounds {
public:
    struct LayerCurvature {
        double curvature;           // κ^curv from paper
        double lipschitz_constant;  // L
        std::string layer_type;
        std::string description;
    };
    
    // Linear layer: f(x) = Wx + b
    // Curvature = 0 (linear), Lipschitz = ||W||_op
    static LayerCurvature linear_layer(
        const Eigen::MatrixXd& W,
        const Eigen::VectorXd& b) {
        
        // Compute operator norm (largest singular value)
        Eigen::JacobiSVD<Eigen::MatrixXd> svd(W);
        double operator_norm = svd.singularValues()(0);
        
        return LayerCurvature{
            0.0,              // Linear functions have zero curvature
            operator_norm,
            "Linear",
            "f(x) = Wx + b"
        };
    }
    
    // ReLU activation: f(x) = max(0, x)
    // Curvature = 0 (piecewise linear), Lipschitz = 1
    static LayerCurvature relu_activation() {
        return LayerCurvature{
            0.0,    // Piecewise linear
            1.0,    // Non-expansive
            "ReLU",
            "f(x) = max(0, x)"
        };
    }
    
    // Softmax: f(x) = exp(x_i) / sum(exp(x_j))
    // From paper: κ ≈ exp(2 * max_logit) for worst case
    // Conservative bound based on interval analysis
    static LayerCurvature softmax_activation(const Interval& input_range) {
        double max_val = input_range.upper();
        
        // Hessian bound for softmax (from paper and standard analysis)
        // ||D²softmax|| ≤ 1/2 in operator norm (Jacobian of gradient)
        // But composition amplifies: depends on scale of inputs
        double curvature = 0.5 * std::exp(2.0 * std::abs(max_val));
        
        // Lipschitz constant = 1 for softmax (non-expansive in infinity norm)
        double lipschitz = 1.0;
        
        return LayerCurvature{
            curvature,
            lipschitz,
            "Softmax",
            "Softmax with max input " + std::to_string(max_val)
        };
    }
    
    // Layer normalization: f(x) = (x - mean(x)) / sqrt(var(x) + eps)
    // From paper: κ ≈ 1/var_min²
    static LayerCurvature layer_norm(
        const Interval& variance_bounds,
        double eps = 1e-5) {
        
        double var_min = variance_bounds.lower();
        if (var_min < eps) {
            var_min = eps;  // Clamp to avoid division by tiny variance
        }
        
        // Curvature inversely proportional to variance squared
        double curvature = 1.0 / (var_min * var_min);
        
        // Lipschitz constant approximately 1/sqrt(var)
        double lipschitz = 1.0 / std::sqrt(var_min);
        
        return LayerCurvature{
            curvature,
            lipschitz,
            "LayerNorm",
            "With variance bound [" + std::to_string(variance_bounds.lower()) +
                ", " + std::to_string(variance_bounds.upper()) + "]"
        };
    }
    
    // GELU activation: f(x) = x * Phi(x) where Phi is Gaussian CDF
    // Smooth nonlinearity, moderate curvature
    static LayerCurvature gelu_activation(const Interval& input_range) {
        // GELU has bounded second derivative
        // |f''(x)| ≤ C for some constant C ≈ 0.4
        // Maximum curvature occurs near x = 0
        
        double max_second_deriv = 0.4;  // Conservative estimate
        double curvature = 0.5 * max_second_deriv;
        
        // GELU is Lipschitz with constant ≈ 1.1
        double lipschitz = 1.1;
        
        return LayerCurvature{
            curvature,
            lipschitz,
            "GELU",
            "GELU activation"
        };
    }
    
    // Embedding layer (lookup table): f(idx) = E[idx]
    // Curvature = 0 (discrete), but need to track condition
    static LayerCurvature embedding(
        int vocab_size,
        int embedding_dim,
        double embedding_norm_bound) {
        
        // Embedding is piecewise constant (zero curvature)
        // Lipschitz constant based on maximum embedding norm difference
        double lipschitz = 2.0 * embedding_norm_bound;
        
        return LayerCurvature{
            0.0,
            lipschitz,
            "Embedding",
            "Vocab size " + std::to_string(vocab_size) + 
                ", dim " + std::to_string(embedding_dim)
        };
    }
    
    // Attention mechanism: f(Q, K, V) = softmax(QK^T/√d) * V
    // Complex composition with high curvature from softmax
    static LayerCurvature attention_layer(
        const Eigen::MatrixXd& Q,  // Query projection weights
        const Eigen::MatrixXd& K,  // Key projection weights
        const Eigen::MatrixXd& V,  // Value projection weights
        int sequence_length,
        double head_dim) {
        
        // Compute spectral norms
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_q(Q);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_k(K);
        Eigen::JacobiSVD<Eigen::MatrixXd> svd_v(V);
        
        double q_norm = svd_q.singularValues()(0);
        double k_norm = svd_k.singularValues()(0);
        double v_norm = svd_v.singularValues()(0);
        
        // QK^T has norm bounded by ||Q|| * ||K||
        double qk_norm = q_norm * k_norm;
        
        // Scaling by 1/sqrt(head_dim)
        double scaled_qk_norm = qk_norm / std::sqrt(head_dim);
        
        // Maximum logit in softmax ≈ scaled_qk_norm * ||x||
        // Assume ||x|| ≈ sqrt(sequence_length) for normalized inputs
        double max_logit = scaled_qk_norm * std::sqrt(sequence_length);
        
        // Softmax curvature (from paper Theorem 5.7)
        double softmax_curv = 0.5 * std::exp(2.0 * max_logit);
        
        // Composition with matrix multiplications
        // κ_total ≈ κ_softmax * L_QK² + κ_QK * L_softmax
        // where L_QK ≈ qk_norm and κ_QK ≈ 0 (bilinear)
        double curvature = softmax_curv * qk_norm * qk_norm;
        
        // Lipschitz constant of full attention
        double lipschitz = v_norm * scaled_qk_norm;
        
        return LayerCurvature{
            curvature,
            lipschitz,
            "Attention",
            "Sequence length " + std::to_string(sequence_length) +
                ", head dim " + std::to_string(static_cast<int>(head_dim))
        };
    }
    
    // Matrix inversion (for certification examples)
    // From paper Example (matrix inversion): κ ≈ κ(A)³
    static LayerCurvature matrix_inverse(double condition_number) {
        // Curvature scales as κ³
        double curvature = 2.0 * std::pow(condition_number, 3.0);
        
        // Lipschitz constant scales as κ²
        double lipschitz = condition_number * condition_number;
        
        return LayerCurvature{
            curvature,
            lipschitz,
            "MatrixInverse",
            "Condition number " + std::to_string(condition_number)
        };
    }
    
    // Composition rule (from paper Theorem 3.4)
    // For g ∘ f: κ_{g∘f} ≤ κ_g * L_f² + κ_f * ||Dg||
    static LayerCurvature compose(
        const LayerCurvature& f,
        const LayerCurvature& g) {
        
        double composed_curvature = 
            g.curvature * f.lipschitz_constant * f.lipschitz_constant +
            f.curvature * g.lipschitz_constant;
        
        double composed_lipschitz = 
            g.lipschitz_constant * f.lipschitz_constant;
        
        return LayerCurvature{
            composed_curvature,
            composed_lipschitz,
            f.layer_type + " ∘ " + g.layer_type,
            "Composition of " + f.description + " and " + g.description
        };
    }
    
    // Sum of curvatures for multiple paths (e.g., residual connections)
    static LayerCurvature sum_paths(
        const std::vector<LayerCurvature>& paths) {
        
        double total_curvature = 0.0;
        double total_lipschitz = 0.0;
        
        for (const auto& path : paths) {
            total_curvature += path.curvature;
            total_lipschitz += path.lipschitz_constant;
        }
        
        return LayerCurvature{
            total_curvature,
            total_lipschitz,
            "SumOfPaths",
            "Sum of " + std::to_string(paths.size()) + " paths"
        };
    }
};

// Precision requirement computation (Theorem 5.7)
class PrecisionComputer {
public:
    // Compute minimum required mantissa bits
    // From paper: p ≥ log₂(c * κ * D² / ε)
    static int compute_minimum_precision(
        double curvature,
        double domain_diameter,
        double target_accuracy,
        double safety_constant = 2.0) {
        
        if (curvature <= 0.0) {
            // Linear case: just need to overcome roundoff
            return static_cast<int>(std::ceil(-std::log2(target_accuracy))) + 2;
        }
        
        // Apply formula from Theorem 5.7
        double arg = safety_constant * curvature * domain_diameter * domain_diameter / target_accuracy;
        
        // Handle numerical issues
        if (arg <= 0.0) {
            throw std::runtime_error("Invalid precision computation arguments");
        }
        
        int precision_bits = static_cast<int>(std::ceil(std::log2(arg)));
        
        // Add safety margin (2 bits as in proposal)
        precision_bits += 2;
        
        return precision_bits;
    }
    
    // Map precision bits to hardware type
    static std::string recommend_hardware(int precision_bits) {
        if (precision_bits <= 8) {
            return "int8 or fp8";
        } else if (precision_bits <= 11) {
            return "float16 (fp16)";
        } else if (precision_bits <= 16) {
            return "bfloat16";
        } else if (precision_bits <= 24) {
            return "float32 (fp32)";
        } else if (precision_bits <= 52) {
            return "float64 (fp64)";
        } else {
            return "extended precision (> fp64)";
        }
    }
    
    // Inverse: given hardware, what accuracy can we guarantee?
    static double guaranteed_accuracy(
        double curvature,
        double domain_diameter,
        int mantissa_bits,
        double safety_constant = 2.0) {
        
        if (curvature <= 0.0) {
            // Linear case
            return std::pow(2.0, -mantissa_bits + 2);
        }
        
        // From p = log₂(c * κ * D² / ε), solve for ε
        double eps = safety_constant * curvature * domain_diameter * domain_diameter / 
                     std::pow(2.0, mantissa_bits);
        
        return eps;
    }
};

} // namespace certified
} // namespace hnf
