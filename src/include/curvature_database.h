#pragma once

#include "numerical_type.h"
#include <torch/torch.h>
#include <cmath>

namespace hnf {

// Curvature computation for primitive operations
// Based on Section 5.3 and the Gallery of Applications

class CurvatureDatabase {
public:
    // Matrix multiplication: f(A,B) = AB
    // From Example 5.3.1: κ ≤ ||A||·||B|| / σ_min(AB)²
    static double matmul_curvature(const torch::Tensor& A, const torch::Tensor& B) {
        // Compute operator norms (spectral norms for matrices)
        double norm_A = torch::linalg::matrix_norm(A, "fro").item<double>();
        double norm_B = torch::linalg::matrix_norm(B, "fro").item<double>();
        
        // For the full curvature, we'd need singular values of AB
        // Conservative estimate: κ ≤ ||A|| · ||B|| assuming well-conditioned
        return norm_A * norm_B;
    }
    
    // Exponential: f(x) = exp(x)
    // From Gallery Example 6: κ = exp(2||x||_∞)
    static double exp_curvature(const torch::Tensor& x) {
        double x_max = x.abs().max().item<double>();
        return std::exp(2.0 * x_max);
    }
    
    // Logarithm: f(x) = log(x)
    // From domain restriction discussion: κ = 1/(2·x_min²)
    static double log_curvature(const torch::Tensor& x) {
        double x_min = x.min().item<double>();
        if (x_min <= 0) {
            return std::numeric_limits<double>::infinity();
        }
        return 1.0 / (2.0 * x_min * x_min);
    }
    
    // Reciprocal: f(x) = 1/x
    // From Example 5.2.3: κ = 2/x_min³
    static double reciprocal_curvature(const torch::Tensor& x) {
        double x_min = x.abs().min().item<double>();
        if (x_min == 0) {
            return std::numeric_limits<double>::infinity();
        }
        return 2.0 / (x_min * x_min * x_min);
    }
    
    // Softmax: f(x)_i = exp(x_i) / Σ exp(x_j)
    // From Gallery Example 2 and Section 5.3: κ ≤ O(n·exp(2(x_max - x_min)))
    static double softmax_curvature(const torch::Tensor& x) {
        double x_max = x.max().item<double>();
        double x_min = x.min().item<double>();
        int64_t n = x.numel();
        
        // The actual curvature involves the Hessian structure
        // Conservative bound from the paper
        return static_cast<double>(n) * std::exp(2.0 * (x_max - x_min));
    }
    
    // ReLU: f(x) = max(0, x)
    // Piecewise linear, so κ = 0 (second derivative is zero almost everywhere)
    static double relu_curvature(const torch::Tensor& x) {
        return 0.0;
    }
    
    // Sigmoid: f(x) = 1/(1 + exp(-x))
    // The second derivative is bounded: |f''(x)| = |f(x)(1-f(x))(1-2f(x))| ≤ 0.25
    static double sigmoid_curvature(const torch::Tensor& x) {
        // Bounded curvature
        return 0.25;
    }
    
    // Tanh: f(x) = tanh(x)
    // Similar bounded curvature
    static double tanh_curvature(const torch::Tensor& x) {
        // |tanh''(x)| = 2|tanh(x)|(1 - tanh²(x))| ≤ 0.5
        return 0.5;
    }
    
    // Layer normalization: f(x) = (x - μ) / σ
    // From the proposal: κ = O(1/σ_min²)
    static double layer_norm_curvature(const torch::Tensor& x) {
        auto mean = x.mean();
        auto var = x.var();
        double std_dev = std::sqrt(var.item<double>() + 1e-5);  // Add epsilon
        
        if (std_dev < 1e-5) {
            return 1e10;  // Very high curvature when variance is tiny
        }
        
        return 1.0 / (std_dev * std_dev);
    }
    
    // Square root: f(x) = sqrt(x)
    // From Example 5.2.3: κ = 1/(4·x_min^(3/2))
    static double sqrt_curvature(const torch::Tensor& x) {
        double x_min = x.min().item<double>();
        if (x_min <= 0) {
            return std::numeric_limits<double>::infinity();
        }
        return 1.0 / (4.0 * std::pow(x_min, 1.5));
    }
    
    // Addition: f(x,y) = x + y
    // Linear operation, κ = 0
    static double add_curvature(const torch::Tensor& x, const torch::Tensor& y) {
        return 0.0;
    }
    
    // Multiplication: f(x,y) = x * y (element-wise)
    // Bilinear, κ = 0 (Hessian is zero)
    static double mul_curvature(const torch::Tensor& x, const torch::Tensor& y) {
        return 0.0;
    }
    
    // Division: f(x,y) = x / y
    // Second derivative involves 1/y³ terms
    static double div_curvature(const torch::Tensor& x, const torch::Tensor& y) {
        double y_min = y.abs().min().item<double>();
        if (y_min == 0) {
            return std::numeric_limits<double>::infinity();
        }
        double x_max = x.abs().max().item<double>();
        return 2.0 * x_max / (y_min * y_min * y_min);
    }
    
    // Power: f(x) = x^n
    // |f''(x)| = |n(n-1)x^(n-2)|
    static double power_curvature(const torch::Tensor& x, double n) {
        if (std::abs(n) < 1.0) {
            // Fractional power
            double x_min = x.min().item<double>();
            if (x_min <= 0) return std::numeric_limits<double>::infinity();
            return std::abs(n * (n - 1)) * std::pow(x_min, n - 2);
        } else {
            // Integer power
            double x_max = x.abs().max().item<double>();
            return std::abs(n * (n - 1)) * std::pow(x_max, std::abs(n - 2));
        }
    }
    
    // Attention mechanism: f(Q,K,V) = softmax(QK^T/√d)V
    // From Gallery Example 4: κ ≤ O(||Q||·||K||·||V||/d · exp(2||QK^T||/√d))
    static double attention_curvature(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& V
    ) {
        double norm_Q = torch::linalg::matrix_norm(Q, "fro").item<double>();
        double norm_K = torch::linalg::matrix_norm(K, "fro").item<double>();
        double norm_V = torch::linalg::matrix_norm(V, "fro").item<double>();
        
        int64_t d = Q.size(-1);
        double sqrt_d = std::sqrt(static_cast<double>(d));
        
        // Compute QK^T and its norm
        auto QKT = torch::matmul(Q, K.transpose(-2, -1)) / sqrt_d;
        double norm_QKT = torch::abs(QKT).max().item<double>();
        
        // The curvature bound from the paper
        return (norm_Q * norm_K * norm_V / sqrt_d) * std::exp(2.0 * norm_QKT);
    }
    
    // Batch normalization (similar to layer norm)
    static double batch_norm_curvature(const torch::Tensor& x) {
        return layer_norm_curvature(x);
    }
    
    // Log-sum-exp: LSE(x) = log(Σ exp(x_i))
    // From Gallery Example 6: using the shifted version has κ = 1
    // The naive version has unbounded curvature
    static double logsumexp_curvature(const torch::Tensor& x, bool shifted = true) {
        if (shifted) {
            // The max-shifted version: stable
            return 1.0;
        } else {
            // Naive version: dominated by exp
            return exp_curvature(x);
        }
    }
};

} // namespace hnf
