#pragma once

#include "numerical_type.h"
#include "curvature_database.h"
#include <torch/torch.h>
#include <cmath>

namespace hnf {
namespace ops {

// Wrapped operations that return NumericalType with tracked precision

// Matrix multiplication
inline NumericalType matmul(const NumericalType& A, const NumericalType& B) {
    torch::Tensor result = torch::matmul(A.data, B.data);
    
    // Compute curvature
    double kappa = CurvatureDatabase::matmul_curvature(A.data, B.data);
    
    // Lipschitz constant for matmul
    double L_A = torch::linalg::matrix_norm(A.data, "fro").item<double>();
    double L_B = torch::linalg::matrix_norm(B.data, "fro").item<double>();
    double L = std::max(L_A, L_B);
    
    NumericalType output(result, L, kappa, A.hardware);
    
    // Compose error functionals
    output.error_functional = [A, B, L, kappa](double eps, HardwareModel H) {
        double phi_A = A.error_functional(eps, H);
        double phi_B = B.error_functional(eps, H);
        double eps_mach = machine_epsilon(H);
        
        // Error from both inputs plus roundoff
        return L * (phi_A + phi_B) + kappa * eps_mach;
    };
    
    return output;
}

// Exponential
inline NumericalType exp(const NumericalType& x) {
    torch::Tensor result = torch::exp(x.data);
    
    double kappa = CurvatureDatabase::exp_curvature(x.data);
    double x_max = x.data.abs().max().item<double>();
    double L = std::exp(x_max);  // Lipschitz constant of exp
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Natural logarithm
inline NumericalType log(const NumericalType& x) {
    torch::Tensor result = torch::log(x.data);
    
    double kappa = CurvatureDatabase::log_curvature(x.data);
    double x_min = x.data.min().item<double>();
    double L = 1.0 / std::max(x_min, 1e-10);  // Lipschitz constant
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Softmax
inline NumericalType softmax(const NumericalType& x, int64_t dim = -1) {
    torch::Tensor result = torch::softmax(x.data, dim);
    
    double kappa = CurvatureDatabase::softmax_curvature(x.data);
    double L = 1.0;  // Softmax is 1-Lipschitz
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// ReLU activation
inline NumericalType relu(const NumericalType& x) {
    torch::Tensor result = torch::relu(x.data);
    
    double kappa = CurvatureDatabase::relu_curvature(x.data);  // = 0
    double L = 1.0;  // ReLU is 1-Lipschitz
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L](double eps, HardwareModel H) {
        // Piecewise linear, so only input error propagates
        return L * x.error_functional(eps, H);
    };
    
    return output;
}

// Sigmoid
inline NumericalType sigmoid(const NumericalType& x) {
    torch::Tensor result = torch::sigmoid(x.data);
    
    double kappa = CurvatureDatabase::sigmoid_curvature(x.data);
    double L = 0.25;  // Sigmoid has Lipschitz constant 1/4
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Tanh
inline NumericalType tanh(const NumericalType& x) {
    torch::Tensor result = torch::tanh(x.data);
    
    double kappa = CurvatureDatabase::tanh_curvature(x.data);
    double L = 1.0;  // Tanh is 1-Lipschitz
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Addition
inline NumericalType add(const NumericalType& x, const NumericalType& y) {
    torch::Tensor result = x.data + y.data;
    
    double kappa = CurvatureDatabase::add_curvature(x.data, y.data);  // = 0
    double L = 1.0;
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, y, L](double eps, HardwareModel H) {
        double phi_x = x.error_functional(eps, H);
        double phi_y = y.error_functional(eps, H);
        double eps_mach = machine_epsilon(H);
        return phi_x + phi_y + eps_mach;  // Addition roundoff
    };
    
    return output;
}

// Multiplication (element-wise)
inline NumericalType mul(const NumericalType& x, const NumericalType& y) {
    torch::Tensor result = x.data * y.data;
    
    double kappa = CurvatureDatabase::mul_curvature(x.data, y.data);  // = 0 (bilinear)
    double L_x = y.data.abs().max().item<double>();
    double L_y = x.data.abs().max().item<double>();
    double L = std::max(L_x, L_y);
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, y, L](double eps, HardwareModel H) {
        double phi_x = x.error_functional(eps, H);
        double phi_y = y.error_functional(eps, H);
        double eps_mach = machine_epsilon(H);
        return L * (phi_x + phi_y) + eps_mach;
    };
    
    return output;
}

// Division
inline NumericalType div(const NumericalType& x, const NumericalType& y) {
    torch::Tensor result = x.data / y.data;
    
    double kappa = CurvatureDatabase::div_curvature(x.data, y.data);
    double y_min = y.data.abs().min().item<double>();
    double L = 1.0 / std::max(y_min, 1e-10);
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, y, L, kappa](double eps, HardwareModel H) {
        double phi_x = x.error_functional(eps, H);
        double phi_y = y.error_functional(eps, H);
        return L * (phi_x + phi_y) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Square root
inline NumericalType sqrt(const NumericalType& x) {
    torch::Tensor result = torch::sqrt(x.data);
    
    double kappa = CurvatureDatabase::sqrt_curvature(x.data);
    double x_min = x.data.min().item<double>();
    double L = 1.0 / (2.0 * std::sqrt(std::max(x_min, 1e-10)));
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Layer normalization
inline NumericalType layer_norm(const NumericalType& x, const std::vector<int64_t>& normalized_shape) {
    // Compute mean and variance
    auto mean = x.data.mean();
    auto var = x.data.var();
    double eps_ln = 1e-5;
    
    torch::Tensor result = (x.data - mean) / torch::sqrt(var + eps_ln);
    
    double kappa = CurvatureDatabase::layer_norm_curvature(x.data);
    double L = 1.0 / std::sqrt(var.item<double>() + eps_ln);
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Log-sum-exp (stable version)
inline NumericalType logsumexp(const NumericalType& x, int64_t dim = -1) {
    // Use the stable max-shifted version (Gallery Example 6)
    auto x_max = std::get<0>(x.data.max(dim, /*keepdim=*/true));
    auto shifted = x.data - x_max;
    auto exp_shifted = torch::exp(shifted);
    auto sum_exp = exp_shifted.sum(dim, /*keepdim=*/true);
    torch::Tensor result = x_max + torch::log(sum_exp);
    
    result = result.squeeze(dim);
    
    // Stable version has bounded curvature
    double kappa = CurvatureDatabase::logsumexp_curvature(x.data, true);
    double L = 1.0;  // LSE is 1-Lipschitz
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Power function
inline NumericalType pow(const NumericalType& x, double exponent) {
    torch::Tensor result = torch::pow(x.data, exponent);
    
    double kappa = CurvatureDatabase::power_curvature(x.data, exponent);
    
    double x_max = x.data.abs().max().item<double>();
    double L = std::abs(exponent) * std::pow(x_max, std::abs(exponent - 1));
    
    NumericalType output(result, L, kappa, x.hardware);
    output.error_functional = [x, L, kappa](double eps, HardwareModel H) {
        return L * x.error_functional(eps, H) + kappa * machine_epsilon(H);
    };
    
    return output;
}

// Attention mechanism (simplified)
inline NumericalType attention(
    const NumericalType& Q,
    const NumericalType& K,
    const NumericalType& V
) {
    int64_t d = Q.data.size(-1);
    double sqrt_d = std::sqrt(static_cast<double>(d));
    
    // Compute QK^T / sqrt(d)
    auto scores = torch::matmul(Q.data, K.data.transpose(-2, -1)) / sqrt_d;
    
    // Apply softmax
    auto attn_weights = torch::softmax(scores, -1);
    
    // Multiply by V
    torch::Tensor result = torch::matmul(attn_weights, V.data);
    
    // Compute curvature
    double kappa = CurvatureDatabase::attention_curvature(Q.data, K.data, V.data);
    
    // Lipschitz constant is product of norms
    double norm_Q = torch::linalg::matrix_norm(Q.data, "fro").item<double>();
    double norm_K = torch::linalg::matrix_norm(K.data, "fro").item<double>();
    double norm_V = torch::linalg::matrix_norm(V.data, "fro").item<double>();
    double L = (norm_Q * norm_K * norm_V) / sqrt_d;
    
    NumericalType output(result, L, kappa, Q.hardware);
    output.error_functional = [Q, K, V, L, kappa](double eps, HardwareModel H) {
        double phi_Q = Q.error_functional(eps, H);
        double phi_K = K.error_functional(eps, H);
        double phi_V = V.error_functional(eps, H);
        return L * (phi_Q + phi_K + phi_V) + kappa * machine_epsilon(H);
    };
    
    return output;
}

} // namespace ops
} // namespace hnf
