#include "precision_tensor.h"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace hnf {
namespace proposal1 {

// ============================================================================
// CurvatureComputer Implementation
// ============================================================================

torch::Tensor CurvatureComputer::hessian_vector_product(
    const torch::Tensor& output,
    const torch::Tensor& input,
    const torch::Tensor& vector
) {
    // Compute gradient
    auto grad = torch::autograd::grad({output}, {input}, {}, /*retain_graph=*/true, /*create_graph=*/true)[0];
    
    // Compute Hessian-vector product: H·v where H is Hessian
    auto hvp = torch::autograd::grad({grad}, {input}, {vector}, /*retain_graph=*/false)[0];
    
    return hvp;
}

double CurvatureComputer::estimate_hessian_norm(
    const std::function<torch::Tensor(const torch::Tensor&)>& f,
    const torch::Tensor& x,
    int num_iterations
) {
    // Power iteration to estimate largest eigenvalue of Hessian
    torch::NoGradGuard no_grad;
    
    // Start with random vector
    auto v = torch::randn_like(x);
    v = v / torch::norm(v);
    
    double eigenvalue = 0.0;
    
    for (int i = 0; i < num_iterations; ++i) {
        auto x_copy = x.detach().clone().requires_grad_(true);
        auto output = f(x_copy);
        
        // Compute Hessian-vector product
        auto Hv = hessian_vector_product(output, x_copy, v);
        
        eigenvalue = torch::dot(v.flatten(), Hv.flatten()).item<double>();
        
        // Normalize
        double norm = torch::norm(Hv).item<double>();
        if (norm > 1e-10) {
            v = Hv / norm;
        } else {
            break;
        }
    }
    
    return std::abs(eigenvalue);
}

double CurvatureComputer::exp_curvature(const torch::Tensor& x) {
    // f(x) = exp(x), f''(x) = exp(x)
    // κ = ||exp(x)|| for worst case
    double x_max = x.abs().max().item<double>();
    return std::exp(x_max);
}

double CurvatureComputer::log_curvature(const torch::Tensor& x) {
    // f(x) = log(x), f''(x) = -1/x²
    // κ = 1/(2·x_min²) from domain restriction
    double x_min = x.min().item<double>();
    if (x_min <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    return 0.5 / (x_min * x_min);
}

double CurvatureComputer::reciprocal_curvature(const torch::Tensor& x) {
    // f(x) = 1/x, f''(x) = 2/x³
    // κ = 1/x_min³ (from Example 5.2.3 in paper)
    double x_min = x.abs().min().item<double>();
    if (x_min == 0) {
        return std::numeric_limits<double>::infinity();
    }
    return 1.0 / (x_min * x_min * x_min);
}

double CurvatureComputer::sqrt_curvature(const torch::Tensor& x) {
    // f(x) = sqrt(x), f''(x) = -1/(4·x^(3/2))
    // κ = 1/(4·x_min^(3/2))
    double x_min = x.min().item<double>();
    if (x_min <= 0) {
        return std::numeric_limits<double>::infinity();
    }
    return 0.25 / std::pow(x_min, 1.5);
}

double CurvatureComputer::power_curvature(const torch::Tensor& x, double n) {
    // f(x) = x^n, f''(x) = n(n-1)x^(n-2)
    double x_val = (n >= 2) ? x.abs().max().item<double>() : x.abs().min().item<double>();
    if (x_val <= 0 && n < 2) {
        return std::numeric_limits<double>::infinity();
    }
    return std::abs(n * (n - 1)) * std::pow(x_val, std::abs(n - 2));
}

double CurvatureComputer::matmul_curvature(const torch::Tensor& A, const torch::Tensor& B) {
    // Matrix multiplication is bilinear, so Hessian is zero
    // But we account for conditioning
    // Conservative: κ ≈ ||A||·||B|| (Frobenius norm)
    auto norm_A = torch::norm(A.flatten()).item<double>();
    auto norm_B = torch::norm(B.flatten()).item<double>();
    return norm_A * norm_B;
}

double CurvatureComputer::softmax_curvature(const torch::Tensor& /* x */) {
    // Softmax has bounded Hessian
    // κ ≤ 0.5 from paper (Gallery Example 2)
    return 0.5;
}

double CurvatureComputer::sigmoid_curvature(const torch::Tensor& /* x */) {
    // σ(x) = 1/(1+exp(-x)), σ''(x) = σ(x)(1-σ(x))(1-2σ(x))
    // Bounded by 0.25
    return 0.25;
}

double CurvatureComputer::tanh_curvature(const torch::Tensor& /* x */) {
    // tanh''(x) = -2·tanh(x)·sech²(x)
    // Bounded by 0.5
    return 0.5;
}

double CurvatureComputer::relu_curvature(const torch::Tensor& /* x */) {
    // Piecewise linear, second derivative is zero a.e.
    return 0.0;
}

double CurvatureComputer::layer_norm_curvature(const torch::Tensor& x) {
    // f(x) = (x - μ) / σ
    // κ ≈ 1/σ²
    auto var = x.var().item<double>();
    if (var < 1e-10) {
        return 1e10;
    }
    return 1.0 / var;
}

double CurvatureComputer::batch_norm_curvature(
    const torch::Tensor& /* x */,
    const torch::Tensor& /* running_mean */,
    const torch::Tensor& running_var
) {
    // Similar to layer norm
    double var = running_var.mean().item<double>();
    if (var < 1e-10) {
        return 1e10;
    }
    return 1.0 / var;
}

double CurvatureComputer::logsumexp_curvature(const torch::Tensor& x, bool stable) {
    if (stable) {
        // Max-shifted version has bounded curvature
        return 1.0;
    } else {
        // Naive version: dominated by exp
        return exp_curvature(x);
    }
}

double CurvatureComputer::attention_curvature(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V
) {
    // From Gallery Example 4 in paper
    // κ ≈ (||Q||·||K||·||V||/√d) · exp(2||QK^T/√d||)
    double norm_Q = torch::norm(Q.flatten()).item<double>();
    double norm_K = torch::norm(K.flatten()).item<double>();
    double norm_V = torch::norm(V.flatten()).item<double>();
    
    int64_t d = Q.size(-1);
    double sqrt_d = std::sqrt(static_cast<double>(d));
    
    auto QKT = torch::matmul(Q, K.transpose(-2, -1)) / sqrt_d;
    double norm_QKT = torch::abs(QKT).max().item<double>();
    
    return (norm_Q * norm_K * norm_V / sqrt_d) * std::exp(2.0 * norm_QKT);
}

double CurvatureComputer::gelu_curvature(const torch::Tensor& /* x */) {
    // GELU is smooth with bounded second derivative
    // Approximate bound
    return 1.0;
}

double CurvatureComputer::silu_curvature(const torch::Tensor& /* x */) {
    // SiLU(x) = x·σ(x), smooth activation
    // Approximate bound
    return 1.0;
}

double CurvatureComputer::conv2d_curvature(const torch::Tensor& /* input */, const torch::Tensor& weight) {
    // Convolution is linear in input, so Hessian w.r.t. input is zero
    // But account for weight norm
    auto norm_w = torch::norm(weight.flatten()).item<double>();
    return norm_w;
}

double CurvatureComputer::div_curvature(const torch::Tensor& x, const torch::Tensor& y) {
    // Division: f(x,y) = x/y
    // Second derivative involves 1/y³ terms
    double y_min = y.abs().min().item<double>();
    if (y_min == 0) {
        return std::numeric_limits<double>::infinity();
    }
    double x_max = x.abs().max().item<double>();
    return 2.0 * x_max / (y_min * y_min * y_min);
}

// ============================================================================
// PrecisionTensor Implementation
// ============================================================================

void PrecisionTensor::compute_domain_diameter() {
    if (data_.numel() == 0) {
        domain_diameter_ = 0.0;
        return;
    }
    
    auto flat = data_.flatten();
    double max_val = flat.max().item<double>();
    double min_val = flat.min().item<double>();
    domain_diameter_ = std::abs(max_val - min_val);
}

void PrecisionTensor::compute_precision_requirement(double target_accuracy) {
    // Theorem 5.7 (Precision Obstruction Theorem):
    // p ≥ log₂(c·κ·D²/ε)
    // where c is a constant (we use c=2 conservatively)
    
    if (curvature_ <= 0 || domain_diameter_ <= 0) {
        required_mantissa_bits_ = mantissa_bits(Precision::FLOAT32);
        return;
    }
    
    const double c = 2.0;  // Conservative constant from theorem
    double arg = (c * curvature_ * domain_diameter_ * domain_diameter_) / target_accuracy;
    
    if (arg <= 1.0) {
        required_mantissa_bits_ = 0;
    } else if (std::isinf(arg) || std::isnan(arg)) {
        required_mantissa_bits_ = 200;  // Beyond practical precision
    } else {
        required_mantissa_bits_ = static_cast<int>(std::ceil(std::log2(arg)));
    }
}

PrecisionTensor::PrecisionTensor(
    const torch::Tensor& data,
    double lipschitz,
    double curvature,
    Precision precision,
    const std::string& op_name
) : data_(data),
    lipschitz_const_(lipschitz),
    curvature_(curvature),
    current_precision_(precision),
    operation_name_(op_name)
{
    compute_domain_diameter();
    compute_precision_requirement(1e-6);  // Default target
    
    // Default error functional: Φ_f(ε, H) = L·ε + Δ(H)
    // where Δ(H) accounts for machine roundoff
    error_functional_ = [lipschitz, curvature](double eps, Precision H) {
        double eps_mach = machine_epsilon(H);
        return lipschitz * eps + curvature * eps_mach + eps_mach;
    };
}

void PrecisionTensor::set_target_accuracy(double eps) {
    compute_precision_requirement(eps);
}

Precision PrecisionTensor::recommend_precision() const {
    if (required_mantissa_bits_ <= 4) return Precision::FP8;
    if (required_mantissa_bits_ <= 7) return Precision::BFLOAT16;
    if (required_mantissa_bits_ <= 10) return Precision::FLOAT16;
    if (required_mantissa_bits_ <= 23) return Precision::FLOAT32;
    if (required_mantissa_bits_ <= 52) return Precision::FLOAT64;
    return Precision::FLOAT128;
}

PrecisionTensor PrecisionTensor::compose(
    const PrecisionTensor& input,
    const torch::Tensor& output_data,
    double new_lipschitz,
    double new_curvature,
    const std::string& op_name
) {
    // Composition of Lipschitz constants: L_{g∘f} = L_g · L_f
    double composed_lipschitz = new_lipschitz * input.lipschitz_const_;
    
    // Composition of curvature (from Proposition in Section 5)
    // κ_{g∘f} ≤ κ_g·L_f² + κ_f·L_g
    double composed_curvature = 
        new_curvature * input.lipschitz_const_ * input.lipschitz_const_ +
        input.curvature_ * new_lipschitz;
    
    PrecisionTensor result(output_data, composed_lipschitz, composed_curvature, 
                          input.current_precision_, op_name);
    
    // Composed error functional (Theorem 3.8: Stability Composition Theorem)
    // Φ_{g∘f}(ε, H) = Φ_g(Φ_f(ε, H), H) + L_g · Φ_f(ε, H)
    auto input_error_func = input.error_functional_;
    result.error_functional_ = [input_error_func, new_lipschitz, new_curvature]
                               (double eps, Precision H) {
        double phi_f = input_error_func(eps, H);
        double eps_mach = machine_epsilon(H);
        double phi_g = new_lipschitz * phi_f + new_curvature * eps_mach + eps_mach;
        return phi_g + new_lipschitz * phi_f;
    };
    
    // Track parent
    result.parents_.push_back(std::make_shared<PrecisionTensor>(input));
    
    return result;
}

std::string PrecisionTensor::to_string() const {
    std::ostringstream oss;
    oss << "PrecisionTensor(op=" << operation_name_ 
        << ", shape=[";
    for (int i = 0; i < data_.dim(); ++i) {
        oss << data_.size(i);
        if (i < data_.dim() - 1) oss << ",";
    }
    oss << "], L=" << std::scientific << std::setprecision(2) << lipschitz_const_
        << ", κ=" << std::scientific << std::setprecision(2) << curvature_
        << ", D=" << std::scientific << std::setprecision(2) << domain_diameter_
        << ", bits_req=" << required_mantissa_bits_
        << ", current=" << precision_name(current_precision_)
        << ", recommend=" << precision_name(recommend_precision())
        << ")";
    return oss.str();
}

std::ostream& operator<<(std::ostream& os, const PrecisionTensor& pt) {
    os << pt.to_string();
    return os;
}

// ============================================================================
// Operations Implementation
// ============================================================================

namespace ops {

PrecisionTensor add(const PrecisionTensor& a, const PrecisionTensor& b) {
    auto result_data = a.data() + b.data();
    
    // Addition is linear, κ = 0
    double new_lipschitz = 1.0;
    double new_curvature = 0.0;
    
    // Compose with both inputs
    auto composed = PrecisionTensor::compose(a, result_data, new_lipschitz, new_curvature, "add");
    
    // Account for second input
    composed.error_functional_ = [a, b](double eps, Precision H) {
        double eps_mach = machine_epsilon(H);
        return a.propagate_error(eps) + b.propagate_error(eps) + eps_mach;
    };
    
    return composed;
}

PrecisionTensor sub(const PrecisionTensor& a, const PrecisionTensor& b) {
    auto result_data = a.data() - b.data();
    double new_lipschitz = 1.0;
    double new_curvature = 0.0;
    
    auto composed = PrecisionTensor::compose(a, result_data, new_lipschitz, new_curvature, "sub");
    composed.error_functional_ = [a, b](double eps, Precision H) {
        double eps_mach = machine_epsilon(H);
        return a.propagate_error(eps) + b.propagate_error(eps) + eps_mach;
    };
    
    return composed;
}

PrecisionTensor mul(const PrecisionTensor& a, const PrecisionTensor& b) {
    auto result_data = a.data() * b.data();
    
    // Multiplication is bilinear, κ = 0
    double max_a = a.data().abs().max().item<double>();
    double max_b = b.data().abs().max().item<double>();
    double L = std::max(max_a, max_b);
    double new_curvature = 0.0;
    
    auto composed = PrecisionTensor::compose(a, result_data, L, new_curvature, "mul");
    composed.error_functional_ = [a, b, L](double eps, Precision H) {
        double eps_mach = machine_epsilon(H);
        return L * (a.propagate_error(eps) + b.propagate_error(eps)) + eps_mach;
    };
    
    return composed;
}

PrecisionTensor div(const PrecisionTensor& a, const PrecisionTensor& b) {
    auto result_data = a.data() / b.data();
    
    double b_min = b.data().abs().min().item<double>();
    double new_lipschitz = 1.0 / std::max(b_min, 1e-10);
    double new_curvature = CurvatureComputer::div_curvature(a.data(), b.data());
    
    return PrecisionTensor::compose(a, result_data, new_lipschitz, new_curvature, "div");
}

PrecisionTensor matmul(const PrecisionTensor& a, const PrecisionTensor& b) {
    auto result_data = torch::matmul(a.data(), b.data());
    
    double norm_a = torch::norm(a.data().flatten()).item<double>();
    double norm_b = torch::norm(b.data().flatten()).item<double>();
    double new_lipschitz = std::max(norm_a, norm_b);
    double new_curvature = CurvatureComputer::matmul_curvature(a.data(), b.data());
    
    return PrecisionTensor::compose(a, result_data, new_lipschitz, new_curvature, "matmul");
}

PrecisionTensor exp(const PrecisionTensor& x) {
    auto result_data = torch::exp(x.data());
    
    double x_max = x.data().abs().max().item<double>();
    double new_lipschitz = std::exp(x_max);
    double new_curvature = CurvatureComputer::exp_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "exp");
}

PrecisionTensor log(const PrecisionTensor& x) {
    auto result_data = torch::log(x.data());
    
    double x_min = x.data().min().item<double>();
    double new_lipschitz = 1.0 / std::max(x_min, 1e-10);
    double new_curvature = CurvatureComputer::log_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "log");
}

PrecisionTensor sqrt(const PrecisionTensor& x) {
    auto result_data = torch::sqrt(x.data());
    
    double x_min = x.data().min().item<double>();
    double new_lipschitz = 1.0 / (2.0 * std::sqrt(std::max(x_min, 1e-10)));
    double new_curvature = CurvatureComputer::sqrt_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "sqrt");
}

PrecisionTensor pow(const PrecisionTensor& x, double exponent) {
    auto result_data = torch::pow(x.data(), exponent);
    
    double x_val = x.data().abs().max().item<double>();
    double new_lipschitz = std::abs(exponent) * std::pow(x_val, std::abs(exponent - 1));
    double new_curvature = CurvatureComputer::power_curvature(x.data(), exponent);
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "pow");
}

PrecisionTensor reciprocal(const PrecisionTensor& x) {
    auto result_data = torch::reciprocal(x.data());
    
    double x_min = x.data().abs().min().item<double>();
    double new_lipschitz = 1.0 / (x_min * x_min);
    double new_curvature = CurvatureComputer::reciprocal_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "reciprocal");
}

PrecisionTensor relu(const PrecisionTensor& x) {
    auto result_data = torch::relu(x.data());
    
    double new_lipschitz = 1.0;  // ReLU is 1-Lipschitz
    double new_curvature = 0.0;   // Piecewise linear
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "relu");
}

PrecisionTensor sigmoid(const PrecisionTensor& x) {
    auto result_data = torch::sigmoid(x.data());
    
    double new_lipschitz = 0.25;  // Sigmoid has max derivative 1/4
    double new_curvature = CurvatureComputer::sigmoid_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "sigmoid");
}

PrecisionTensor tanh(const PrecisionTensor& x) {
    auto result_data = torch::tanh(x.data());
    
    double new_lipschitz = 1.0;  // Tanh is 1-Lipschitz
    double new_curvature = CurvatureComputer::tanh_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "tanh");
}

PrecisionTensor gelu(const PrecisionTensor& x) {
    auto result_data = torch::gelu(x.data());
    
    double new_lipschitz = 1.0;  // GELU is approximately 1-Lipschitz
    double new_curvature = CurvatureComputer::gelu_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "gelu");
}

PrecisionTensor silu(const PrecisionTensor& x) {
    auto result_data = x.data() * torch::sigmoid(x.data());
    
    double new_lipschitz = 1.1;  // SiLU Lipschitz constant
    double new_curvature = CurvatureComputer::silu_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "silu");
}

PrecisionTensor softmax(const PrecisionTensor& x, int64_t dim) {
    auto result_data = torch::softmax(x.data(), dim);
    
    double new_lipschitz = 1.0;  // Softmax is 1-Lipschitz
    double new_curvature = CurvatureComputer::softmax_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "softmax");
}

PrecisionTensor log_softmax(const PrecisionTensor& x, int64_t dim) {
    auto result_data = torch::log_softmax(x.data(), dim);
    
    double new_lipschitz = 1.0;
    double new_curvature = 1.0;  // Composition of log and softmax
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "log_softmax");
}

PrecisionTensor layer_norm(const PrecisionTensor& x, const std::vector<int64_t>& /* normalized_shape */, double eps) {
    auto mean = x.data().mean();
    auto var = x.data().var();
    auto result_data = (x.data() - mean) / torch::sqrt(var + eps);
    
    double std_dev = std::sqrt(var.item<double>() + eps);
    double new_lipschitz = 1.0 / std_dev;
    double new_curvature = CurvatureComputer::layer_norm_curvature(x.data());
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "layer_norm");
}

PrecisionTensor batch_norm(const PrecisionTensor& x, const torch::Tensor& running_mean, 
                           const torch::Tensor& running_var, double eps) {
    auto result_data = (x.data() - running_mean) / torch::sqrt(running_var + eps);
    
    double std_dev = std::sqrt(running_var.mean().item<double>() + eps);
    double new_lipschitz = 1.0 / std_dev;
    double new_curvature = CurvatureComputer::batch_norm_curvature(x.data(), running_mean, running_var);
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "batch_norm");
}

PrecisionTensor logsumexp(const PrecisionTensor& x, int64_t dim) {
    // Use stable version (Gallery Example 6)
    auto x_max = std::get<0>(x.data().max(dim, /*keepdim=*/true));
    auto shifted = x.data() - x_max;
    auto exp_shifted = torch::exp(shifted);
    auto sum_exp = exp_shifted.sum(dim, /*keepdim=*/true);
    auto result_data = (x_max + torch::log(sum_exp)).squeeze(dim);
    
    double new_lipschitz = 1.0;  // LSE is 1-Lipschitz
    double new_curvature = CurvatureComputer::logsumexp_curvature(x.data(), true);
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "logsumexp");
}

PrecisionTensor attention(const PrecisionTensor& Q, const PrecisionTensor& K, const PrecisionTensor& V) {
    int64_t d = Q.data().size(-1);
    double sqrt_d = std::sqrt(static_cast<double>(d));
    
    auto scores = torch::matmul(Q.data(), K.data().transpose(-2, -1)) / sqrt_d;
    auto attn_weights = torch::softmax(scores, -1);
    auto result_data = torch::matmul(attn_weights, V.data());
    
    double norm_Q = torch::norm(Q.data().flatten()).item<double>();
    double norm_K = torch::norm(K.data().flatten()).item<double>();
    double norm_V = torch::norm(V.data().flatten()).item<double>();
    double new_lipschitz = (norm_Q * norm_K * norm_V) / sqrt_d;
    double new_curvature = CurvatureComputer::attention_curvature(Q.data(), K.data(), V.data());
    
    return PrecisionTensor::compose(Q, result_data, new_lipschitz, new_curvature, "attention");
}

PrecisionTensor conv2d(const PrecisionTensor& input, const torch::Tensor& weight, const torch::Tensor& bias) {
    torch::Tensor result_data;
    if (bias.defined()) {
        result_data = torch::conv2d(input.data(), weight, bias);
    } else {
        result_data = torch::conv2d(input.data(), weight);
    }
    
    double norm_w = torch::norm(weight.flatten()).item<double>();
    double new_lipschitz = norm_w;
    double new_curvature = CurvatureComputer::conv2d_curvature(input.data(), weight);
    
    return PrecisionTensor::compose(input, result_data, new_lipschitz, new_curvature, "conv2d");
}

PrecisionTensor dropout(const PrecisionTensor& x, double p, bool training) {
    auto result_data = training ? torch::dropout(x.data(), p, true) : x.data();
    
    double new_lipschitz = training ? (1.0 / (1.0 - p)) : 1.0;
    double new_curvature = 0.0;  // Dropout is piecewise linear
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "dropout");
}

PrecisionTensor transpose(const PrecisionTensor& x) {
    auto result_data = x.data().transpose(-2, -1);
    
    double new_lipschitz = x.lipschitz();
    double new_curvature = x.curvature();
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "transpose");
}

PrecisionTensor mul_scalar(const PrecisionTensor& x, double scalar) {
    auto result_data = x.data() * scalar;
    
    double new_lipschitz = std::abs(scalar) * x.lipschitz();
    double new_curvature = scalar * scalar * x.curvature();
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "mul_scalar");
}

PrecisionTensor sum(const PrecisionTensor& x, int64_t dim) {
    auto result_data = (dim == -1) ? x.data().sum() : x.data().sum(dim);
    
    double new_lipschitz = x.lipschitz();
    double new_curvature = x.curvature();
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "sum");
}

PrecisionTensor neg(const PrecisionTensor& x) {
    auto result_data = -x.data();
    
    double new_lipschitz = x.lipschitz();
    double new_curvature = x.curvature();
    
    return PrecisionTensor::compose(x, result_data, new_lipschitz, new_curvature, "neg");
}

} // namespace ops

} // namespace proposal1
} // namespace hnf
