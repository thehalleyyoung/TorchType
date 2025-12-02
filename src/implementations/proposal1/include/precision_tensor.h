#pragma once

#include <torch/torch.h>
#include <memory>
#include <vector>
#include <string>
#include <functional>
#include <cmath>
#include <limits>
#include <iostream>
#include <iomanip>
#include <sstream>

namespace hnf {
namespace proposal1 {

// Hardware precision models from IEEE 754
enum class Precision {
    FP8,        // 4-bit mantissa (experimental)
    BFLOAT16,   // 7-bit mantissa
    FLOAT16,    // 10-bit mantissa
    FLOAT32,    // 23-bit mantissa
    FLOAT64,    // 52-bit mantissa
    FLOAT128    // 112-bit mantissa
};

inline int mantissa_bits(Precision p) {
    switch(p) {
        case Precision::FP8: return 4;
        case Precision::BFLOAT16: return 7;
        case Precision::FLOAT16: return 10;
        case Precision::FLOAT32: return 23;
        case Precision::FLOAT64: return 52;
        case Precision::FLOAT128: return 112;
    }
    return 23;
}

inline double machine_epsilon(Precision p) {
    return std::pow(2.0, -mantissa_bits(p));
}

inline std::string precision_name(Precision p) {
    switch(p) {
        case Precision::FP8: return "fp8";
        case Precision::BFLOAT16: return "bfloat16";
        case Precision::FLOAT16: return "fp16";
        case Precision::FLOAT32: return "fp32";
        case Precision::FLOAT64: return "fp64";
        case Precision::FLOAT128: return "fp128";
    }
    return "unknown";
}

// Forward declare for circular dependency
class PrecisionTensor;

// Curvature computation using second derivatives
// Implements κ_f^curv = sup_x ||D²f(x)|| · ||Df(x)^{-1}||²
class CurvatureComputer {
public:
    // Compute Hessian-vector product using double backprop
    static torch::Tensor hessian_vector_product(
        const torch::Tensor& output,
        const torch::Tensor& input,
        const torch::Tensor& vector
    );
    
    // Estimate operator norm of Hessian using power iteration
    static double estimate_hessian_norm(
        const std::function<torch::Tensor(const torch::Tensor&)>& f,
        const torch::Tensor& x,
        int num_iterations = 10
    );
    
    // Compute curvature for common operations (closed form when possible)
    static double exp_curvature(const torch::Tensor& x);
    static double log_curvature(const torch::Tensor& x);
    static double reciprocal_curvature(const torch::Tensor& x);
    static double sqrt_curvature(const torch::Tensor& x);
    static double power_curvature(const torch::Tensor& x, double exponent);
    static double matmul_curvature(const torch::Tensor& A, const torch::Tensor& B);
    static double softmax_curvature(const torch::Tensor& x);
    static double sigmoid_curvature(const torch::Tensor& x);
    static double tanh_curvature(const torch::Tensor& x);
    static double relu_curvature(const torch::Tensor& x);
    static double layer_norm_curvature(const torch::Tensor& x);
    static double batch_norm_curvature(const torch::Tensor& x, const torch::Tensor& running_mean, const torch::Tensor& running_var);
    static double logsumexp_curvature(const torch::Tensor& x, bool stable = true);
    static double attention_curvature(const torch::Tensor& Q, const torch::Tensor& K, const torch::Tensor& V);
    static double gelu_curvature(const torch::Tensor& x);
    static double silu_curvature(const torch::Tensor& x);
    static double conv2d_curvature(const torch::Tensor& input, const torch::Tensor& weight);
    static double div_curvature(const torch::Tensor& x, const torch::Tensor& y);
};

namespace ops {

// Forward declarations for friend access
PrecisionTensor add(const PrecisionTensor& a, const PrecisionTensor& b);
PrecisionTensor sub(const PrecisionTensor& a, const PrecisionTensor& b);
PrecisionTensor mul(const PrecisionTensor& a, const PrecisionTensor& b);

} // namespace ops

// Main PrecisionTensor class implementing Numerical Type from HNF paper
// Represents elements of the category NMet
class PrecisionTensor {
private:
    torch::Tensor data_;
    double lipschitz_const_;
    double curvature_;
    double domain_diameter_;
    Precision current_precision_;
    int required_mantissa_bits_;
    
    // Error functional Φ_f(ε, H) from Definition 3.3
    std::function<double(double, Precision)> error_functional_;
    
    // Computation history for graph tracing
    std::string operation_name_;
    std::vector<std::shared_ptr<PrecisionTensor>> parents_;
    
    // Friend declarations for operations that need to modify error_functional_
    friend PrecisionTensor ops::add(const PrecisionTensor& a, const PrecisionTensor& b);
    friend PrecisionTensor ops::sub(const PrecisionTensor& a, const PrecisionTensor& b);
    friend PrecisionTensor ops::mul(const PrecisionTensor& a, const PrecisionTensor& b);
    
    // Compute domain diameter from tensor statistics
    void compute_domain_diameter();
    
    // Compute precision requirement using Theorem 5.7 (Precision Obstruction)
    // p ≥ log₂(c·κ·D²/ε) where c is constant, κ is curvature, D is diameter
    void compute_precision_requirement(double target_accuracy);

public:
    // Constructors
    PrecisionTensor(
        const torch::Tensor& data,
        double lipschitz = 1.0,
        double curvature = 0.0,
        Precision precision = Precision::FLOAT32,
        const std::string& op_name = "input"
    );
    
    // Accessors
    const torch::Tensor& data() const { return data_; }
    torch::Tensor& data() { return data_; }
    double lipschitz() const { return lipschitz_const_; }
    double curvature() const { return curvature_; }
    double diameter() const { return domain_diameter_; }
    Precision current_precision() const { return current_precision_; }
    int required_bits() const { return required_mantissa_bits_; }
    const std::string& operation() const { return operation_name_; }
    
    // Set target accuracy and recompute precision requirement
    void set_target_accuracy(double eps);
    
    // Check if current hardware precision is sufficient
    bool is_precision_sufficient() const {
        return mantissa_bits(current_precision_) >= required_mantissa_bits_;
    }
    
    // Recommend minimum precision
    Precision recommend_precision() const;
    
    // Compute error propagation: Φ_f(ε_in, H)
    double propagate_error(double input_error) const {
        return error_functional_(input_error, current_precision_);
    }
    
    // Composition: create new PrecisionTensor from operation
    // Implements morphism composition in NMet
    static PrecisionTensor compose(
        const PrecisionTensor& input,
        const torch::Tensor& output_data,
        double new_lipschitz,
        double new_curvature,
        const std::string& op_name
    );
    
    // Pretty printing
    std::string to_string() const;
    friend std::ostream& operator<<(std::ostream& os, const PrecisionTensor& pt);
};

// Namespace for operations that return PrecisionTensor
// Each implements a numerical morphism in NMet
namespace ops {

// Arithmetic operations
PrecisionTensor add(const PrecisionTensor& a, const PrecisionTensor& b);
PrecisionTensor sub(const PrecisionTensor& a, const PrecisionTensor& b);
PrecisionTensor mul(const PrecisionTensor& a, const PrecisionTensor& b);
PrecisionTensor div(const PrecisionTensor& a, const PrecisionTensor& b);
PrecisionTensor matmul(const PrecisionTensor& a, const PrecisionTensor& b);

// Transcendental functions
PrecisionTensor exp(const PrecisionTensor& x);
PrecisionTensor log(const PrecisionTensor& x);
PrecisionTensor sqrt(const PrecisionTensor& x);
PrecisionTensor pow(const PrecisionTensor& x, double exponent);
PrecisionTensor reciprocal(const PrecisionTensor& x);

// Activation functions
PrecisionTensor relu(const PrecisionTensor& x);
PrecisionTensor sigmoid(const PrecisionTensor& x);
PrecisionTensor tanh(const PrecisionTensor& x);
PrecisionTensor gelu(const PrecisionTensor& x);
PrecisionTensor silu(const PrecisionTensor& x);
PrecisionTensor softmax(const PrecisionTensor& x, int64_t dim = -1);
PrecisionTensor log_softmax(const PrecisionTensor& x, int64_t dim = -1);

// Normalization
PrecisionTensor layer_norm(const PrecisionTensor& x, const std::vector<int64_t>& normalized_shape, double eps = 1e-5);
PrecisionTensor batch_norm(const PrecisionTensor& x, const torch::Tensor& running_mean, const torch::Tensor& running_var, double eps = 1e-5);

// Advanced operations
PrecisionTensor logsumexp(const PrecisionTensor& x, int64_t dim = -1);
PrecisionTensor attention(const PrecisionTensor& Q, const PrecisionTensor& K, const PrecisionTensor& V);
PrecisionTensor conv2d(const PrecisionTensor& input, const torch::Tensor& weight, const torch::Tensor& bias = torch::Tensor());

// Dropout (precision-preserving)
PrecisionTensor dropout(const PrecisionTensor& x, double p, bool training = true);

// Additional operations needed for tests
PrecisionTensor transpose(const PrecisionTensor& x);
PrecisionTensor mul_scalar(const PrecisionTensor& x, double scalar);
PrecisionTensor sum(const PrecisionTensor& x, int64_t dim = -1);
PrecisionTensor neg(const PrecisionTensor& x);

} // namespace ops

} // namespace proposal1
} // namespace hnf
