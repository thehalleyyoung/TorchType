#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <optional>

namespace hnf {
namespace quantization {

/**
 * @brief Precision requirements from HNF Theorem 4.7 (Precision Obstruction Theorem)
 * 
 * For a C³ morphism f with curvature κ_f > 0 on domain of diameter D:
 * p ≥ log₂(c · κ_f · D² / ε) mantissa bits are NECESSARY
 * 
 * This is a sharp lower bound - no algorithm can achieve better precision
 * with fewer bits. This implementation computes per-layer curvature and
 * derives optimal bit allocations.
 */
struct PrecisionRequirement {
    std::string layer_name;
    double curvature;              // κ_f^{curv} from Definition 4.1
    double diameter;               // Domain diameter D
    double lipschitz_constant;     // L_f for composition
    double target_accuracy;        // Desired ε
    int min_bits_required;         // Theorem 4.7 lower bound
    int allocated_bits;            // Actual allocation (≥ min_bits_required)
    double expected_error;         // Predicted quantization error
    
    // Compute minimum bits from Theorem 4.7
    int compute_min_bits(double constant_c = 1.0) const {
        if (curvature <= 0 || target_accuracy <= 0) return 4;
        double bits = std::log2((constant_c * curvature * diameter * diameter) / target_accuracy);
        return std::max(4, static_cast<int>(std::ceil(bits)));
    }
};

/**
 * @brief Layer-specific statistics for quantization analysis
 */
struct LayerStatistics {
    std::string name;
    torch::Tensor weight;
    torch::Tensor bias;
    
    // Activation statistics from calibration
    double input_min;
    double input_max;
    double input_mean;
    double input_std;
    double output_min;
    double output_max;
    
    // Curvature metrics
    double condition_number;       // For linear layers: σ_max/σ_min
    double spectral_norm;         // ||W||_op
    double hessian_spectral_norm; // ||D²f||_op (estimated)
    double curvature;             // κ_f^{curv} = (1/2)||D²f||_op
    
    // Size information
    int64_t num_parameters;
    std::vector<int64_t> weight_shape;
    
    LayerStatistics() 
        : input_min(0), input_max(0), input_mean(0), input_std(1)
        , output_min(0), output_max(0)
        , condition_number(1), spectral_norm(1), hessian_spectral_norm(0)
        , curvature(1), num_parameters(0) {}
};

/**
 * @brief Quantization configuration for a single layer
 */
struct LayerQuantConfig {
    std::string name;
    int bits;                     // Bit width (4, 6, 8, 16, etc.)
    bool quantize_weights;
    bool quantize_activations;
    double scale;                 // Quantization scale factor
    double zero_point;            // Quantization zero point
    
    LayerQuantConfig() 
        : bits(8), quantize_weights(true), quantize_activations(false)
        , scale(1.0), zero_point(0.0) {}
};

/**
 * @brief Main analyzer for curvature-based quantization
 * 
 * Implements the algorithm from Proposal #9:
 * 1. Collect activation statistics via calibration
 * 2. Compute per-layer curvature estimates
 * 3. Determine bit allocations using Theorem 4.7
 * 4. Apply quantization with certified error bounds
 */
class CurvatureQuantizationAnalyzer {
public:
    explicit CurvatureQuantizationAnalyzer(
        torch::nn::Module& model,
        double target_accuracy = 1e-3,
        int min_bits = 4,
        int max_bits = 16);
    
    /**
     * @brief Run calibration to collect activation statistics
     * @param data_loader Calibration dataset
     * @param num_batches Number of batches to process (-1 = all)
     */
    void calibrate(
        const std::vector<torch::Tensor>& calibration_data,
        int num_batches = -1);
    
    /**
     * @brief Compute curvature for all quantizable layers
     * 
     * Uses different methods based on layer type:
     * - Linear: condition number of weight matrix
     * - Conv2d: spectral norm estimation
     * - LayerNorm: variance-based curvature
     * - Softmax: exponential curvature from max input
     */
    void compute_curvature();
    
    /**
     * @brief Optimize bit allocation given a total bit budget
     * @param total_bits Total bit budget (measured in bits × parameters)
     * @return Map from layer name to bit allocation
     */
    std::unordered_map<std::string, int> optimize_bit_allocation(
        double average_bits);
    
    /**
     * @brief Allocate bits based on accuracy requirements (Theorem 4.7)
     * @param target_accuracy Target accuracy ε for each layer
     * @return Map from layer name to bit allocation
     */
    std::unordered_map<std::string, int> allocate_by_accuracy(
        double target_accuracy);
    
    /**
     * @brief Get precision requirements for all layers
     */
    std::vector<PrecisionRequirement> get_precision_requirements() const;
    
    /**
     * @brief Get layer statistics
     */
    const std::unordered_map<std::string, LayerStatistics>& get_layer_stats() const {
        return layer_stats_;
    }
    
    /**
     * @brief Estimate total quantization error using HNF composition (Theorem 3.4)
     * 
     * For composition f_n ∘ ... ∘ f_1:
     * Φ_{f_n ∘ ... ∘ f_1}(ε) ≤ Σᵢ (∏ⱼ₌ᵢ₊₁ⁿ Lⱼ) · Φᵢ(εᵢ)
     */
    double estimate_total_error(
        const std::unordered_map<std::string, int>& bit_allocation) const;
    
private:
    torch::nn::Module& model_;
    double target_accuracy_;
    int min_bits_;
    int max_bits_;
    
    // Layer statistics collected during calibration
    std::unordered_map<std::string, LayerStatistics> layer_stats_;
    
    // Layer execution order (for compositional error analysis)
    std::vector<std::string> layer_order_;
    
    // Hooks for activation capture
    struct HookHandle {
        std::string name;
        torch::OrderedDict<std::string, torch::nn::Module>::Iterator module_it;
    };
    std::vector<HookHandle> hooks_;
    
    // Helper methods
    void register_hooks();
    void remove_hooks();
    
    double compute_linear_curvature(const torch::Tensor& weight);
    double compute_conv_curvature(const torch::Tensor& weight);
    double compute_layernorm_curvature(double variance, int normalized_dim);
    double compute_softmax_curvature(double max_input);
    
    // Calibration callback
    void activation_hook(
        const std::string& name,
        const torch::Tensor& input,
        const torch::Tensor& output);
};

/**
 * @brief Applies quantization with per-layer bit widths
 * 
 * Implements symmetric quantization: Q(x) = round(x / scale) where
 * scale = max(|x|) / (2^(bits-1) - 1)
 */
class PrecisionAwareQuantizer {
public:
    explicit PrecisionAwareQuantizer(
        const std::unordered_map<std::string, LayerQuantConfig>& config);
    
    /**
     * @brief Quantize a model in-place with per-layer precision
     */
    void quantize_model(torch::nn::Module& model);
    
    /**
     * @brief Quantize a single tensor
     */
    torch::Tensor quantize_tensor(
        const torch::Tensor& tensor,
        int bits,
        double* out_scale = nullptr,
        double* out_zero_point = nullptr);
    
    /**
     * @brief Dequantize a tensor
     */
    torch::Tensor dequantize_tensor(
        const torch::Tensor& quantized,
        double scale,
        double zero_point,
        int bits);
    
    /**
     * @brief Create quantization-aware training wrapper
     * 
     * Uses straight-through estimator for backprop
     */
    torch::nn::Module create_qat_model(torch::nn::Module& model);
    
private:
    std::unordered_map<std::string, LayerQuantConfig> config_;
    
    void quantize_linear(
        torch::nn::Linear& layer,
        const LayerQuantConfig& config);
    
    void quantize_conv2d(
        torch::nn::Conv2d& layer,
        const LayerQuantConfig& config);
};

/**
 * @brief Bit-width optimizer using curvature-guided allocation
 * 
 * Solves the optimization problem:
 * minimize Σᵢ κᵢ · 2^(-bᵢ)  (total weighted quantization error)
 * subject to Σᵢ bᵢ · |θᵢ| ≤ B  (bit budget constraint)
 *           bₘᵢₙ ≤ bᵢ ≤ bₘₐₓ    (bit range constraints)
 * 
 * where κᵢ is curvature, bᵢ is bits for layer i, |θᵢ| is parameter count
 */
class BitWidthOptimizer {
public:
    BitWidthOptimizer(
        const std::unordered_map<std::string, LayerStatistics>& layer_stats,
        int min_bits,
        int max_bits);
    
    /**
     * @brief Optimize bit allocation for a given average bit budget
     * @param average_bits Target average bits per parameter
     * @return Bit allocation per layer
     */
    std::unordered_map<std::string, int> optimize(double average_bits);
    
    /**
     * @brief Allocate bits proportional to log(curvature)
     * 
     * Simple heuristic: bᵢ ∝ log₂(κᵢ)
     */
    std::unordered_map<std::string, int> proportional_allocation(
        double average_bits);
    
    /**
     * @brief Solve using gradient descent on relaxed problem
     */
    std::unordered_map<std::string, int> gradient_based_optimization(
        double average_bits,
        int max_iterations = 100);
    
    /**
     * @brief Greedy allocation: give more bits to high-curvature layers first
     */
    std::unordered_map<std::string, int> greedy_allocation(
        double average_bits);
    
private:
    std::unordered_map<std::string, LayerStatistics> layer_stats_;
    int min_bits_;
    int max_bits_;
    
    double compute_error(const std::unordered_map<std::string, double>& bits_float);
    void project_to_budget(
        std::unordered_map<std::string, double>& bits_float,
        double average_bits);
};

/**
 * @brief Validation and testing utilities
 */
class QuantizationValidator {
public:
    /**
     * @brief Compare original vs quantized model accuracy
     */
    static double evaluate_accuracy_degradation(
        torch::nn::Module& original,
        torch::nn::Module& quantized,
        const std::vector<std::pair<torch::Tensor, torch::Tensor>>& test_data);
    
    /**
     * @brief Measure actual quantization error vs theoretical prediction
     */
    static double measure_actual_error(
        const torch::Tensor& original_output,
        const torch::Tensor& quantized_output);
    
    /**
     * @brief Verify Theorem 4.7: check that allocated bits meet requirements
     */
    static bool verify_precision_requirements(
        const std::vector<PrecisionRequirement>& requirements,
        const std::unordered_map<std::string, int>& allocation);
    
    /**
     * @brief Print detailed quantization report
     */
    static void print_quantization_report(
        const CurvatureQuantizationAnalyzer& analyzer,
        const std::unordered_map<std::string, int>& allocation);
};

} // namespace quantization
} // namespace hnf
