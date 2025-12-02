#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <chrono>

namespace hnf {
namespace profiler {

// Forward declarations
class CurvatureProfiler;
class TrainingMonitor;

/**
 * @brief Represents the curvature invariant κ_f^{curv} = (1/2)||D²f||_op
 * 
 * From HNF paper Definition 4.1 (def:curvature):
 * For a C² map f, the curvature at point a is defined as:
 * κ_f^{curv}(a) = (1/2) sup_{||h||=1} ||D²f_a(h,h)||
 * 
 * This measures the second-order deviation from linearity and provides
 * lower bounds on required mantissa bits via Theorem 4.7 (thm:obstruction):
 * p ≥ log₂(c · κ · D² / ε)
 */
struct CurvatureMetrics {
    double spectral_norm_hessian;  // ||D²f||_op
    double kappa_curv;              // (1/2)||D²f||_op
    double lipschitz_constant;      // L_f = ||Df||_op
    double condition_number;        // For layer operations
    double gradient_norm;           // ||∇f||
    int step;                       // Training step
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    
    // Precision requirement estimate from Theorem 4.7
    double required_mantissa_bits(double diameter, double target_eps) const {
        if (kappa_curv <= 0 || target_eps <= 0) return 0.0;
        // p ≥ log₂(c · κ · D² / ε) where c is a constant (we use c=1 conservatively)
        return std::log2((kappa_curv * diameter * diameter) / target_eps);
    }
};

/**
 * @brief Configuration for Hessian-vector product computation
 * 
 * Uses Pearlmutter's trick for efficient Hvp computation:
 * Hvp(v) = ∇(∇f · v) computed via forward-mode AD
 */
struct HvpConfig {
    int num_power_iterations;      // For spectral norm estimation
    double power_iter_tolerance;  // Convergence tolerance
    bool use_finite_differences; // Fallback method
    double fd_epsilon;            // Finite difference step
    int num_random_directions;      // For stochastic estimation
    
    HvpConfig()
        : num_power_iterations(20)
        , power_iter_tolerance(1e-6)
        , use_finite_differences(false)
        , fd_epsilon(1e-5)
        , num_random_directions(10)
    {}
};

/**
 * @brief Estimates the spectral norm ||D²f||_op via power iteration
 * 
 * For a layer with parameters θ and loss L, computes:
 * ||H||_op where H = ∇²L(θ)
 * 
 * This is done by power iteration on the Hessian-vector product operator.
 */
class HessianSpectralNormEstimator {
public:
    explicit HessianSpectralNormEstimator(const HvpConfig& config = HvpConfig());
    
    /**
     * @brief Estimate spectral norm of Hessian at current point
     * @param loss Scalar loss tensor (must have grad_fn)
     * @param parameters Parameters to compute Hessian w.r.t.
     * @return Estimated ||D²L||_op
     */
    double estimate_spectral_norm(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Compute Hessian-vector product H·v using Pearlmutter's trick
     * @param loss Scalar loss
     * @param parameters Parameters
     * @param v Direction vector
     * @return Hv (same structure as parameters)
     */
    std::vector<torch::Tensor> hessian_vector_product(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters,
        const std::vector<torch::Tensor>& v);
    
private:
    HvpConfig config_;
    
    // Helper: flatten parameter list to single vector
    torch::Tensor flatten_params(const std::vector<torch::Tensor>& params);
    
    // Helper: unflatten vector back to parameter structure
    std::vector<torch::Tensor> unflatten_like(
        torch::Tensor flat,
        const std::vector<torch::Tensor>& like);
};

/**
 * @brief Per-layer curvature profiler with hook-based tracking
 * 
 * Implements the profiling system from Proposal 5:
 * - Computes κ_ℓ^{curv}(t) at each training step
 * - Tracks time series {κ_ℓ(t)}
 * - Correlates with gradient norms, loss spikes, etc.
 */
class CurvatureProfiler {
public:
    CurvatureProfiler(
        torch::nn::Module& model,
        const HvpConfig& hvp_config = HvpConfig());
    
    /**
     * @brief Compute curvature metrics for all tracked layers
     * @param loss Scalar loss tensor
     * @param step Current training step
     * @return Map from layer name to curvature metrics
     */
    std::unordered_map<std::string, CurvatureMetrics> compute_curvature(
        torch::Tensor loss,
        int step);
    
    /**
     * @brief Add a layer to track
     * @param name Layer identifier  
     * @param module Pointer to the layer module
     */
    void track_layer(const std::string& name, torch::nn::Module* module);
    
    /**
     * @brief Add a layer to track (shared_ptr version)
     * @param name Layer identifier
     * @param module Pointer to the layer module
     */
    void track_layer_shared(const std::string& name, std::shared_ptr<torch::nn::Module> module);
    
    /**
     * @brief Get curvature history for a layer
     */
    const std::vector<CurvatureMetrics>& get_history(const std::string& layer_name) const;
    
    /**
     * @brief Get all tracked layer names
     */
    std::vector<std::string> get_tracked_layers() const;
    
    /**
     * @brief Clear all history
     */
    void clear_history();
    
    /**
     * @brief Export metrics to CSV file
     */
    void export_to_csv(const std::string& filename) const;
    
private:
    torch::nn::Module& model_;
    HessianSpectralNormEstimator hessian_estimator_;
    
    // Layer name -> module mapping (we store raw pointers since model owns them)
    std::unordered_map<std::string, torch::nn::Module*> tracked_layers_;
    
    // Layer name -> shared_ptr for modules we own
    std::unordered_map<std::string, std::shared_ptr<torch::nn::Module>> owned_modules_;
    
    // Layer name -> history of metrics
    std::unordered_map<std::string, std::vector<CurvatureMetrics>> history_;
    
    // Compute Lipschitz constant for a layer
    double compute_lipschitz_constant(torch::nn::Module* module);
    
    // Extract parameters from a module
    std::vector<torch::Tensor> get_module_parameters(torch::nn::Module* module);
};

/**
 * @brief Training stability monitor with predictive warnings
 * 
 * Implements the monitoring system from Proposal 5:
 * - Detects high curvature regions
 * - Predicts training failures before they occur
 * - Suggests interventions (LR reduction, etc.)
 */
class TrainingMonitor {
public:
    struct Config {
        double warning_threshold;    // κ threshold for warnings
        double danger_threshold;     // κ threshold for danger
        int prediction_horizon;      // Steps to look ahead
        int min_history_length;       // Min points for prediction
        bool auto_intervene;       // Auto-reduce LR on danger
        double lr_reduction_factor;  // LR *= this on intervention
        
        Config() 
            : warning_threshold(1e6)
            , danger_threshold(1e9)
            , prediction_horizon(100)
            , min_history_length(20)
            , auto_intervene(false)
            , lr_reduction_factor(0.5)
        {}
    };
    
    explicit TrainingMonitor(
        CurvatureProfiler& profiler,
        const Config& config = Config());
    
    /**
     * @brief Process one training step and generate warnings
     * @param loss Current loss value
     * @param step Training step
     * @return Vector of warning messages
     */
    std::vector<std::string> on_step(torch::Tensor loss, int step);
    
    /**
     * @brief Predict if training will fail within horizon
     * @return (will_fail, layer_name, projected_curvature)
     */
    std::tuple<bool, std::string, double> predict_failure();
    
    /**
     * @brief Get suggested learning rate adjustment
     * @return Recommended LR multiplier (1.0 = no change)
     */
    double suggest_lr_adjustment() const;
    
    /**
     * @brief Check if curvature exceeds thresholds
     */
    bool is_warning_state() const { return warning_state_; }
    bool is_danger_state() const { return danger_state_; }
    
private:
    CurvatureProfiler& profiler_;
    Config config_;
    
    bool warning_state_ = false;
    bool danger_state_ = false;
    std::string problematic_layer_;
    
    // Fit exponential model to curvature time series
    double extrapolate_curvature(
        const std::vector<CurvatureMetrics>& history,
        int horizon) const;
};

/**
 * @brief Curvature-aware learning rate scheduler
 * 
 * Adapts learning rate based on local curvature:
 * η(t) ∝ 1 / κ^{curv}(γ̃(t))
 * 
 * This implements the homotopy-based LR scheduling from the HNF framework.
 */
class CurvatureAdaptiveLR {
public:
    struct Config {
        double base_lr;
        double target_curvature;  // Target κ for stable training
        double min_lr;
        double max_lr;
        bool use_max_curvature;  // Use max over layers vs. average
        
        Config()
            : base_lr(1e-3)
            , target_curvature(1e4)
            , min_lr(1e-6)
            , max_lr(1.0)
            , use_max_curvature(true)
        {}
    };
    
    explicit CurvatureAdaptiveLR(
        CurvatureProfiler& profiler,
        const Config& config = Config());
    
    /**
     * @brief Compute recommended learning rate for current step
     * @param step Training step
     * @return Recommended learning rate
     */
    double compute_lr(int step);
    
    /**
     * @brief Apply computed LR to optimizer
     */
    void step(torch::optim::Optimizer& optimizer, int training_step);
    
private:
    CurvatureProfiler& profiler_;
    Config config_;
    double current_lr_;
};

} // namespace profiler
} // namespace hnf
