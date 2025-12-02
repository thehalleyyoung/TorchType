#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <memory>
#include <unordered_map>
#include <functional>
#include <chrono>
#include <cmath>

namespace hnf {
namespace homotopy {

/**
 * @file homotopy_lr.hpp
 * @brief Curvature-Adaptive Learning Rate Scheduler based on HNF Theory
 * 
 * Implements Proposal 7: Homotopy Learning Rate for Transformers
 * 
 * Theoretical Foundation (from hnf_paper.tex):
 * - Training traces a path in both loss space γ: [0,T] → ℒ and parameter space γ̃: [0,T] → Θ
 * - Related by gradient flow: dγ̃/dt = -η · ∇_θ L(γ̃(t))
 * - Local curvature: κ^{curv}(θ) = ||∇²L(θ)|| · ||∇L(θ)||^{-2}
 * - Optimal learning rate: η(t) ∝ 1/κ^{curv}(γ̃(t))
 * 
 * Key Insight: Warmup emerges naturally from high initial curvature without explicit scheduling!
 */

// Forward declarations
class CurvatureEstimator;
class HomotopyLRScheduler;
class PerLayerHomotopyLR;

/**
 * @brief Configuration for Hutchinson's trace estimator
 * 
 * Hutchinson's method: E[v^T H v] = trace(H) for random v ~ N(0,I) or Rademacher
 * For spectral norm: ||H|| ≈ max eigenvalue estimated via power iteration on Hvp
 */
struct HutchinsonConfig {
    int num_samples;              // Number of random vectors for trace estimation
    int power_iterations;         // Power iteration steps for top eigenvalue
    double power_iter_tol;        // Convergence tolerance for power iteration
    bool use_rademacher;          // Use Rademacher {-1,+1} instead of Gaussian
    int estimation_frequency;     // Estimate every N steps (for efficiency)
    double ema_decay;             // Exponential moving average decay for smoothing
    
    HutchinsonConfig()
        : num_samples(10)
        , power_iterations(20)
        , power_iter_tol(1e-6)
        , use_rademacher(true)
        , estimation_frequency(10)
        , ema_decay(0.9)
    {}
};

/**
 * @brief Stores curvature metrics at a training step
 */
struct CurvatureMetrics {
    double spectral_norm_hessian;  // ||∇²L||_op
    double kappa_curv;              // κ^{curv} = ||∇²L|| / ||∇L||²
    double gradient_norm;           // ||∇L||
    double trace_hessian;           // tr(∇²L) from Hutchinson
    int step;
    std::chrono::time_point<std::chrono::steady_clock> timestamp;
    
    // Precision requirement from Theorem 4.7 (thm:obstruction)
    // p ≥ log₂(c · κ · D² / ε)
    double required_mantissa_bits(double diameter, double target_eps) const {
        if (kappa_curv <= 0 || target_eps <= 0) return 0.0;
        return std::log2((kappa_curv * diameter * diameter) / target_eps);
    }
};

/**
 * @brief Estimates loss landscape curvature using Hutchinson's method
 * 
 * Three methods implemented:
 * 1. Hutchinson trace estimator: tr(H) ≈ (1/m) Σ v_i^T H v_i
 * 2. Power iteration for spectral norm: ||H|| = max |λ_i|
 * 3. Lanczos iteration for more accurate eigenvalue bounds
 */
class CurvatureEstimator {
public:
    explicit CurvatureEstimator(const HutchinsonConfig& config = HutchinsonConfig());
    
    /**
     * @brief Estimate curvature κ = ||∇²L|| / ||∇L||² at current parameters
     * @param loss Scalar loss tensor (requires grad)
     * @param parameters Model parameters
     * @return CurvatureMetrics with all estimated quantities
     */
    CurvatureMetrics estimate(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Compute Hessian-vector product using Pearlmutter's trick
     * 
     * Hvp(v) = ∇(∇L · v) using forward-mode AD over backward-mode AD
     * This is the key primitive for all curvature estimation methods.
     */
    std::vector<torch::Tensor> hessian_vector_product(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters,
        const std::vector<torch::Tensor>& v);
    
    /**
     * @brief Estimate tr(H) using Hutchinson's method
     * tr(H) ≈ (1/m) Σ v_i^T H v_i where v_i ~ N(0,I) or Rademacher
     */
    double estimate_trace_hutchinson(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Estimate ||H|| using power iteration
     * Iterates: v_{k+1} = H v_k / ||H v_k|| until convergence
     */
    double estimate_spectral_norm_power(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Estimate top-k eigenvalues using Lanczos iteration
     * More accurate than power iteration but slower
     */
    std::vector<double> estimate_top_eigenvalues_lanczos(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters,
        int k = 5);
    
    // Accessors
    const HutchinsonConfig& config() const { return config_; }
    const std::vector<CurvatureMetrics>& history() const { return history_; }
    
private:
    HutchinsonConfig config_;
    std::vector<CurvatureMetrics> history_;
    int step_counter_ = 0;
    
    // Cached EMA values for smoothing
    double ema_spectral_norm_ = 0.0;
    double ema_trace_ = 0.0;
    bool ema_initialized_ = false;
    
    // Helper: generate random vector with same structure as parameters
    std::vector<torch::Tensor> generate_random_vector(
        const std::vector<torch::Tensor>& like,
        bool rademacher = false);
    
    // Helper: flatten parameter list to single tensor
    torch::Tensor flatten_tensors(const std::vector<torch::Tensor>& tensors);
    
    // Helper: unflatten tensor back to parameter structure
    std::vector<torch::Tensor> unflatten_like(
        torch::Tensor flat,
        const std::vector<torch::Tensor>& like);
    
    // Helper: compute dot product <a, b>
    double dot_product(
        const std::vector<torch::Tensor>& a,
        const std::vector<torch::Tensor>& b);
    
    // Helper: compute ||v||²
    double norm_squared(const std::vector<torch::Tensor>& v);
    
    // Helper: normalize vector v /= ||v||
    void normalize_inplace(std::vector<torch::Tensor>& v);
    
    // Helper: v *= scalar
    void scale_inplace(std::vector<torch::Tensor>& v, double scalar);
};

/**
 * @brief Homotopy-based learning rate scheduler
 * 
 * Core idea: η(t) = η_base / (1 + α · κ(t) / κ_target)
 * 
 * When curvature is high → LR decreases (need smaller steps)
 * When curvature is low → LR increases (can take bigger steps)
 * 
 * This naturally produces:
 * - Warmup: High initial curvature → low LR initially
 * - Adaptation: Adjust to local geometry automatically
 * - No hyperparameter tuning: Only need base_lr and target κ
 */
class HomotopyLRScheduler {
public:
    struct Config {
        double base_lr;              // Maximum learning rate
        double target_curvature;     // Target κ for stable training
        double min_lr;               // Floor to prevent too-small LR
        double max_lr;               // Ceiling
        double alpha;                // Adaptation strength
        int warmup_steps;            // Initial steps to collect curvature stats
        bool adaptive_target;        // Learn target_curvature from data
        double target_percentile;    // If adaptive: use this percentile of history
        
        Config()
            : base_lr(1e-3)
            , target_curvature(1e4)
            , min_lr(1e-7)
            , max_lr(1.0)
            , alpha(1.0)
            , warmup_steps(100)
            , adaptive_target(true)
            , target_percentile(0.75)
        {}
    };
    
    HomotopyLRScheduler(
        const Config& config,
        const HutchinsonConfig& hvp_config = HutchinsonConfig());
    
    /**
     * @brief Compute learning rate for current step
     * @param loss Current loss (for curvature estimation)
     * @param parameters Model parameters
     * @param step Training step number
     * @return Computed learning rate
     */
    double step(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters,
        int step_num);
    
    /**
     * @brief Apply computed LR to PyTorch optimizer
     */
    void apply_to_optimizer(torch::optim::Optimizer& optimizer);
    
    /**
     * @brief Get current learning rate (last computed)
     */
    double get_current_lr() const { return current_lr_; }
    
    /**
     * @brief Get current curvature estimate
     */
    double get_current_curvature() const { return current_curvature_; }
    
    /**
     * @brief Get curvature history
     */
    const std::vector<CurvatureMetrics>& get_curvature_history() const {
        return estimator_.history();
    }
    
    /**
     * @brief Export metrics to CSV for analysis
     */
    void export_metrics(const std::string& filename) const;
    
private:
    Config config_;
    CurvatureEstimator estimator_;
    double current_lr_;
    double current_curvature_;
    int step_count_ = 0;
    
    // Adaptive target curvature
    double adaptive_target_curvature() const;
    
    // Compute LR from curvature
    double compute_lr_from_curvature(double kappa);
};

/**
 * @brief Per-layer learning rates based on per-layer curvature
 * 
 * Different layers can have vastly different curvature:
 * - Attention layers: κ_attn ∝ exp(2 · max(QK^T))
 * - LayerNorm: κ_LN ∝ 1/σ²
 * - FFN: κ_FFN ∝ ||W||²
 * 
 * This scheduler assigns η_ℓ ∝ 1/κ_ℓ for each layer ℓ
 */
class PerLayerHomotopyLR {
public:
    struct Config {
        double base_lr;
        double min_lr;
        double max_lr;
        bool normalize_by_median;    // Scale so median layer gets base_lr
        int estimation_frequency;    // Estimate every N steps
        
        Config()
            : base_lr(1e-3)
            , min_lr(1e-7)
            , max_lr(1.0)
            , normalize_by_median(true)
            , estimation_frequency(10)
        {}
    };
    
    PerLayerHomotopyLR(
        const Config& config,
        const HutchinsonConfig& hvp_config = HutchinsonConfig());
    
    /**
     * @brief Register a layer to track
     * @param name Layer identifier (e.g., "layer.0.attn")
     * @param parameters Parameters belonging to this layer
     */
    void register_layer(
        const std::string& name,
        const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Compute per-layer learning rates
     * @param loss Total loss
     * @param step_num Training step
     * @return Map from layer name to recommended LR
     */
    std::unordered_map<std::string, double> step(
        torch::Tensor loss,
        int step_num);
    
    /**
     * @brief Apply to optimizer with per-parameter LR
     * Requires optimizer to support per-parameter options
     */
    void apply_to_optimizer(torch::optim::Optimizer& optimizer);
    
    /**
     * @brief Get curvature for a specific layer
     */
    double get_layer_curvature(const std::string& layer_name) const;
    
    /**
     * @brief Export per-layer metrics
     */
    void export_metrics(const std::string& filename) const;
    
private:
    Config config_;
    HutchinsonConfig hvp_config_;
    
    struct LayerInfo {
        std::vector<torch::Tensor> parameters;
        CurvatureEstimator estimator;
        double current_lr;
        double current_curvature;
        
        LayerInfo(const HutchinsonConfig& cfg) : estimator(cfg), current_lr(1e-3), current_curvature(1.0) {}
    };
    
    std::unordered_map<std::string, LayerInfo> layers_;
    int step_count_ = 0;
};

/**
 * @brief Curvature-aware gradient clipping
 * 
 * In high-curvature regions, clip more aggressively:
 * clip_norm = base_clip_norm / (1 + κ / κ_target)
 * 
 * This prevents gradient explosions when curvature is high.
 */
class CurvatureAwareGradientClipper {
public:
    struct Config {
        double base_clip_norm;
        double curvature_target;
        double min_clip_norm;
        
        Config()
            : base_clip_norm(1.0)
            , curvature_target(1e4)
            , min_clip_norm(0.01)
        {}
    };
    
    CurvatureAwareGradientClipper(const Config& config);
    
    /**
     * @brief Clip gradients based on current curvature
     * @param parameters Model parameters (with .grad)
     * @param curvature Current curvature estimate
     * @return Effective clip norm used
     */
    double clip_gradients(
        const std::vector<torch::Tensor>& parameters,
        double curvature);
    
private:
    Config config_;
};

/**
 * @brief Warmup strategy based on curvature tracking
 * 
 * Instead of fixed warmup steps, increase LR only when curvature allows:
 * - Start with very low LR
 * - Monitor κ(t)
 * - Increase LR when κ < threshold
 * - Stop increasing when κ rises again or target LR reached
 */
class CurvatureAwareWarmup {
public:
    struct Config {
        double target_lr;
        double initial_lr_fraction;  // Start at target_lr * this
        double curvature_threshold;  // Only increase if κ < this
        double increase_factor;      // Multiply LR by this each step
        int max_warmup_steps;        // Safety limit
        
        Config()
            : target_lr(1e-3)
            , initial_lr_fraction(0.01)
            , curvature_threshold(1e6)
            , increase_factor(1.01)
            , max_warmup_steps(1000)
        {}
    };
    
    CurvatureAwareWarmup(const Config& config, const HutchinsonConfig& hvp_config);
    
    /**
     * @brief Step warmup and return current LR
     * @param loss Current loss
     * @param parameters Model parameters
     * @return Current learning rate
     */
    double step(torch::Tensor loss, const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Check if warmup is complete
     */
    bool is_complete() const { return warmup_complete_; }
    
private:
    Config config_;
    CurvatureEstimator estimator_;
    double current_lr_;
    int step_count_ = 0;
    bool warmup_complete_ = false;
};

/**
 * @brief Integration with standard PyTorch optimizers
 * 
 * Wraps a PyTorch optimizer and automatically adjusts LR based on curvature
 */
class HomotopyOptimizer {
public:
    HomotopyOptimizer(
        std::shared_ptr<torch::optim::Optimizer> base_optimizer,
        const HomotopyLRScheduler::Config& scheduler_config,
        const HutchinsonConfig& hvp_config = HutchinsonConfig());
    
    /**
     * @brief Perform optimization step with automatic LR adjustment
     * @param loss Current loss (for curvature estimation)
     */
    void step(torch::Tensor loss);
    
    /**
     * @brief Zero gradients (pass-through to base optimizer)
     */
    void zero_grad() { base_optimizer_->zero_grad(); }
    
    /**
     * @brief Get current learning rate
     */
    double get_lr() const { return scheduler_->get_current_lr(); }
    
    /**
     * @brief Get curvature history
     */
    const std::vector<CurvatureMetrics>& get_curvature_history() const {
        return scheduler_->get_curvature_history();
    }
    
    /**
     * @brief Export training metrics
     */
    void export_metrics(const std::string& filename) const {
        scheduler_->export_metrics(filename);
    }
    
private:
    std::shared_ptr<torch::optim::Optimizer> base_optimizer_;
    std::unique_ptr<HomotopyLRScheduler> scheduler_;
    int step_count_ = 0;
    
    // Extract parameters from optimizer
    std::vector<torch::Tensor> get_parameters() const;
};

} // namespace homotopy
} // namespace hnf
