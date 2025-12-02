#pragma once

#include "curvature_profiler.hpp"
#include <torch/torch.h>
#include <vector>
#include <map>
#include <functional>

namespace hnf {
namespace profiler {
namespace advanced {

/**
 * @brief Riemannian metric tensor for the parameter space
 * 
 * Implements the Fisher Information Matrix and related geometric structures
 * from the HNF framework. This gives a true geometric understanding of the
 * training landscape beyond simple curvature.
 */
class RiemannianMetricTensor {
public:
    struct MetricData {
        torch::Tensor metric_tensor;  // G_ij = E[∂log p/∂θ_i ∂log p/∂θ_j]
        std::vector<double> eigenvalues;
        double condition_number;
        double volume_element;  // sqrt(det(G))
        double scalar_curvature;  // Ricci scalar
        
        // Christoffel symbols for geodesic computation
        torch::Tensor christoffel_symbols;
    };
    
    /**
     * @brief Compute the Fisher Information Matrix (metric tensor)
     * @param model Neural network model
     * @param data Sample data tensor for computation
     * @param target Sample target tensor
     * @return Metric tensor and derived quantities
     */
    static MetricData compute_metric_tensor(
        torch::nn::Module& model,
        torch::Tensor data,
        torch::Tensor target);
    
    /**
     * @brief Compute Ricci curvature tensor
     * 
     * The Ricci tensor measures how volumes change under parallel transport.
     * High Ricci curvature indicates the parameter space is "curved" and
     * optimization will be difficult.
     */
    static torch::Tensor compute_ricci_tensor(const MetricData& metric);
    
    /**
     * @brief Compute geodesic between two parameter configurations
     * 
     * Finds the shortest path in parameter space under the Riemannian metric.
     * This is the "natural" path for optimization.
     */
    static std::vector<torch::Tensor> compute_geodesic(
        const torch::Tensor& start_params,
        const torch::Tensor& end_params,
        const MetricData& metric,
        int num_steps = 100);
};

/**
 * @brief Sectional curvature computation
 * 
 * From HNF theory, sectional curvature K(π) for a 2-plane π measures
 * how geodesics diverge/converge. This is crucial for understanding
 * whether SGD trajectories will converge.
 */
class SectionalCurvature {
public:
    /**
     * @brief Compute sectional curvature in random 2-planes
     * @param metric Riemannian metric
     * @param num_samples Number of random planes to sample
     * @return Vector of sectional curvatures
     */
    static std::vector<double> sample_sectional_curvatures(
        const RiemannianMetricTensor::MetricData& metric,
        int num_samples = 100);
    
    /**
     * @brief Check if sectional curvature is uniformly bounded
     * 
     * If K(π) ≥ κ > 0 for all 2-planes π, then the space has positive
     * curvature and convergence guarantees apply.
     */
    static bool is_positively_curved(
        const RiemannianMetricTensor::MetricData& metric,
        double threshold = 0.0);
};

/**
 * @brief Curvature-based loss spike predictor
 * 
 * Uses historical curvature data to predict when loss spikes will occur.
 * This is a key application showing HNF theory has real predictive power.
 */
class LossSpikePredictor {
public:
    struct PredictionResult {
        bool spike_predicted;
        int steps_until_spike;  // Estimated number of steps
        double confidence;      // 0-1, how confident we are
        std::string cause;      // Which layer/region is problematic
        double recommended_lr_scale;  // Suggested LR adjustment
    };
    
    /**
     * @brief Train predictor on historical data
     * @param curvature_history Map of layer -> time series of curvatures
     * @param loss_history Time series of losses
     * @param spike_indices Indices where spikes occurred
     */
    void train(
        const std::map<std::string, std::vector<double>>& curvature_history,
        const std::vector<double>& loss_history,
        const std::vector<int>& spike_indices);
    
    /**
     * @brief Predict if a spike will occur soon
     * @param current_curvatures Current curvature values
     * @param recent_curvature_trend Recent time series
     * @return Prediction result with details
     */
    PredictionResult predict(
        const std::map<std::string, double>& current_curvatures,
        const std::map<std::string, std::vector<double>>& recent_curvature_trend);
    
private:
    // Simple ML model: linear regression on curvature features
    torch::Tensor weights_;
    double bias_;
    bool trained_ = false;
    
    // Feature engineering
    std::vector<double> extract_features(
        const std::map<std::string, double>& current,
        const std::map<std::string, std::vector<double>>& history);
};

/**
 * @brief Curvature flow optimization
 * 
 * Instead of standard gradient descent, follow the curvature flow:
 * dθ/dt = -∇f - λ κ^{curv} ∇κ^{curv}
 * 
 * This actively avoids high-curvature regions, potentially enabling
 * convergence on problems where standard methods fail.
 */
class CurvatureFlowOptimizer {
public:
    struct Config {
        double learning_rate;
        double curvature_penalty;  // λ in the flow equation
        double momentum;
        bool use_adaptive_penalty;  // Adjust λ based on curvature
        int warmup_steps;  // Steps before curvature kicks in
    };
    
    CurvatureFlowOptimizer(
        std::vector<torch::Tensor> parameters,
        const Config& config);
    
    /**
     * @brief Take an optimization step
     * @param loss Current loss
     * @param profiler Curvature profiler for computing κ
     */
    void step(torch::Tensor loss, CurvatureProfiler& profiler);
    
    /**
     * @brief Zero gradients (standard optimizer interface)
     */
    void zero_grad();
    
    /**
     * @brief Get current parameters
     */
    std::vector<torch::Tensor>& parameters() { return parameters_; }
    
private:
    std::vector<torch::Tensor> parameters_;
    std::vector<torch::Tensor> momentum_buffer_;
    Config config_;
    int step_count_ = 0;
    
    // Compute gradient of curvature w.r.t. parameters
    std::vector<torch::Tensor> compute_curvature_gradient(
        CurvatureProfiler& profiler,
        torch::Tensor loss);
};

/**
 * @brief Pathological problem generator
 * 
 * Creates optimization problems that are specifically designed to be
 * difficult for standard optimizers but potentially solvable with
 * curvature-aware methods.
 */
class PathologicalProblemGenerator {
public:
    enum class ProblemType {
        HIGH_CURVATURE_VALLEY,   // Narrow valley with κ >> 1
        SADDLE_PROLIFERATION,    // Many saddle points
        ILL_CONDITIONED_HESSIAN, // κ(H) >> 1
        OSCILLATORY_LANDSCAPE,   // Rapidly changing curvature
        MIXED_PRECISION_TRAP     // Requires >fp64 precision
    };
    
    /**
     * @brief Generate a pathological optimization problem
     * @param type Type of difficulty
     * @param dimension Problem dimension
     * @param severity How difficult (1-10 scale)
     * @return Loss function and ground truth minimum
     */
    static std::pair<
        std::function<torch::Tensor(torch::Tensor)>,
        torch::Tensor
    > generate(ProblemType type, int dimension, int severity);
    
    /**
     * @brief Test if an optimizer can solve the problem
     * @param optimizer Optimizer to test
     * @param problem Loss function
     * @param true_minimum Known minimum
     * @param max_iterations Maximum steps
     * @return Success rate and final error
     */
    static std::pair<bool, double> test_solver(
        std::function<void(torch::Tensor)> optimizer,
        std::function<torch::Tensor(torch::Tensor)> problem,
        const torch::Tensor& true_minimum,
        int max_iterations);
};

/**
 * @brief Curvature-guided neural architecture search
 * 
 * Use HNF curvature bounds to guide architecture design:
 * - Avoid architectures with provably high curvature
 * - Prefer structures with bounded compositional curvature
 * - Design layers that maintain low condition numbers
 */
class CurvatureGuidedNAS {
public:
    struct ArchitectureSpec {
        std::vector<int> layer_sizes;
        std::vector<std::string> activation_types;
        bool use_normalization;
        bool use_skip_connections;
        
        // Predicted properties from HNF theory
        double predicted_curvature;
        double predicted_condition_number;
        int required_precision_bits;
    };
    
    /**
     * @brief Evaluate an architecture before training
     * @param layer_sizes Layer dimensions
     * @param activations Activation function names
     * @return Predicted curvature and other metrics
     */
    static ArchitectureSpec evaluate_architecture(
        const std::vector<int>& layer_sizes,
        const std::vector<std::string>& activations);
    
    /**
     * @brief Search for architecture with bounded curvature
     * @param search_space Possible configurations
     * @param max_curvature Maximum acceptable curvature
     * @return Best architecture found
     */
    static ArchitectureSpec search(
        const std::vector<ArchitectureSpec>& search_space,
        double max_curvature);
};

/**
 * @brief Precision certificate generator
 * 
 * Uses Z3 SMT solver to formally verify precision requirements.
 * Generates certificates that prove a given precision is sufficient.
 */
class PrecisionCertificateGenerator {
public:
    struct Certificate {
        bool is_valid;
        int required_bits;
        std::string proof;  // Human-readable proof
        std::vector<std::string> assumptions;
        std::vector<std::string> conclusions;
    };
    
    /**
     * @brief Generate formal proof that precision is sufficient
     * @param curvature Measured curvature
     * @param diameter Domain diameter
     * @param target_error Target accuracy
     * @return Formal certificate
     */
    static Certificate generate_certificate(
        double curvature,
        double diameter,
        double target_error);
    
    /**
     * @brief Verify certificate using Z3
     * @param cert Certificate to verify
     * @return True if proof is valid
     */
    static bool verify_certificate(const Certificate& cert);
};

} // namespace advanced
} // namespace profiler
} // namespace hnf
