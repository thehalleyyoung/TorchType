#pragma once

#include <torch/torch.h>
#include <vector>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

namespace hnf {
namespace profiler {

/**
 * @brief Exact Hessian computation and analysis
 * 
 * Implements rigorous Hessian computation as per HNF paper Definition 4.1.
 * Unlike the gradient proxy in curvature_profiler.cpp, this computes the
 * actual second derivative matrix and its spectral norm.
 * 
 * From HNF paper:
 * κ_f^{curv}(a) = (1/2) sup_{||h||=1} ||D²f_a(h,h)||
 *               = (1/2) ||D²f_a||_op (spectral norm)
 */
class ExactHessianComputer {
public:
    struct HessianMetrics {
        double spectral_norm;          // ||H||_op (largest eigenvalue)
        double frobenius_norm;         // ||H||_F
        double trace;                  // tr(H)
        double determinant;            // det(H) 
        std::vector<double> eigenvalues;  // All eigenvalues
        double condition_number;       // ||H|| * ||H^{-1}||
        int rank;                      // Numerical rank
        bool is_positive_definite;     // All eigenvalues > 0
        
        // HNF-specific metrics
        double kappa_curv;             // (1/2) * spectral_norm
        double precision_requirement_bits(double diameter, double eps) const {
            // Theorem 4.7: p ≥ log₂(c · κ · D² / ε)
            if (kappa_curv <= 0 || eps <= 0) return 0.0;
            return std::log2((kappa_curv * diameter * diameter) / eps);
        }
    };
    
    /**
     * @brief Compute exact Hessian matrix for a scalar loss
     * 
     * This is the ground truth implementation that computes the full
     * n×n Hessian matrix H where H_ij = ∂²L/∂θ_i∂θ_j
     * 
     * WARNING: This is O(n²) in memory and O(n³) for eigendecomposition.
     * Only use for small models (< 10k parameters) or subsets of parameters.
     * 
     * @param loss Scalar loss tensor
     * @param parameters Parameters to compute Hessian with respect to
     * @return Hessian matrix as Eigen::MatrixXd
     */
    static Eigen::MatrixXd compute_hessian_matrix(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Compute comprehensive Hessian metrics
     * 
     * Computes the full Hessian and extracts all relevant metrics including
     * eigenvalues, spectral norm, condition number, etc.
     * 
     * @param loss Scalar loss
     * @param parameters Parameters
     * @return Complete Hessian analysis
     */
    static HessianMetrics compute_metrics(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters);
    
    /**
     * @brief Compute Hessian spectral norm via randomized power iteration
     * 
     * More efficient than full eigendecomposition for large matrices.
     * Uses Hutchinson's trace estimator with power iteration.
     * 
     * @param loss Scalar loss
     * @param parameters Parameters
     * @param num_iterations Power iteration count
     * @param num_samples Random samples for stochastic estimation
     * @return Estimated ||H||_op
     */
    static double compute_spectral_norm_stochastic(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters,
        int num_iterations = 20,
        int num_samples = 10);
    
    /**
     * @brief Verify curvature bound via finite differences
     * 
     * Checks that the computed Hessian matches finite-difference approximation:
     * H_ij ≈ (f(x + e_i + e_j) - f(x + e_i) - f(x + e_j) + f(x)) / h²
     * 
     * This is a validation method to ensure our autograd-based Hessian is correct.
     * 
     * @param loss Loss function
     * @param parameters Current parameters
     * @param step_size Finite difference step (default: 1e-5)
     * @return Maximum relative error between autograd and finite diff
     */
    static double verify_hessian_finite_diff(
        std::function<torch::Tensor(const std::vector<torch::Tensor>&)> loss_fn,
        const std::vector<torch::Tensor>& parameters,
        double step_size = 1e-5);

private:
    /**
     * @brief Flatten parameter list into a single vector
     */
    static torch::Tensor flatten_parameters(const std::vector<torch::Tensor>& params);
    
    /**
     * @brief Compute gradient as flattened vector
     */
    static torch::Tensor compute_gradient(
        torch::Tensor loss,
        const std::vector<torch::Tensor>& parameters);
};

/**
 * @brief Compositional curvature bound verification
 * 
 * Implements and validates Theorem 3.1 (Composition Law) and Lemma 4.2
 * (Compositional Curvature Bound) from the HNF paper.
 * 
 * From HNF paper Lemma 4.2:
 * For morphisms f: A → B and g: B → C with curvatures κ_f, κ_g and
 * Lipschitz constants L_f, L_g:
 * 
 * κ_{g∘f}^{curv} ≤ κ_g · L_f² + L_g · κ_f
 * 
 * This class provides tools to:
 * 1. Measure actual compositional curvature
 * 2. Verify it satisfies the theoretical bound
 * 3. Identify when bounds are tight vs loose
 */
class CompositionalCurvatureValidator {
public:
    struct CompositionMetrics {
        // Individual layer metrics
        double kappa_f;  // Curvature of first layer
        double kappa_g;  // Curvature of second layer
        double L_f;      // Lipschitz constant of f
        double L_g;      // Lipschitz constant of g
        
        // Composition metrics
        double kappa_composed_actual;     // Measured κ_{g∘f}
        double kappa_composed_bound;      // κ_g·L_f² + L_g·κ_f
        double bound_tightness;           // actual / bound
        
        // Validation
        bool bound_satisfied;             // actual ≤ bound
        double bound_slack;               // bound - actual
        
        std::string to_string() const;
    };
    
    /**
     * @brief Validate compositional curvature bound for two sequential layers
     * 
     * Given a network with layers f and g, computes:
     * 1. Individual curvatures κ_f, κ_g
     * 2. Lipschitz constants L_f, L_g
     * 3. Composed curvature κ_{g∘f}
     * 4. Theoretical bound κ_g·L_f² + L_g·κ_f
     * 5. Verifies bound holds
     * 
     * @param loss_f Loss through first layer only
     * @param loss_gf Loss through both layers
     * @param params_f Parameters of first layer
     * @param params_g Parameters of second layer
     * @param input_samples Sample inputs for Lipschitz estimation
     * @return Composition metrics
     */
    static CompositionMetrics validate_composition(
        std::function<torch::Tensor(torch::Tensor)> layer_f,
        std::function<torch::Tensor(torch::Tensor)> layer_g,
        std::function<torch::Tensor(torch::Tensor)> loss_fn,
        torch::Tensor input,
        const std::vector<torch::Tensor>& params_f,
        const std::vector<torch::Tensor>& params_g);
    
    /**
     * @brief Estimate Lipschitz constant of a layer
     * 
     * Uses empirical sampling: L_f ≈ max_{x,y} ||f(x) - f(y)|| / ||x - y||
     * 
     * For neural networks with weight matrix W, the spectral norm ||W||_op
     * provides an exact Lipschitz constant.
     * 
     * @param layer Layer function
     * @param input_samples Random samples from input domain
     * @param use_spectral_norm If true and layer is linear, use exact spectral norm
     * @return Estimated Lipschitz constant
     */
    static double estimate_lipschitz_constant(
        std::function<torch::Tensor(torch::Tensor)> layer,
        const std::vector<torch::Tensor>& input_samples,
        bool use_spectral_norm = true);
    
    /**
     * @brief Validate compositional bounds for entire network
     * 
     * For a network with layers f_1, ..., f_n, validates that:
     * κ_{f_n ∘ ... ∘ f_1} ≤ sum of compositional bounds
     * 
     * @param layers Individual layer functions
     * @param loss_fn Loss function
     * @param input Input tensor
     * @param all_params Parameters for all layers
     * @return Metrics for each composition
     */
    static std::vector<CompositionMetrics> validate_deep_composition(
        const std::vector<std::function<torch::Tensor(torch::Tensor)>>& layers,
        std::function<torch::Tensor(torch::Tensor)> loss_fn,
        torch::Tensor input,
        const std::vector<std::vector<torch::Tensor>>& all_params);
};

} // namespace profiler
} // namespace hnf
