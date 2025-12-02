#pragma once

#include "kv_cache_types.hpp"
#include <torch/torch.h>
#include <vector>
#include <optional>

namespace hnf {
namespace kv_cache {

/**
 * Formal verification of HNF Theorem 5.7 precision bounds
 * 
 * Uses interval arithmetic and symbolic methods to verify that:
 * 1. The computed precision bounds are necessary (lower bound)
 * 2. The implemented precision is sufficient (upper bound)
 * 3. Error accumulation is correctly tracked
 */
class HNFTheoremVerifier {
public:
    struct VerificationResult {
        bool is_valid;
        double computed_curvature;
        int required_precision_bits;
        int assigned_precision_bits;
        double theoretical_error_bound;
        double empirical_error_observed;
        std::string failure_reason;
        
        // Detailed breakdown
        struct {
            double attention_component;
            double gradient_component;
            double hessian_component;
        } curvature_breakdown;
    };
    
    /**
     * Verify that assigned precision meets HNF Theorem 5.7 requirements
     * 
     * Theorem 5.7 states: p >= log_2(c * κ * D^2 / ε)
     * 
     * @param curvature: Computed κ_f value
     * @param diameter: Domain diameter D
     * @param target_epsilon: Desired accuracy ε
     * @param assigned_precision: Precision level actually assigned (16, 8, or 4 bits)
     * @param c_constant: The explicit constant c (default ~4.0 from paper)
     * @return: Detailed verification result
     */
    static VerificationResult verify_precision_assignment(
        double curvature,
        double diameter,
        double target_epsilon,
        int assigned_precision,
        double c_constant = 4.0
    );
    
    /**
     * Verify precision for an entire position curvature map
     */
    static std::vector<VerificationResult> verify_precision_map(
        const std::vector<PositionCurvature>& curvatures,
        const std::vector<PrecisionLevel>& precisions,
        double diameter,
        double target_epsilon
    );
    
    /**
     * Compute theoretical error bound for given precision and curvature
     * This gives the GUARANTEED maximum error from the theorem
     */
    static double compute_theoretical_error_bound(
        double curvature,
        double diameter,
        int precision_bits
    );
    
    /**
     * Empirically measure actual error when using given precision
     * Compares full precision computation with reduced precision
     */
    static double measure_empirical_error(
        const torch::Tensor& full_precision_tensor,
        const torch::Tensor& reduced_precision_tensor
    );
    
    /**
     * Check if curvature computation is monotonically increasing
     * as attention/gradients increase (sanity check)
     */
    static bool verify_curvature_monotonicity(
        const std::vector<PositionCurvature>& curvatures
    );
    
    /**
     * Verify composition law: error functionals compose correctly
     * Φ_{g ∘ f}(ε) ≤ Φ_g(Φ_f(ε)) + L_g · Φ_f(ε)
     */
    static bool verify_composition_law(
        double epsilon_in,
        double phi_f,
        double phi_g,
        double lipschitz_g,
        double phi_composed
    );
    
    /**
     * Compute the "sharpness" of the bound - how close we are to the
     * theoretical minimum precision
     */
    static double compute_bound_sharpness(
        double curvature,
        double diameter,
        double target_epsilon,
        int assigned_precision
    );
    
    // Interval arithmetic for conservative bounds
    struct Interval {
        double lower;
        double upper;
        
        Interval operator*(const Interval& other) const;
        Interval operator+(const Interval& other) const;
        Interval log2() const;
    };
    
    static Interval compute_curvature_interval(
        const PositionCurvature& curv
    );
    
private:
};

/**
 * Formal correctness checker using SMT solving (if Z3 available)
 * This provides *proven* correctness guarantees
 */
class FormalCorrectnessChecker {
public:
    /**
     * Generate SMT formula encoding Theorem 5.7 and check satisfiability
     * Returns true if the assignment is provably correct
     */
    static bool check_precision_correctness_smt(
        const std::vector<PositionCurvature>& curvatures,
        const std::vector<PrecisionLevel>& precisions,
        double target_epsilon
    );
    
    /**
     * Generate counter-example if precision is insufficient
     * Returns input that violates error bound, if any exists
     */
    static std::optional<torch::Tensor> find_counterexample(
        const PositionCurvature& curv,
        PrecisionLevel precision,
        double target_epsilon
    );
};

} // namespace kv_cache
} // namespace hnf
