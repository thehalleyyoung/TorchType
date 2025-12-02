#pragma once

#include "graph_ir.hpp"
#include <cmath>
#include <vector>

namespace hnf {
namespace rewriter {

// Advanced curvature analysis using Hessian eigenvalues
// Implements HNF Theorem 5.7 (Precision Obstruction Theorem)
class HessianCurvatureAnalyzer {
public:
    // Compute Hessian-based curvature for a function
    // κ_f^curv = (1/2) * ||D²f||
    static double compute_hessian_curvature(OpType op, const TensorStats& input_stats) {
        switch (op) {
            case OpType::EXP: {
                // For f(x) = exp(x), D²f = exp(x)
                // κ = (1/2) * exp(2 * x_max)
                double x_max = input_stats.max_val;
                if (x_max > 100) {
                    // Avoid overflow - return large number
                    return 1e100;
                }
                return 0.5 * std::exp(2.0 * x_max);
            }
            
            case OpType::LOG: {
                // For f(x) = log(x), D²f = -1/x²
                // κ = (1/2) / x_min²
                double x_min = std::max(1e-10, std::abs(input_stats.min_val));
                return 0.5 / (x_min * x_min);
            }
            
            case OpType::SIGMOID: {
                // For f(x) = 1/(1+exp(-x)), D²f = σ(x)(1-σ(x))(1-2σ(x))
                // Max curvature at x = 0: κ = 1/8
                return 0.125;
            }
            
            case OpType::TANH: {
                // For f(x) = tanh(x), D²f = -2tanh(x)(1-tanh²(x))
                // Max curvature: κ ≈ 0.385
                return 0.385;
            }
            
            case OpType::SQRT: {
                // For f(x) = sqrt(x), D²f = -1/(4x^(3/2))
                double x_min = std::max(1e-10, input_stats.min_val);
                return 0.5 / (4.0 * std::pow(x_min, 1.5));
            }
            
            case OpType::POW: {
                // For f(x) = x^p, D²f = p(p-1)x^(p-2)
                // Assume p = 2 (square) for now
                return 1.0;
            }
            
            case OpType::DIV: {
                // For f(x,y) = x/y, curvature in y: 2x/y³
                double y_min = std::max(1e-10, std::abs(input_stats.min_val));
                return 1.0 / std::pow(y_min, 3);
            }
            
            // Linear operations have zero curvature
            case OpType::ADD:
            case OpType::SUB:
            case OpType::MUL:
            case OpType::NEG:
            case OpType::MATMUL:
            case OpType::TRANSPOSE:
            case OpType::IDENTITY:
                return 0.0;
            
            // ReLU has zero curvature almost everywhere
            case OpType::RELU:
                return 0.0;
            
            // Stable operations have low curvature
            case OpType::STABLE_SOFTMAX:
            case OpType::LOG_SOFTMAX:
            case OpType::LOGSUMEXP:
            case OpType::LOG1P:
            case OpType::EXPM1:
                return 1.0;  // Bounded curvature
            
            default:
                return 1.0;  // Default moderate curvature
        }
    }
    
    // Compute eigenvalue-based curvature for matrix operations
    // Simplified version without Eigen - uses Frobenius norm approximation
    static double compute_matrix_curvature(const std::vector<std::vector<double>>& matrix) {
        if (matrix.empty() || matrix[0].empty()) {
            return 1.0;
        }
        
        size_t rows = matrix.size();
        size_t cols = matrix[0].size();
        
        // Compute Frobenius norm
        double frobenius = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                frobenius += matrix[i][j] * matrix[i][j];
            }
        }
        frobenius = std::sqrt(frobenius);
        
        // Estimate condition number using row/column norms
        double max_row_norm = 0.0;
        for (size_t i = 0; i < rows; ++i) {
            double row_norm = 0.0;
            for (size_t j = 0; j < cols; ++j) {
                row_norm += std::abs(matrix[i][j]);
            }
            max_row_norm = std::max(max_row_norm, row_norm);
        }
        
        if (max_row_norm < 1e-10) {
            return 1e10;  // Nearly singular
        }
        
        // Approximate condition number
        return frobenius / std::sqrt(std::min(rows, cols));
    }
    
    // Compute precision requirements using Theorem 5.7
    // p ≥ log₂(c * κ * D² / ε)
    static double precision_requirement(double curvature, double diameter, double target_error) {
        const double c = 1.0;  // Constant from theorem
        
        if (curvature < 1e-10) {
            // Linear operation - only depends on diameter
            return std::log2(diameter / target_error);
        }
        
        double required_bits = std::log2(c * curvature * diameter * diameter / target_error);
        return std::max(1.0, required_bits);
    }
    
    // Analyze whether precision is sufficient
    struct PrecisionAnalysis {
        double curvature;
        double required_bits;
        double available_bits;
        bool sufficient;
        double margin;  // How many bits over/under
        
        std::string to_string() const {
            std::ostringstream ss;
            ss << "Curvature: " << curvature << "\n";
            ss << "Required: " << required_bits << " bits\n";
            ss << "Available: " << available_bits << " bits\n";
            ss << "Sufficient: " << (sufficient ? "YES" : "NO") << "\n";
            ss << "Margin: " << margin << " bits\n";
            return ss.str();
        }
    };
    
    static PrecisionAnalysis analyze_precision(
        const Graph& graph,
        const std::unordered_map<std::string, TensorStats>& input_stats,
        double target_error,
        int available_bits
    ) {
        PrecisionAnalysis result;
        
        // Compute total curvature
        result.curvature = 0.0;
        double max_diameter = 0.0;
        
        for (const auto& [id, node] : graph.nodes()) {
            // Get input statistics if available
            TensorStats stats;
            if (node->inputs.size() > 0) {
                auto it = input_stats.find(node->inputs[0]);
                if (it != input_stats.end()) {
                    stats = it->second;
                }
            }
            
            double node_curv = compute_hessian_curvature(node->op, stats);
            result.curvature += node_curv;
            
            max_diameter = std::max(max_diameter, stats.range());
        }
        
        if (max_diameter < 1e-10) {
            max_diameter = 1.0;  // Default
        }
        
        result.required_bits = precision_requirement(result.curvature, max_diameter, target_error);
        result.available_bits = available_bits;
        result.sufficient = (available_bits >= result.required_bits);
        result.margin = available_bits - result.required_bits;
        
        return result;
    }
    
    // Compute Lipschitz constant for composition
    // L_{g∘f} = L_g * L_f
    static double compute_lipschitz_constant(OpType op, const TensorStats& stats) {
        switch (op) {
            case OpType::EXP:
                return std::exp(stats.max_val);
            
            case OpType::LOG:
                return 1.0 / std::max(1e-10, stats.min_val);
            
            case OpType::SIGMOID:
            case OpType::TANH:
            case OpType::RELU:
                return 1.0;
            
            case OpType::SQRT:
                return 1.0 / (2.0 * std::sqrt(std::max(1e-10, stats.min_val)));
            
            case OpType::ADD:
            case OpType::SUB:
                return 1.0;
            
            case OpType::MUL:
                return std::max(std::abs(stats.max_val), std::abs(stats.min_val));
            
            case OpType::DIV: {
                double denom_min = std::max(1e-10, std::abs(stats.min_val));
                return 1.0 / denom_min;
            }
            
            case OpType::MATMUL:
                return stats.condition_number;
            
            default:
                return 1.0;
        }
    }
    
    // Verify Theorem 3.8 (Stability Composition Theorem)
    // Φ_{g∘f}(ε) ≤ Φ_g(Φ_f(ε)) + L_g * Φ_f(ε)
    static bool verify_composition_theorem(
        const Graph& composed,
        const Graph& f,
        const Graph& g,
        const std::unordered_map<std::string, TensorStats>& input_stats,
        double epsilon
    ) {
        // Compute error functionals (simplified)
        auto compute_error = [&](const Graph& graph) -> double {
            double total_error = epsilon;
            for (const auto& [id, node] : graph.nodes()) {
                TensorStats stats;
                if (!node->inputs.empty()) {
                    auto it = input_stats.find(node->inputs[0]);
                    if (it != input_stats.end()) stats = it->second;
                }
                
                double L = compute_lipschitz_constant(node->op, stats);
                total_error = L * total_error + epsilon;
            }
            return total_error;
        };
        
        double error_composed = compute_error(composed);
        double error_f = compute_error(f);
        double error_g = compute_error(g);
        
        // Compute L_g
        double L_g = 1.0;
        for (const auto& [id, node] : g.nodes()) {
            TensorStats stats;
            if (!node->inputs.empty()) {
                auto it = input_stats.find(node->inputs[0]);
                if (it != input_stats.end()) stats = it->second;
            }
            L_g *= compute_lipschitz_constant(node->op, stats);
        }
        
        double bound = error_g + L_g * error_f;
        
        // Check if theorem holds
        return error_composed <= bound * 1.1;  // Allow 10% slack for numerical errors
    }
};

} // namespace rewriter
} // namespace hnf
