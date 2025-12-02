#pragma once

#include "graph_ir.hpp"
#include <cmath>
#include <limits>
#include <algorithm>

namespace hnf {
namespace rewriter {

// Curvature computation based on HNF paper Section 5.3
class CurvatureAnalyzer {
public:
    // Compute curvature bound for a single node given input statistics
    // Implements the formulas from Theorem 5.7 and Gallery examples
    static double compute_node_curvature(const Node& node, 
                                        const std::unordered_map<std::string, TensorStats>& stats) {
        
        switch (node.op) {
            case OpType::EXP: {
                // κ_exp = e^(2x_max) from Gallery Example 6
                if (node.inputs.empty()) return 0.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double x_max = it->second.max_val;
                // Cap at reasonable value to avoid overflow
                if (x_max > 100) return std::exp(200.0);
                return std::exp(2.0 * x_max);
            }
            
            case OpType::LOG: {
                // κ_log = 1/(2x_min^2) from Example 5.21 in paper
                if (node.inputs.empty()) return 0.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double x_min = std::max(1e-10, std::abs(it->second.min_val));
                return 1.0 / (2.0 * x_min * x_min);
            }
            
            case OpType::DIV: {
                // κ_div = 2/|d|^3 where d is denominator
                if (node.inputs.size() < 2) return 0.0;
                auto it = stats.find(node.inputs[1]);
                if (it == stats.end()) return 1.0;
                double denom_min = std::max(1e-10, std::abs(it->second.min_val));
                return 2.0 / (denom_min * denom_min * denom_min);
            }
            
            case OpType::SQRT: {
                // κ_sqrt = 1/(4x^(3/2)) from Example 5.21
                if (node.inputs.empty()) return 0.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double x_min = std::max(1e-10, it->second.min_val);
                return 1.0 / (4.0 * std::pow(x_min, 1.5));
            }
            
            case OpType::POW: {
                // For x^n, κ ≈ n(n-1)/2 * x^(n-2) max value
                if (node.inputs.empty()) return 0.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double exp = node.attrs.get_float("exponent", 2.0);
                double x_max = std::max(1.0, std::abs(it->second.max_val));
                if (std::abs(exp) < 2.0) return 0.0;
                return std::abs(exp * (exp - 1.0) / 2.0) * std::pow(x_max, std::abs(exp) - 2.0);
            }
            
            case OpType::SIGMOID: {
                // κ_sigmoid = 1/2 (from Section 5.3)
                return 0.5;
            }
            
            case OpType::TANH: {
                // κ_tanh = 1 
                return 1.0;
            }
            
            case OpType::MATMUL: {
                // κ_matmul = cond(A) * cond(B) from Proposition 5.20
                double cond_A = 1.0, cond_B = 1.0;
                if (node.inputs.size() >= 1) {
                    auto it = stats.find(node.inputs[0]);
                    if (it != stats.end()) cond_A = it->second.condition_number;
                }
                if (node.inputs.size() >= 2) {
                    auto it = stats.find(node.inputs[1]);
                    if (it != stats.end()) cond_B = it->second.condition_number;
                }
                return cond_A * cond_B;
            }
            
            case OpType::SOFTMAX: {
                // Naive softmax has κ = e^(2 * range(logits))
                // This is from Gallery Example 4
                if (node.inputs.empty()) return 0.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double range = it->second.range();
                if (range > 100) return std::exp(200.0);
                return std::exp(2.0 * range);
            }
            
            case OpType::STABLE_SOFTMAX: {
                // Stable softmax (with max subtraction) has κ = O(1)
                // This is proven in Gallery Example 4
                return 1.0;
            }
            
            case OpType::LOGSUMEXP: {
                // Naive logsumexp has exponential curvature
                if (node.inputs.empty()) return 0.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double x_max = it->second.max_val;
                if (x_max > 100) return std::exp(200.0);
                return std::exp(2.0 * x_max);
            }
            
            // Linear operations have zero curvature
            case OpType::ADD:
            case OpType::SUB:
            case OpType::MUL:
            case OpType::NEG:
            case OpType::TRANSPOSE:
            case OpType::SUM:
            case OpType::MAX:
            case OpType::MIN:
            case OpType::MEAN:
            case OpType::RELU:
            case OpType::CONSTANT:
            case OpType::INPUT:
            case OpType::OUTPUT:
            case OpType::IDENTITY:
                return 0.0;
            
            case OpType::LOG_SOFTMAX: {
                // log_softmax is numerically stable, κ = O(1)
                return 1.0;
            }
            
            default:
                return 1.0;
        }
    }
    
    // Compute Lipschitz constant for a node
    static double compute_lipschitz(const Node& node,
                                   const std::unordered_map<std::string, TensorStats>& stats) {
        
        switch (node.op) {
            case OpType::ADD:
            case OpType::SUB:
            case OpType::IDENTITY:
                return 1.0;
            
            case OpType::MUL: {
                // L_mul = max(|a|, |b|)
                double max_val = 1.0;
                for (const auto& input_id : node.inputs) {
                    auto it = stats.find(input_id);
                    if (it != stats.end()) {
                        max_val = std::max(max_val, std::max(std::abs(it->second.min_val),
                                                             std::abs(it->second.max_val)));
                    }
                }
                return max_val;
            }
            
            case OpType::DIV: {
                // L_div = 1/|d_min|
                if (node.inputs.size() < 2) return 1.0;
                auto it = stats.find(node.inputs[1]);
                if (it == stats.end()) return 1.0;
                double denom_min = std::max(1e-10, std::abs(it->second.min_val));
                return 1.0 / denom_min;
            }
            
            case OpType::EXP: {
                // L_exp = e^(x_max)
                if (node.inputs.empty()) return 1.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double x_max = it->second.max_val;
                if (x_max > 100) return std::exp(100.0);
                return std::exp(x_max);
            }
            
            case OpType::LOG: {
                // L_log = 1/x_min
                if (node.inputs.empty()) return 1.0;
                auto it = stats.find(node.inputs[0]);
                if (it == stats.end()) return 1.0;
                double x_min = std::max(1e-10, it->second.min_val);
                return 1.0 / x_min;
            }
            
            case OpType::MATMUL: {
                // L_matmul = ||A|| * ||B|| (operator norms)
                double norm_A = 1.0, norm_B = 1.0;
                if (node.inputs.size() >= 1) {
                    auto it = stats.find(node.inputs[0]);
                    if (it != stats.end()) {
                        norm_A = std::max(std::abs(it->second.min_val), 
                                        std::abs(it->second.max_val));
                    }
                }
                if (node.inputs.size() >= 2) {
                    auto it = stats.find(node.inputs[1]);
                    if (it != stats.end()) {
                        norm_B = std::max(std::abs(it->second.min_val),
                                        std::abs(it->second.max_val));
                    }
                }
                return norm_A * norm_B;
            }
            
            case OpType::SIGMOID:
            case OpType::TANH:
            case OpType::RELU:
                return 1.0;  // All have Lipschitz constant 1
            
            case OpType::SOFTMAX:
            case OpType::STABLE_SOFTMAX:
            case OpType::LOG_SOFTMAX:
                return 1.0;
            
            default:
                return 1.0;
        }
    }
    
    // Propagate statistics forward through the graph
    static std::unordered_map<std::string, TensorStats> propagate_stats(
        const Graph& graph,
        const std::unordered_map<std::string, TensorStats>& input_stats) {
        
        std::unordered_map<std::string, TensorStats> result = input_stats;
        
        auto topo = graph.topological_order();
        for (const auto& node_id : topo) {
            auto node = graph.get_node(node_id);
            if (!node) continue;
            
            TensorStats output_stats;
            
            // Compute output statistics based on operation
            if (node->inputs.empty()) {
                // Leaf node (input or constant)
                if (input_stats.count(node_id)) {
                    output_stats = input_stats.at(node_id);
                }
            } else {
                // Get input statistics
                std::vector<TensorStats> inp_stats;
                for (const auto& inp_id : node->inputs) {
                    if (result.count(inp_id)) {
                        inp_stats.push_back(result.at(inp_id));
                    }
                }
                
                if (!inp_stats.empty()) {
                    output_stats = estimate_output_stats(*node, inp_stats);
                }
            }
            
            result[node_id] = output_stats;
        }
        
        return result;
    }
    
    // Estimate output statistics from operation and input statistics
    static TensorStats estimate_output_stats(const Node& node, 
                                            const std::vector<TensorStats>& input_stats) {
        TensorStats output;
        
        if (input_stats.empty()) return output;
        
        const auto& in0 = input_stats[0];
        
        switch (node.op) {
            case OpType::ADD:
                if (input_stats.size() >= 2) {
                    const auto& in1 = input_stats[1];
                    output.min_val = in0.min_val + in1.min_val;
                    output.max_val = in0.max_val + in1.max_val;
                    output.mean_val = in0.mean_val + in1.mean_val;
                    output.std_val = std::sqrt(in0.std_val * in0.std_val + 
                                              in1.std_val * in1.std_val);
                }
                break;
            
            case OpType::MUL:
                if (input_stats.size() >= 2) {
                    const auto& in1 = input_stats[1];
                    double vals[] = {in0.min_val * in1.min_val, in0.min_val * in1.max_val,
                                    in0.max_val * in1.min_val, in0.max_val * in1.max_val};
                    output.min_val = *std::min_element(vals, vals + 4);
                    output.max_val = *std::max_element(vals, vals + 4);
                    output.mean_val = in0.mean_val * in1.mean_val;
                }
                break;
            
            case OpType::EXP:
                output.min_val = std::exp(in0.min_val);
                output.max_val = std::exp(in0.max_val);
                output.mean_val = std::exp(in0.mean_val);
                break;
            
            case OpType::LOG:
                output.min_val = std::log(std::max(1e-10, in0.min_val));
                output.max_val = std::log(std::max(1e-10, in0.max_val));
                output.mean_val = std::log(std::max(1e-10, in0.mean_val));
                break;
            
            case OpType::MAX:
                output.min_val = in0.min_val;
                output.max_val = in0.max_val;
                output.mean_val = in0.max_val;  // Approximate
                break;
            
            case OpType::SOFTMAX:
            case OpType::STABLE_SOFTMAX:
                output.min_val = 0.0;
                output.max_val = 1.0;
                output.mean_val = 0.5;
                break;
            
            default:
                output = in0;  // Default: propagate input stats
                break;
        }
        
        output.condition_number = in0.condition_number;
        return output;
    }
    
    // Compute total curvature of entire graph
    static double total_curvature(const Graph& graph,
                                 const std::unordered_map<std::string, TensorStats>& input_stats) {
        auto stats = propagate_stats(graph, input_stats);
        
        double total = 0.0;
        for (const auto& [id, node] : graph.nodes()) {
            double curv = compute_node_curvature(*node, stats);
            total += curv;
        }
        
        return total;
    }
};

} // namespace rewriter
} // namespace hnf
