#pragma once

#include "graph_ir.hpp"
#include "curvature.hpp"
#include <vector>
#include <unordered_map>
#include <cmath>
#include <algorithm>

namespace hnf {
namespace rewriter {

// Gradient stability analyzer
// Tracks numerical stability of backpropagation through computation graphs
class GradientStabilityAnalyzer {
public:
    struct GradientStats {
        double magnitude;
        double variance;
        bool exploding;  // gradient > 100
        bool vanishing;  // gradient < 1e-6
        double condition_number;
        
        GradientStats() : magnitude(0), variance(0), exploding(false), 
                         vanishing(false), condition_number(1.0) {}
    };
    
    // Compute gradient stability for a forward graph
    static std::unordered_map<std::string, GradientStats> analyze_gradients(
        const Graph& forward_graph,
        const std::unordered_map<std::string, TensorStats>& forward_stats
    ) {
        std::unordered_map<std::string, GradientStats> gradient_stats;
        
        // Backward pass in reverse topological order
        auto topo_order = forward_graph.topological_order();
        std::reverse(topo_order.begin(), topo_order.end());
        
        // Initialize output gradients to 1
        for (const auto& out_id : forward_graph.outputs()) {
            GradientStats& stats = gradient_stats[out_id];
            stats.magnitude = 1.0;
            stats.variance = 0.0;
            stats.condition_number = 1.0;
        }
        
        // Backpropagate
        for (const auto& node_id : topo_order) {
            auto node = forward_graph.get_node(node_id);
            if (!node) continue;
            
            // Get gradient at this node
            GradientStats& grad_stats = gradient_stats[node_id];
            
            // Compute gradient contribution from this operation
            double local_grad = compute_local_gradient(
                node->op, 
                forward_stats.count(node_id) ? forward_stats.at(node_id) : TensorStats()
            );
            
            // Propagate to inputs
            for (const auto& input_id : node->inputs) {
                GradientStats& input_grad = gradient_stats[input_id];
                
                // Chain rule: ∂L/∂input = ∂L/∂output * ∂output/∂input
                input_grad.magnitude += grad_stats.magnitude * local_grad;
                input_grad.variance += grad_stats.variance + local_grad * local_grad * grad_stats.variance;
                input_grad.condition_number = std::max(
                    input_grad.condition_number,
                    grad_stats.condition_number * local_grad
                );
            }
        }
        
        // Classify gradients
        for (auto& [id, stats] : gradient_stats) {
            stats.exploding = (stats.magnitude > 100.0);
            stats.vanishing = (stats.magnitude < 1e-6);
        }
        
        return gradient_stats;
    }
    
    // Compute local gradient (derivative) for an operation
    static double compute_local_gradient(OpType op, const TensorStats& stats) {
        switch (op) {
            case OpType::EXP: {
                // d/dx exp(x) = exp(x)
                double x_max = stats.max_val;
                if (x_max > 50) return 1e20;  // Very large
                return std::exp(x_max);
            }
            
            case OpType::LOG: {
                // d/dx log(x) = 1/x
                double x_min = std::max(1e-10, stats.min_val);
                return 1.0 / x_min;
            }
            
            case OpType::SIGMOID: {
                // d/dx σ(x) = σ(x)(1-σ(x)) ≤ 0.25
                return 0.25;
            }
            
            case OpType::TANH: {
                // d/dx tanh(x) = 1 - tanh²(x) ≤ 1
                return 1.0;
            }
            
            case OpType::RELU: {
                // d/dx ReLU(x) = 1 if x > 0, else 0
                return 0.5;  // Average over distribution
            }
            
            case OpType::SQRT: {
                // d/dx sqrt(x) = 1/(2*sqrt(x))
                double x_mean = std::max(1e-10, stats.mean_val);
                return 1.0 / (2.0 * std::sqrt(x_mean));
            }
            
            case OpType::POW: {
                // d/dx x^p = p * x^(p-1)
                // Assume p=2 (square)
                return 2.0 * stats.mean_val;
            }
            
            case OpType::ADD:
            case OpType::SUB:
                return 1.0;
            
            case OpType::MUL:
                return stats.mean_val;
            
            case OpType::DIV: {
                double denom = std::max(1e-10, std::abs(stats.mean_val));
                return 1.0 / denom;
            }
            
            case OpType::NEG:
                return -1.0;
            
            case OpType::MATMUL:
                return stats.condition_number;
            
            // Stable operations have well-behaved gradients
            case OpType::STABLE_SOFTMAX:
            case OpType::LOG_SOFTMAX:
            case OpType::LOGSUMEXP:
                return 1.0;
            
            default:
                return 1.0;
        }
    }
    
    // Analyze gradient flow through a deep network
    struct NetworkGradientAnalysis {
        std::vector<double> layer_gradients;  // Gradient magnitude at each layer
        double total_gradient_scale;  // Product of all gradients
        bool has_gradient_explosion;
        bool has_gradient_vanishing;
        int problematic_layer;  // -1 if no problem
        double worst_condition_number;
        
        std::string to_string() const {
            std::ostringstream ss;
            ss << "Network Gradient Analysis:\n";
            ss << "  Total gradient scale: " << total_gradient_scale << "\n";
            ss << "  Gradient explosion: " << (has_gradient_explosion ? "YES" : "NO") << "\n";
            ss << "  Gradient vanishing: " << (has_gradient_vanishing ? "YES" : "NO") << "\n";
            if (problematic_layer >= 0) {
                ss << "  Problematic layer: " << problematic_layer << "\n";
            }
            ss << "  Worst condition number: " << worst_condition_number << "\n";
            ss << "  Layer gradients: [";
            for (size_t i = 0; i < layer_gradients.size(); ++i) {
                if (i > 0) ss << ", ";
                ss << layer_gradients[i];
            }
            ss << "]\n";
            return ss.str();
        }
    };
    
    static NetworkGradientAnalysis analyze_network_gradients(
        const Graph& network,
        const std::unordered_map<std::string, TensorStats>& stats,
        const std::vector<std::string>& layer_outputs
    ) {
        NetworkGradientAnalysis analysis;
        analysis.total_gradient_scale = 1.0;
        analysis.has_gradient_explosion = false;
        analysis.has_gradient_vanishing = false;
        analysis.problematic_layer = -1;
        analysis.worst_condition_number = 1.0;
        
        // Compute gradients
        auto grad_stats = analyze_gradients(network, stats);
        
        // Extract layer-wise gradients
        for (size_t i = 0; i < layer_outputs.size(); ++i) {
            const std::string& layer_id = layer_outputs[i];
            if (grad_stats.count(layer_id)) {
                const auto& stats = grad_stats.at(layer_id);
                analysis.layer_gradients.push_back(stats.magnitude);
                analysis.total_gradient_scale *= stats.magnitude;
                
                if (stats.exploding && analysis.problematic_layer < 0) {
                    analysis.has_gradient_explosion = true;
                    analysis.problematic_layer = i;
                }
                if (stats.vanishing && analysis.problematic_layer < 0) {
                    analysis.has_gradient_vanishing = true;
                    analysis.problematic_layer = i;
                }
                
                analysis.worst_condition_number = std::max(
                    analysis.worst_condition_number,
                    stats.condition_number
                );
            }
        }
        
        return analysis;
    }
    
    // Compute stable gradient alternatives
    // Returns map from node ID to suggested stable operation
    static std::unordered_map<std::string, OpType> suggest_stable_gradients(
        const Graph& network,
        const std::unordered_map<std::string, GradientStats>& grad_stats
    ) {
        std::unordered_map<std::string, OpType> suggestions;
        
        for (const auto& [node_id, stats] : grad_stats) {
            auto node = network.get_node(node_id);
            if (!node) continue;
            
            // Suggest stable alternatives for problematic operations
            if (stats.exploding || stats.vanishing) {
                switch (node->op) {
                    case OpType::EXP:
                        // Use EXPM1 for small values
                        suggestions[node_id] = OpType::EXPM1;
                        break;
                    
                    case OpType::LOG:
                        // Use LOG1P for values near 1
                        suggestions[node_id] = OpType::LOG1P;
                        break;
                    
                    case OpType::DIV:
                        // Use softmax for normalization
                        if (stats.exploding) {
                            suggestions[node_id] = OpType::STABLE_SOFTMAX;
                        }
                        break;
                    
                    case OpType::MATMUL:
                        // Suggest batch normalization or layer normalization
                        if (stats.exploding) {
                            suggestions[node_id] = OpType::LAYER_NORM;
                        }
                        break;
                    
                    default:
                        break;
                }
            }
        }
        
        return suggestions;
    }
    
    // Compute gradient curvature (second-order information)
    // Important for optimization algorithms like Newton's method
    static double compute_gradient_curvature(
        OpType op,
        const TensorStats& forward_stats
    ) {
        switch (op) {
            case OpType::EXP: {
                // D²(exp(x)) = exp(x)
                return std::exp(forward_stats.max_val);
            }
            
            case OpType::LOG: {
                // D²(log(x)) = -1/x²
                double x_min = std::max(1e-10, forward_stats.min_val);
                return 1.0 / (x_min * x_min);
            }
            
            case OpType::SIGMOID: {
                // D²(σ(x)) has max value 1/8 at x=0
                return 0.125;
            }
            
            case OpType::TANH: {
                // D²(tanh(x))
                return 0.5;
            }
            
            case OpType::RELU: {
                // ReLU has zero second derivative except at 0
                return 0.0;
            }
            
            default:
                return 0.0;
        }
    }
};

} // namespace rewriter
} // namespace hnf
