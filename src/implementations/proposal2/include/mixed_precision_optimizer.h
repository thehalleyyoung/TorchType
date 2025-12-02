#pragma once

#include "computation_graph.h"
#include "precision_sheaf.h"
#include <algorithm>
#include <queue>
#include <limits>

namespace hnf {
namespace sheaf {

// Result of precision optimization
struct OptimizationResult {
    PrecisionAssignment optimal_assignment;
    std::vector<Cocycle> obstructions;
    int h0_dimension;  // Number of global sections
    bool success;
    double estimated_memory_saving;  // Compared to float32 everywhere
    std::string status_message;
    
    // Per-layer breakdown
    std::unordered_map<std::string, std::string> precision_rationale;
};

// Mixed-Precision Optimizer using Sheaf Cohomology
// Implements Algorithm from Proposal #2
class MixedPrecisionOptimizer {
private:
    ComputationGraph& graph;
    double target_accuracy;
    int max_iterations;
    int min_precision;
    int max_precision;
    
public:
    MixedPrecisionOptimizer(
        ComputationGraph& g,
        double eps = 1e-5,
        int max_iter = 100,
        int min_prec = 7,
        int max_prec = 52
    ) : graph(g), 
        target_accuracy(eps),
        max_iterations(max_iter),
        min_precision(min_prec),
        max_precision(max_prec)
    {}
    
    // Main optimization routine
    OptimizationResult optimize() {
        OptimizationResult result;
        result.success = false;
        
        // Step 1: Compute minimum precision for each node
        compute_node_min_precisions();
        
        // Step 2: Try to find global section at minimum precision
        int current_precision = min_precision;
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // Initialize all nodes to current precision
            PrecisionAssignment initial_assignment;
            for (const auto& [name, node] : graph.nodes) {
                initial_assignment[name] = current_precision;
            }
            
            // Build cover and sheaf
            auto cover = OpenCover::star_cover(graph);
            PrecisionSheaf sheaf(graph, target_accuracy, cover);
            
            // Check for global sections
            auto H0 = sheaf.compute_H0();
            result.h0_dimension = static_cast<int>(H0.size());
            
            if (!H0.empty()) {
                // Found valid assignment!
                result.optimal_assignment = H0[0];
                result.success = true;
                result.status_message = "Found optimal mixed-precision assignment";
                break;
            }
            
            // Compute obstruction
            auto obstruction = sheaf.get_obstruction();
            if (obstruction) {
                result.obstructions.push_back(*obstruction);
                
                // Increase precision where obstruction is nonzero
                resolve_obstruction(*obstruction, cover, initial_assignment);
            } else {
                // Increase global precision
                current_precision = get_next_precision(current_precision);
                
                if (current_precision > max_precision) {
                    result.status_message = "No feasible precision assignment found within precision bounds";
                    break;
                }
            }
        }
        
        // If we didn't find a solution, fall back to node-by-node assignment
        if (!result.success) {
            result.optimal_assignment = fallback_assignment();
            result.success = true;
            result.status_message = "Using fallback node-by-node assignment";
        }
        
        // Compute memory savings
        result.estimated_memory_saving = compute_memory_saving(result.optimal_assignment);
        
        // Generate rationale for each node's precision
        generate_rationale(result);
        
        return result;
    }
    
private:
    void compute_node_min_precisions() {
        for (auto& [name, node] : graph.nodes) {
            node->compute_min_precision(target_accuracy);
        }
    }
    
    void resolve_obstruction(
        const Cocycle& obstruction,
        const OpenCover& cover,
        PrecisionAssignment& assignment
    ) {
        // Find nodes where obstruction is largest
        std::vector<std::pair<std::string, int>> gaps;
        
        for (const auto& [edge_idx, gap] : obstruction.values) {
            if (gap == 0) continue;
            
            int i = edge_idx.first;
            int j = edge_idx.second;
            
            auto inter = OpenCover::intersection(cover.sets[i], cover.sets[j]);
            
            for (const auto& node_name : inter) {
                if (graph.nodes.count(node_name)) {
                    gaps.emplace_back(node_name, std::abs(gap));
                }
            }
        }
        
        // Sort by gap size
        std::sort(gaps.begin(), gaps.end(), 
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Increase precision at most sensitive nodes
        int nodes_to_upgrade = std::min(static_cast<int>(gaps.size()), 5);
        for (int i = 0; i < nodes_to_upgrade; ++i) {
            const auto& node_name = gaps[i].first;
            const auto& node = graph.nodes[node_name];
            
            // Increase to next standard precision
            int current = node->assigned_precision;
            int next = get_next_precision(current);
            node->assigned_precision = next;
            assignment[node_name] = next;
        }
    }
    
    int get_next_precision(int current) const {
        std::vector<int> standard_precisions = {7, 10, 16, 23, 32, 52, 64, 112};
        
        for (int p : standard_precisions) {
            if (p > current) return p;
        }
        
        return 112;  // max
    }
    
    PrecisionAssignment fallback_assignment() const {
        PrecisionAssignment assignment;
        
        // Assign each node its minimum required precision
        for (const auto& [name, node] : graph.nodes) {
            assignment[name] = node->get_hardware_precision();
        }
        
        return assignment;
    }
    
    double compute_memory_saving(const PrecisionAssignment& assignment) const {
        double total_fp32 = 0.0;
        double total_optimized = 0.0;
        
        for (const auto& [name, precision] : assignment) {
            if (!graph.nodes.count(name)) continue;
            
            const auto& node = graph.nodes.at(name);
            
            // Estimate memory based on tensor sizes
            int64_t num_elements = 1;
            for (auto dim : node->output_shape) {
                num_elements *= dim;
            }
            
            total_fp32 += num_elements * 4.0;  // 32 bits = 4 bytes
            total_optimized += num_elements * (precision / 8.0);
        }
        
        if (total_fp32 > 0) {
            return (total_fp32 - total_optimized) / total_fp32;
        }
        
        return 0.0;
    }
    
    void generate_rationale(OptimizationResult& result) {
        for (const auto& [name, precision] : result.optimal_assignment) {
            if (!graph.nodes.count(name)) continue;
            
            const auto& node = graph.nodes.at(name);
            
            std::string rationale;
            
            if (node->curvature > 100.0) {
                rationale = "High curvature (" + std::to_string(node->curvature) + 
                           ") requires high precision";
            } else if (node->op_type == "softmax") {
                rationale = "Softmax operation is numerically sensitive";
            } else if (node->op_type == "attention") {
                rationale = "Attention mechanism requires high precision for QK^T softmax";
            } else if (node->curvature < 0.1) {
                rationale = "Low curvature allows reduced precision";
            } else {
                rationale = "Standard precision for " + node->op_type;
            }
            
            result.precision_rationale[name] = rationale;
        }
    }
    
public:
    // Analyze a specific subgraph
    OptimizationResult analyze_subgraph(const std::unordered_set<std::string>& nodes) {
        auto sub = graph.subgraph(nodes);
        MixedPrecisionOptimizer sub_optimizer(sub, target_accuracy, max_iterations);
        return sub_optimizer.optimize();
    }
    
    // Compare with uniform precision baseline
    struct ComparisonResult {
        double accuracy_uniform_fp16;
        double accuracy_uniform_fp32;
        double accuracy_optimized;
        double memory_uniform_fp16;
        double memory_uniform_fp32;
        double memory_optimized;
    };
    
    ComparisonResult compare_with_baseline(const PrecisionAssignment& optimized) const {
        ComparisonResult result;
        
        // Estimate accuracy for each approach
        result.accuracy_uniform_fp16 = estimate_accuracy(10);
        result.accuracy_uniform_fp32 = estimate_accuracy(23);
        result.accuracy_optimized = estimate_accuracy_mixed(optimized);
        
        // Compute memory usage
        result.memory_uniform_fp16 = compute_memory_uniform(10);
        result.memory_uniform_fp32 = compute_memory_uniform(23);
        result.memory_optimized = compute_memory_mixed(optimized);
        
        return result;
    }
    
private:
    double estimate_accuracy(int uniform_precision) const {
        // Use error functional composition
        double total_error = 0.0;
        
        const auto& order = graph.topological_order();
        std::unordered_map<std::string, double> node_errors;
        
        double input_eps = 1e-10;
        
        for (const auto& name : order) {
            if (!graph.nodes.count(name)) continue;
            
            const auto& node = graph.nodes.at(name);
            
            // Accumulate input errors
            double incoming_error = input_eps;
            for (const auto& input : node->inputs) {
                if (node_errors.count(input)) {
                    incoming_error = std::max(incoming_error, node_errors[input]);
                }
            }
            
            // Apply error functional
            double error = node->error_functional(incoming_error, uniform_precision);
            node_errors[name] = error;
            total_error = std::max(total_error, error);
        }
        
        return total_error;
    }
    
    double estimate_accuracy_mixed(const PrecisionAssignment& assignment) const {
        double total_error = 0.0;
        
        const auto& order = graph.topological_order();
        std::unordered_map<std::string, double> node_errors;
        
        double input_eps = 1e-10;
        
        for (const auto& name : order) {
            if (!graph.nodes.count(name) || !assignment.count(name)) continue;
            
            const auto& node = graph.nodes.at(name);
            int precision = assignment.at(name);
            
            double incoming_error = input_eps;
            for (const auto& input : node->inputs) {
                if (node_errors.count(input)) {
                    incoming_error = std::max(incoming_error, node_errors[input]);
                }
            }
            
            double error = node->error_functional(incoming_error, precision);
            node_errors[name] = error;
            total_error = std::max(total_error, error);
        }
        
        return total_error;
    }
    
    double compute_memory_uniform(int precision) const {
        double total = 0.0;
        
        for (const auto& [name, node] : graph.nodes) {
            int64_t num_elements = 1;
            for (auto dim : node->output_shape) {
                num_elements *= dim;
            }
            total += num_elements * (precision / 8.0);
        }
        
        return total;
    }
    
    double compute_memory_mixed(const PrecisionAssignment& assignment) const {
        double total = 0.0;
        
        for (const auto& [name, precision] : assignment) {
            if (!graph.nodes.count(name)) continue;
            
            const auto& node = graph.nodes.at(name);
            int64_t num_elements = 1;
            for (auto dim : node->output_shape) {
                num_elements *= dim;
            }
            total += num_elements * (precision / 8.0);
        }
        
        return total;
    }
};

} // namespace sheaf
} // namespace hnf
