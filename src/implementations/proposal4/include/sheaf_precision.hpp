#pragma once

#include "graph_ir.hpp"
#include "curvature.hpp"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <cmath>
#include <algorithm>

namespace hnf {
namespace rewriter {

// Implements HNF Section 4: The Precision Sheaf
// 
// Key idea: Precision requirements form a sheaf over the computation graph.
// The sheaf P_G maps each open set U ⊆ G to a set of precision assignments.
// Sheaf cohomology H¹(G; P_G) classifies obstructions to global precision assignment.

// Precision assignment for a node
struct PrecisionAssignment {
    std::string node_id;
    int mantissa_bits;  // Required precision
    double local_error;  // Local error contribution
    double propagated_error;  // Total error after composition
    
    PrecisionAssignment() : mantissa_bits(53), local_error(0), propagated_error(0) {}
    PrecisionAssignment(const std::string& id, int bits) 
        : node_id(id), mantissa_bits(bits), local_error(0), propagated_error(0) {}
};

// A section of the precision sheaf over a subgraph
struct PrecisionSection {
    std::unordered_set<std::string> domain;  // Nodes in this open set
    std::unordered_map<std::string, PrecisionAssignment> assignments;
    
    // Restriction to a smaller open set
    PrecisionSection restrict_to(const std::unordered_set<std::string>& subdomain) const {
        PrecisionSection restricted;
        restricted.domain = subdomain;
        
        for (const auto& node_id : subdomain) {
            if (assignments.count(node_id)) {
                restricted.assignments[node_id] = assignments.at(node_id);
            }
        }
        
        return restricted;
    }
    
    // Check if two sections agree on their overlap
    bool agrees_with(const PrecisionSection& other) const {
        for (const auto& node_id : domain) {
            if (other.domain.count(node_id)) {
                // Both sections cover this node - check if they assign same precision
                if (assignments.count(node_id) && other.assignments.count(node_id)) {
                    const auto& a1 = assignments.at(node_id);
                    const auto& a2 = other.assignments.at(node_id);
                    
                    // Allow 1-bit tolerance for floating point issues
                    if (std::abs(a1.mantissa_bits - a2.mantissa_bits) > 1) {
                        return false;
                    }
                }
            }
        }
        return true;
    }
    
    // Merge with another section (if compatible)
    bool merge(const PrecisionSection& other) {
        if (!agrees_with(other)) {
            return false;  // Incompatible sections
        }
        
        // Merge domains
        for (const auto& node_id : other.domain) {
            domain.insert(node_id);
        }
        
        // Merge assignments
        for (const auto& [id, assignment] : other.assignments) {
            if (!assignments.count(id)) {
                assignments[id] = assignment;
            }
        }
        
        return true;
    }
};

// The precision sheaf over a computation graph
class PrecisionSheaf {
private:
    const Graph& graph_;
    const std::unordered_map<std::string, TensorStats>& stats_;
    double target_epsilon_;
    
public:
    PrecisionSheaf(const Graph& graph, 
                   const std::unordered_map<std::string, TensorStats>& stats,
                   double target_epsilon = 1e-6)
        : graph_(graph), stats_(stats), target_epsilon_(target_epsilon) {}
    
    // Compute local precision requirements for a node
    // Implements HNF Definition 4.3 (Local Precision Section)
    PrecisionAssignment compute_local_precision(const std::string& node_id) const {
        auto node = graph_.get_node(node_id);
        if (!node) {
            return PrecisionAssignment(node_id, 53);  // Default double precision
        }
        
        // Get curvature
        TensorStats node_stats;
        if (stats_.count(node_id)) {
            node_stats = stats_.at(node_id);
        } else if (!node->inputs.empty() && stats_.count(node->inputs[0])) {
            node_stats = stats_.at(node->inputs[0]);
        }
        
        // Create a temporary stats map for compute_node_curvature
        std::unordered_map<std::string, TensorStats> node_stats_map;
        for (const auto& inp : node->inputs) {
            if (stats_.count(inp)) {
                node_stats_map[inp] = stats_.at(inp);
            }
        }
        
        double curvature = CurvatureAnalyzer::compute_node_curvature(*node, node_stats_map);
        double diameter = node_stats.range();
        if (diameter < 1e-10) diameter = 1.0;
        
        // Apply Theorem 5.7: p ≥ log₂(c * κ * D² / ε)
        const double c = 1.0;
        double required_bits = std::log2(c * curvature * diameter * diameter / target_epsilon_);
        
        // Clamp to reasonable range
        int mantissa_bits = std::max(8, std::min(128, static_cast<int>(std::ceil(required_bits))));
        
        PrecisionAssignment assignment(node_id, mantissa_bits);
        assignment.local_error = curvature * target_epsilon_;
        
        return assignment;
    }
    
    // Compute precision section over a subgraph
    PrecisionSection compute_section(const std::unordered_set<std::string>& nodes) const {
        PrecisionSection section;
        section.domain = nodes;
        
        for (const auto& node_id : nodes) {
            section.assignments[node_id] = compute_local_precision(node_id);
        }
        
        // Propagate errors through dependencies
        auto topo_order = graph_.topological_order();
        for (const auto& node_id : topo_order) {
            if (!nodes.count(node_id)) continue;
            
            auto node = graph_.get_node(node_id);
            if (!node) continue;
            
            auto& assignment = section.assignments[node_id];
            assignment.propagated_error = assignment.local_error;
            
            // Add errors from inputs
            for (const auto& input_id : node->inputs) {
                if (section.assignments.count(input_id)) {
                    assignment.propagated_error += section.assignments[input_id].propagated_error;
                }
            }
        }
        
        return section;
    }
    
    // Compute sheaf cohomology H¹(G; P_G)
    // Non-zero cohomology indicates obstruction to global precision assignment
    struct SheafCohomology {
        int dimension;  // Dimension of H¹
        bool has_obstruction;  // True if H¹ ≠ 0
        std::vector<std::string> conflicting_nodes;  // Nodes with conflicting requirements
        double total_inconsistency;  // Measure of how inconsistent
        
        std::string to_string() const {
            std::ostringstream ss;
            ss << "Sheaf Cohomology H¹(G; P_G):\n";
            ss << "  Dimension: " << dimension << "\n";
            ss << "  Obstruction: " << (has_obstruction ? "YES" : "NO") << "\n";
            ss << "  Total inconsistency: " << total_inconsistency << "\n";
            if (!conflicting_nodes.empty()) {
                ss << "  Conflicting nodes: ";
                for (size_t i = 0; i < conflicting_nodes.size(); ++i) {
                    if (i > 0) ss << ", ";
                    ss << conflicting_nodes[i];
                }
                ss << "\n";
            }
            return ss.str();
        }
    };
    
    SheafCohomology compute_cohomology() const {
        SheafCohomology result;
        result.dimension = 0;
        result.has_obstruction = false;
        result.total_inconsistency = 0.0;
        
        // Create open cover of the graph
        // For simplicity, use star neighborhoods of each node
        std::vector<PrecisionSection> cover;
        
        for (const auto& [node_id, node] : graph_.nodes()) {
            // Star of node: node + all its neighbors
            std::unordered_set<std::string> star;
            star.insert(node_id);
            
            // Add inputs
            for (const auto& inp : node->inputs) {
                star.insert(inp);
            }
            
            // Add nodes that use this as input
            for (const auto& [other_id, other_node] : graph_.nodes()) {
                for (const auto& inp : other_node->inputs) {
                    if (inp == node_id) {
                        star.insert(other_id);
                        break;
                    }
                }
            }
            
            // Compute section over this star
            PrecisionSection section = compute_section(star);
            cover.push_back(section);
        }
        
        // Check Čech cocycle condition
        // For each pair of overlapping opens, check if sections agree
        for (size_t i = 0; i < cover.size(); ++i) {
            for (size_t j = i + 1; j < cover.size(); ++j) {
                const auto& s1 = cover[i];
                const auto& s2 = cover[j];
                
                // Find intersection
                std::unordered_set<std::string> intersection;
                for (const auto& id : s1.domain) {
                    if (s2.domain.count(id)) {
                        intersection.insert(id);
                    }
                }
                
                if (intersection.empty()) continue;
                
                // Check if restrictions agree
                auto r1 = s1.restrict_to(intersection);
                auto r2 = s2.restrict_to(intersection);
                
                if (!r1.agrees_with(r2)) {
                    result.has_obstruction = true;
                    result.dimension++;
                    
                    // Record conflicting nodes
                    for (const auto& id : intersection) {
                        if (r1.assignments.count(id) && r2.assignments.count(id)) {
                            int diff = std::abs(
                                r1.assignments.at(id).mantissa_bits - 
                                r2.assignments.at(id).mantissa_bits
                            );
                            if (diff > 1) {
                                result.conflicting_nodes.push_back(id);
                                result.total_inconsistency += diff;
                            }
                        }
                    }
                }
            }
        }
        
        return result;
    }
    
    // Attempt to resolve obstructions by finding global section
    // Uses greedy algorithm to minimize total bits while respecting composition
    PrecisionSection find_global_section() const {
        PrecisionSection global;
        
        // Initialize with all nodes
        for (const auto& [node_id, node] : graph_.nodes()) {
            global.domain.insert(node_id);
        }
        
        // Compute initial assignments
        for (const auto& node_id : global.domain) {
            global.assignments[node_id] = compute_local_precision(node_id);
        }
        
        // Iteratively refine to resolve inconsistencies
        bool changed = true;
        int iterations = 0;
        const int max_iterations = 100;
        
        while (changed && iterations < max_iterations) {
            changed = false;
            iterations++;
            
            // For each node, check if precision is sufficient for its consumers
            for (const auto& [node_id, node] : graph_.nodes()) {
                auto& assignment = global.assignments[node_id];
                
                // Find nodes that consume this node's output
                int max_required = assignment.mantissa_bits;
                
                for (const auto& [other_id, other_node] : graph_.nodes()) {
                    for (const auto& inp : other_node->inputs) {
                        if (inp == node_id) {
                            // This node is an input to other_node
                            // Check if our precision is sufficient
                            const auto& consumer_assignment = global.assignments[other_id];
                            
                            // Consumer needs input precision ≥ its own precision - Lipschitz factor
                            TensorStats input_stats_temp;
                            if (stats_.count(other_id)) {
                                input_stats_temp = stats_.at(other_id);
                            }
                            
                            // Create stats map for the node
                            std::unordered_map<std::string, TensorStats> temp_stats_map;
                            for (const auto& inp : other_node->inputs) {
                                if (stats_.count(inp)) {
                                    temp_stats_map[inp] = stats_.at(inp);
                                }
                            }
                            
                            double L = CurvatureAnalyzer::compute_node_curvature(*other_node, temp_stats_map);
                            int required_input_bits = static_cast<int>(
                                consumer_assignment.mantissa_bits + std::log2(std::max(1.0, L))
                            );
                            
                            max_required = std::max(max_required, required_input_bits);
                        }
                    }
                }
                
                if (max_required > assignment.mantissa_bits) {
                    assignment.mantissa_bits = max_required;
                    changed = true;
                }
            }
        }
        
        return global;
    }
    
    // Compute precision budget for entire graph
    struct PrecisionBudget {
        double total_required_bits;
        double max_node_bits;
        double min_node_bits;
        double avg_node_bits;
        std::unordered_map<std::string, int> node_precisions;
        
        std::string to_string() const {
            std::ostringstream ss;
            ss << "Precision Budget:\n";
            ss << "  Total bits: " << total_required_bits << "\n";
            ss << "  Max node: " << max_node_bits << " bits\n";
            ss << "  Min node: " << min_node_bits << " bits\n";
            ss << "  Average: " << avg_node_bits << " bits\n";
            return ss.str();
        }
    };
    
    PrecisionBudget compute_budget() const {
        PrecisionBudget budget;
        budget.total_required_bits = 0;
        budget.max_node_bits = 0;
        budget.min_node_bits = 1000;
        
        auto global = find_global_section();
        
        for (const auto& [id, assignment] : global.assignments) {
            int bits = assignment.mantissa_bits;
            budget.total_required_bits += bits;
            budget.max_node_bits = std::max(budget.max_node_bits, static_cast<double>(bits));
            budget.min_node_bits = std::min(budget.min_node_bits, static_cast<double>(bits));
            budget.node_precisions[id] = bits;
        }
        
        if (!global.assignments.empty()) {
            budget.avg_node_bits = budget.total_required_bits / global.assignments.size();
        }
        
        return budget;
    }
};

} // namespace rewriter
} // namespace hnf
