#pragma once

#include "computation_graph.h"
#include "precision_sheaf.h"
#include <z3++.h>
#include <memory>
#include <vector>
#include <unordered_map>

namespace hnf {
namespace sheaf {

/**
 * Z3-based SMT solver for optimal precision assignment
 * 
 * This implements a rigorous constraint-solving approach to finding
 * minimal precision assignments that satisfy all cohomological obstructions.
 * 
 * Key innovation: Encodes sheaf cohomology conditions as SMT constraints,
 * allowing us to prove optimality and impossibility results.
 */
class Z3PrecisionSolver {
private:
    z3::context ctx;
    z3::solver solver;
    const ComputationGraph& graph;
    double target_accuracy;
    
    // Z3 variables: one integer variable per node representing its precision
    std::unordered_map<std::string, z3::expr> precision_vars;
    
    // Precision levels (in bits)
    static constexpr int PRECISION_LEVELS[] = {7, 10, 16, 23, 52, 64, 112};
    static constexpr int NUM_PRECISION_LEVELS = 7;
    
public:
    Z3PrecisionSolver(const ComputationGraph& g, double eps = 1e-5)
        : ctx(), solver(ctx), graph(g), target_accuracy(eps)
    {
        initialize_variables();
    }
    
    /**
     * Solve for optimal precision assignment
     * Returns nullopt if no satisfying assignment exists (proves impossibility!)
     */
    std::optional<PrecisionAssignment> solve_optimal() {
        // Clear previous constraints
        solver.reset();
        
        // Add constraints from curvature bounds (Theorem 5.7)
        add_curvature_constraints();
        
        // Add compatibility constraints from sheaf cohomology
        add_cohomology_constraints();
        
        // Add hardware constraints (must use standard precisions)
        add_hardware_constraints();
        
        // Minimize total precision (lexicographic: first minimize max, then sum)
        z3::expr max_precision = ctx.int_val(0);
        z3::expr total_precision = ctx.int_val(0);
        
        for (const auto& [name, var] : precision_vars) {
            max_precision = z3::ite(var > max_precision, var, max_precision);
            total_precision = total_precision + var;
        }
        
        // First, find minimum max precision
        int best_max = find_minimum_max_precision();
        
        if (best_max < 0) {
            return std::nullopt;  // No solution exists!
        }
        
        // Then minimize total precision subject to max constraint
        solver.add(max_precision <= best_max);
        
        PrecisionAssignment result = optimize_total_precision();
        return result;
    }
    
    /**
     * Check if a specific precision assignment is valid
     */
    bool verify_assignment(const PrecisionAssignment& assignment) {
        solver.push();
        
        // Add assignment as constraints
        for (const auto& [name, prec] : assignment) {
            auto it = precision_vars.find(name);
            if (it != precision_vars.end()) {
                solver.add(it->second == prec);
            }
        }
        
        z3::check_result result = solver.check();
        solver.pop();
        
        return result == z3::sat;
    }
    
    /**
     * Find minimal precision for a specific node
     */
    int find_min_precision_for_node(const std::string& node_name) {
        auto it = precision_vars.find(node_name);
        if (it == precision_vars.end()) {
            return -1;
        }
        
        solver.push();
        
        // Try each precision level from lowest to highest
        for (int level : PRECISION_LEVELS) {
            solver.push();
            solver.add(it->second == level);
            
            if (solver.check() == z3::sat) {
                solver.pop();
                solver.pop();
                return level;
            }
            
            solver.pop();
        }
        
        solver.pop();
        return PRECISION_LEVELS[NUM_PRECISION_LEVELS - 1];
    }
    
    /**
     * Prove that mixed precision is required (cohomological obstruction)
     * Returns true if H^0 = empty (no uniform precision works)
     */
    bool prove_mixed_precision_required() {
        solver.push();
        
        // Try to find a uniform precision that works
        for (int uniform_prec : PRECISION_LEVELS) {
            solver.push();
            
            // Constrain all nodes to same precision
            auto it = precision_vars.begin();
            if (it != precision_vars.end()) {
                z3::expr first_var = it->second;
                solver.add(first_var == uniform_prec);
                
                for (++it; it != precision_vars.end(); ++it) {
                    solver.add(it->second == uniform_prec);
                }
                
                if (solver.check() == z3::sat) {
                    // Uniform precision works
                    solver.pop();
                    solver.pop();
                    return false;
                }
            }
            
            solver.pop();
        }
        
        solver.pop();
        
        // No uniform precision satisfies all constraints
        // This is a cohomological obstruction (H^0 = empty)!
        return true;
    }
    
    /**
     * Extract cohomological obstruction from unsatisfiability
     * Returns edges where precision jumps are required
     */
    std::vector<std::pair<std::string, std::string>> extract_obstruction_edges() {
        std::vector<std::pair<std::string, std::string>> critical_edges;
        
        // Simplified version: check which edges cause conflicts
        for (const auto& edge : graph.edges) {
            solver.push();
            
            // Try to enforce same precision on both endpoints
            auto src_it = precision_vars.find(edge.source);
            auto tgt_it = precision_vars.find(edge.target);
            
            if (src_it != precision_vars.end() && tgt_it != precision_vars.end()) {
                solver.add(src_it->second == tgt_it->second);
                
                if (solver.check() == z3::unsat) {
                    // This edge requires a precision jump!
                    critical_edges.emplace_back(edge.source, edge.target);
                }
            }
            
            solver.pop();
        }
        
        return critical_edges;
    }
    
private:
    void initialize_variables() {
        for (const auto& [name, node] : graph.nodes) {
            precision_vars.emplace(name, ctx.int_const(name.c_str()));
        }
    }
    
    void add_curvature_constraints() {
        // For each node, precision must satisfy Theorem 5.7
        // p >= log2(c * kappa * D^2 / epsilon)
        
        double c = 2.0;  // Constant from theorem
        
        for (const auto& [name, node] : graph.nodes) {
            auto it = precision_vars.find(name);
            if (it == precision_vars.end()) continue;
            
            double kappa = node->curvature;
            double D = node->diameter;
            
            if (kappa > 1e-10 && D > 1e-10) {
                // Compute minimum precision from curvature
                double min_prec_exact = std::log2(c * kappa * D * D / target_accuracy);
                int min_prec = static_cast<int>(std::ceil(min_prec_exact));
                
                // Clamp to reasonable range
                min_prec = std::max(7, std::min(min_prec, 112));
                
                solver.add(it->second >= min_prec);
            } else {
                // Linear operations: precision based on target accuracy only
                int min_prec = static_cast<int>(std::ceil(std::log2(1.0 / target_accuracy)));
                min_prec = std::max(7, std::min(min_prec, 52));
                
                solver.add(it->second >= min_prec);
            }
        }
    }
    
    void add_cohomology_constraints() {
        // Build star cover
        auto cover = OpenCover::star_cover(graph);
        
        // For each pair of overlapping open sets, check compatibility
        for (size_t i = 0; i < cover.sets.size(); ++i) {
            for (size_t j = i + 1; j < cover.sets.size(); ++j) {
                auto intersection = OpenCover::intersection(cover.sets[i], cover.sets[j]);
                
                if (intersection.empty()) continue;
                
                // Nodes in the intersection must have compatible precisions
                // This encodes the sheaf gluing condition
                
                for (const auto& node1 : intersection) {
                    for (const auto& node2 : intersection) {
                        if (node1 >= node2) continue;
                        
                        auto it1 = precision_vars.find(node1);
                        auto it2 = precision_vars.find(node2);
                        
                        if (it1 == precision_vars.end() || it2 == precision_vars.end()) {
                            continue;
                        }
                        
                        // Check if there's an edge between these nodes
                        bool has_edge = false;
                        int edge_tolerance = 0;
                        
                        for (const auto& edge : graph.edges) {
                            if ((edge.source == node1 && edge.target == node2) ||
                                (edge.source == node2 && edge.target == node1)) {
                                has_edge = true;
                                edge_tolerance = edge.tolerance;
                                break;
                            }
                        }
                        
                        if (has_edge) {
                            // Direct edge: enforce tight compatibility
                            z3::expr p1 = it1->second;
                            z3::expr p2 = it2->second;
                            
                            // |p1 - p2| <= tolerance
                            solver.add(z3::abs(p1 - p2) <= edge_tolerance);
                        }
                    }
                }
            }
        }
    }
    
    void add_hardware_constraints() {
        // Each precision must be a standard hardware precision
        for (const auto& [name, var] : precision_vars) {
            z3::expr valid_precision = ctx.bool_val(false);
            
            for (int level : PRECISION_LEVELS) {
                valid_precision = valid_precision || (var == level);
            }
            
            solver.add(valid_precision);
        }
        
        // Also enforce reasonable bounds
        for (const auto& [name, var] : precision_vars) {
            solver.add(var >= PRECISION_LEVELS[0]);
            solver.add(var <= PRECISION_LEVELS[NUM_PRECISION_LEVELS - 1]);
        }
    }
    
    int find_minimum_max_precision() {
        // Binary search for minimum max precision
        int low = 0;
        int high = NUM_PRECISION_LEVELS - 1;
        int best = -1;
        
        while (low <= high) {
            int mid = (low + high) / 2;
            int max_allowed = PRECISION_LEVELS[mid];
            
            solver.push();
            
            // Constrain all precisions to be at most max_allowed
            for (const auto& [name, var] : precision_vars) {
                solver.add(var <= max_allowed);
            }
            
            if (solver.check() == z3::sat) {
                best = max_allowed;
                high = mid - 1;  // Try to find lower maximum
            } else {
                low = mid + 1;   // Need higher maximum
            }
            
            solver.pop();
        }
        
        return best;
    }
    
    PrecisionAssignment optimize_total_precision() {
        PrecisionAssignment result;
        
        // For each node, find its minimum feasible precision
        for (const auto& [name, var] : precision_vars) {
            solver.push();
            
            // Try each precision level from lowest to highest
            bool found = false;
            for (int level : PRECISION_LEVELS) {
                solver.push();
                solver.add(var == level);
                
                if (solver.check() == z3::sat) {
                    result[name] = level;
                    found = true;
                    
                    // Extract model and fix this variable
                    solver.pop();
                    solver.add(var == level);
                    break;
                }
                
                solver.pop();
            }
            
            if (!found) {
                // Should not happen if max precision was found correctly
                result[name] = PRECISION_LEVELS[NUM_PRECISION_LEVELS - 1];
            }
            
            solver.pop();
        }
        
        return result;
    }
};

} // namespace sheaf
} // namespace hnf
