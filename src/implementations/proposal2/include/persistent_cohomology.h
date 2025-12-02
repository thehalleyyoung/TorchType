#pragma once

#include "computation_graph.h"
#include "precision_sheaf.h"
#include <vector>
#include <map>
#include <set>
#include <algorithm>
#include <cmath>

namespace hnf {
namespace sheaf {

/**
 * Persistence diagram for tracking cohomology across filtration
 * 
 * In persistent cohomology, we track how H^0 and H^1 change as we vary
 * the target accuracy epsilon. This reveals critical accuracy thresholds
 * where mixed precision becomes required or unnecessary.
 */
struct PersistenceInterval {
    double birth;      // Accuracy level where feature appears
    double death;      // Accuracy level where feature disappears
    int dimension;     // Cohomology dimension (0 or 1)
    
    // Associated data
    std::string description;
    std::vector<std::string> critical_nodes;
    
    double persistence() const {
        return std::log10(death) - std::log10(birth);
    }
    
    bool is_infinite() const {
        return death >= 1e100;
    }
};

struct PersistenceDiagram {
    std::vector<PersistenceInterval> intervals;
    
    // Bottleneck distance to another diagram
    double bottleneck_distance(const PersistenceDiagram& other) const {
        // Simplified bottleneck distance computation
        // In full implementation, this would use Hungarian algorithm
        
        if (intervals.empty() && other.intervals.empty()) {
            return 0.0;
        }
        
        double max_dist = 0.0;
        
        for (const auto& interval : intervals) {
            double min_match_dist = 1e100;
            
            for (const auto& other_interval : other.intervals) {
                if (interval.dimension != other_interval.dimension) continue;
                
                double dist = std::max(
                    std::abs(interval.birth - other_interval.birth),
                    std::abs(interval.death - other_interval.death)
                );
                
                min_match_dist = std::min(min_match_dist, dist);
            }
            
            // Distance to diagonal
            double diagonal_dist = std::abs(interval.death - interval.birth) / 2.0;
            min_match_dist = std::min(min_match_dist, diagonal_dist);
            
            max_dist = std::max(max_dist, min_match_dist);
        }
        
        return max_dist;
    }
    
    // Filter intervals by persistence threshold
    PersistenceDiagram filter_by_persistence(double threshold) const {
        PersistenceDiagram filtered;
        
        for (const auto& interval : intervals) {
            if (interval.persistence() >= threshold) {
                filtered.intervals.push_back(interval);
            }
        }
        
        return filtered;
    }
};

/**
 * Persistent cohomology analyzer for precision sheaves
 * 
 * Key innovation: We vary epsilon from high (easy) to low (hard) and track:
 * - When does H^0 become empty? (mixed precision becomes required)
 * - When do new obstructions appear in H^1?
 * - Which nodes are the critical bottlenecks?
 * 
 * This gives us a topological fingerprint of the precision requirements.
 */
class PersistentCohomologyAnalyzer {
private:
    const ComputationGraph& graph;
    std::vector<double> epsilon_values;
    
    // Cached cohomology dimensions at each epsilon
    std::map<double, int> h0_dimensions;
    std::map<double, int> h1_dimensions;
    
    // Tracked features
    std::map<double, std::set<std::string>> active_obstructions;
    
public:
    PersistentCohomologyAnalyzer(const ComputationGraph& g) 
        : graph(g) 
    {
        // Generate geometric sequence of epsilon values
        generate_epsilon_filtration(1e-1, 1e-10, 50);
    }
    
    /**
     * Compute persistence diagram by varying epsilon
     */
    PersistenceDiagram compute_persistence_diagram() {
        PersistenceDiagram diagram;
        
        // Build cover (same for all epsilon values)
        auto cover = OpenCover::star_cover(graph);
        
        // Track when H^0 generators appear/disappear
        std::map<std::string, double> h0_births;
        std::map<std::string, double> h0_deaths;
        
        // Track when H^1 generators appear/disappear
        std::map<std::string, double> h1_births;
        std::map<std::string, double> h1_deaths;
        
        std::set<std::string> previous_h0_gens;
        std::set<std::string> previous_h1_gens;
        
        for (double eps : epsilon_values) {
            // Build sheaf at this epsilon
            PrecisionSheaf sheaf(const_cast<ComputationGraph&>(graph), eps, cover);
            
            // Compute cohomology
            auto H0 = sheaf.compute_H0();
            auto H1_cocycles = sheaf.compute_H1();
            
            h0_dimensions[eps] = static_cast<int>(H0.size());
            h1_dimensions[eps] = static_cast<int>(H1_cocycles.size());
            
            // Identify H^0 generators (equivalence classes of global sections)
            std::set<std::string> current_h0_gens;
            for (size_t i = 0; i < H0.size(); ++i) {
                std::string gen_id = "H0_" + std::to_string(i);
                current_h0_gens.insert(gen_id);
                
                if (!previous_h0_gens.count(gen_id)) {
                    h0_births[gen_id] = eps;
                }
            }
            
            // Check which H^0 generators died
            for (const auto& gen_id : previous_h0_gens) {
                if (!current_h0_gens.count(gen_id)) {
                    h0_deaths[gen_id] = eps;
                }
            }
            
            // Identify H^1 generators (cohomology classes of obstructions)
            std::set<std::string> current_h1_gens;
            for (size_t i = 0; i < H1_cocycles.size(); ++i) {
                std::string gen_id = "H1_" + std::to_string(i);
                current_h1_gens.insert(gen_id);
                
                if (!previous_h1_gens.count(gen_id)) {
                    h1_births[gen_id] = eps;
                }
            }
            
            // Check which H^1 generators died
            for (const auto& gen_id : previous_h1_gens) {
                if (!current_h1_gens.count(gen_id)) {
                    h1_deaths[gen_id] = eps;
                }
            }
            
            previous_h0_gens = current_h0_gens;
            previous_h1_gens = current_h1_gens;
        }
        
        // Remaining generators are infinite (persist to epsilon = 0)
        for (const auto& gen_id : previous_h0_gens) {
            h0_deaths[gen_id] = 1e-100;  // Essentially zero
        }
        
        for (const auto& gen_id : previous_h1_gens) {
            h1_deaths[gen_id] = 1e-100;
        }
        
        // Create persistence intervals for H^0
        for (const auto& [gen_id, birth] : h0_births) {
            PersistenceInterval interval;
            interval.birth = birth;
            interval.death = h0_deaths.count(gen_id) ? h0_deaths[gen_id] : 1e100;
            interval.dimension = 0;
            interval.description = "Global precision assignment " + gen_id;
            
            diagram.intervals.push_back(interval);
        }
        
        // Create persistence intervals for H^1
        for (const auto& [gen_id, birth] : h1_births) {
            PersistenceInterval interval;
            interval.birth = birth;
            interval.death = h1_deaths.count(gen_id) ? h1_deaths[gen_id] : 1e100;
            interval.dimension = 1;
            interval.description = "Cohomological obstruction " + gen_id;
            
            diagram.intervals.push_back(interval);
        }
        
        return diagram;
    }
    
    /**
     * Find critical accuracy threshold where mixed precision becomes required
     * 
     * This is the largest epsilon where H^0 becomes empty
     */
    double find_mixed_precision_threshold() {
        auto cover = OpenCover::star_cover(graph);
        
        // Binary search for the threshold
        double low = epsilon_values.back();
        double high = epsilon_values.front();
        double threshold = high;
        
        for (double eps : epsilon_values) {
            PrecisionSheaf sheaf(const_cast<ComputationGraph&>(graph), eps, cover);
            auto H0 = sheaf.compute_H0();
            
            if (H0.empty()) {
                // Mixed precision required at this epsilon
                threshold = eps;
                break;
            }
        }
        
        return threshold;
    }
    
    /**
     * Compute Betti numbers as a function of epsilon
     * Returns two vectors: (beta_0(eps), beta_1(eps))
     */
    std::pair<std::vector<double>, std::vector<double>> compute_betti_curves() {
        std::vector<double> beta_0_values;
        std::vector<double> beta_1_values;
        
        auto cover = OpenCover::star_cover(graph);
        
        for (double eps : epsilon_values) {
            PrecisionSheaf sheaf(const_cast<ComputationGraph&>(graph), eps, cover);
            
            auto H0 = sheaf.compute_H0();
            auto H1 = sheaf.compute_H1();
            
            beta_0_values.push_back(static_cast<double>(H0.size()));
            beta_1_values.push_back(static_cast<double>(H1.size()));
        }
        
        return {beta_0_values, beta_1_values};
    }
    
    /**
     * Identify critical nodes that create cohomological obstructions
     * 
     * These are the nodes whose curvature forces precision increases
     */
    std::vector<std::pair<std::string, double>> identify_critical_nodes() {
        std::vector<std::pair<std::string, double>> critical_nodes;
        
        // For each node, compute its "obstruction score"
        // This measures how much the node contributes to H^1
        
        for (const auto& [name, node] : graph.nodes) {
            double obstruction_score = 0.0;
            
            // High curvature = high obstruction potential
            obstruction_score += std::log10(1.0 + node->curvature);
            
            // Mismatch between node's precision and neighbors
            double precision_variance = compute_precision_variance_at_node(name);
            obstruction_score += precision_variance;
            
            // Connectivity: highly connected nodes cause more obstructions
            auto neighbors = graph.get_neighbors(name);
            obstruction_score += std::log10(1.0 + neighbors.size()) * 0.5;
            
            if (obstruction_score > 1e-6) {
                critical_nodes.emplace_back(name, obstruction_score);
            }
        }
        
        // Sort by obstruction score (descending)
        std::sort(critical_nodes.begin(), critical_nodes.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        return critical_nodes;
    }
    
    /**
     * Compute spectral sequence for multi-scale precision analysis
     * 
     * The spectral sequence E_r^{p,q} converges to the cohomology of the
     * total complex, allowing us to decompose precision requirements by scale.
     */
    struct SpectralSequencePage {
        int r;  // Page number
        std::map<std::pair<int, int>, int> dimensions;  // E_r^{p,q} dimensions
        
        int total_dimension() const {
            int total = 0;
            for (const auto& [pq, dim] : dimensions) {
                total += dim;
            }
            return total;
        }
    };
    
    std::vector<SpectralSequencePage> compute_spectral_sequence(int max_page = 5) {
        std::vector<SpectralSequencePage> pages;
        
        // E_0 page: direct sum of precision requirements at each node
        SpectralSequencePage E0;
        E0.r = 0;
        
        int node_idx = 0;
        for (const auto& [name, node] : graph.nodes) {
            // Each node contributes to (p, 0) where p is its index in topological order
            E0.dimensions[{node_idx, 0}] = 1;  // One precision choice per node
            node_idx++;
        }
        
        pages.push_back(E0);
        
        // E_1 page: account for edge compatibilities
        SpectralSequencePage E1;
        E1.r = 1;
        
        // Compute boundary map dimensions
        for (const auto& edge : graph.edges) {
            // Edge contributes to differentials
            // Simplified: just track dimension changes
            
            // Find indices of source and target nodes
            int src_idx = 0, tgt_idx = 0;
            int idx = 0;
            for (const auto& [name, node] : graph.nodes) {
                if (name == edge.source) src_idx = idx;
                if (name == edge.target) tgt_idx = idx;
                idx++;
            }
            
            // Differential d_1: E_1^{p,0} -> E_1^{p+1,0}
            // Compatibility constraint reduces dimension
            int p = std::min(src_idx, tgt_idx);
            
            if (E0.dimensions.count({p, 0})) {
                E1.dimensions[{p, 0}] = std::max(0, E0.dimensions[{p, 0}] - 1);
            }
        }
        
        pages.push_back(E1);
        
        // Higher pages: iterate differentials
        for (int r = 2; r <= max_page; ++r) {
            SpectralSequencePage E_r;
            E_r.r = r;
            
            // Differential d_r: E_r^{p,q} -> E_r^{p+r, q-r+1}
            // Kernel / Image computation (simplified)
            
            const auto& E_prev = pages.back();
            for (const auto& [pq, dim] : E_prev.dimensions) {
                if (dim > 0) {
                    // Most entries stabilize after a few pages
                    E_r.dimensions[pq] = dim;
                }
            }
            
            pages.push_back(E_r);
            
            // Check for stabilization
            if (r > 2 && E_r.total_dimension() == E_prev.total_dimension()) {
                break;  // Spectral sequence has converged
            }
        }
        
        return pages;
    }
    
    /**
     * Analyze precision stability under perturbations
     * 
     * How does the optimal precision assignment change when we perturb
     * the curvature or accuracy requirements?
     */
    struct StabilityAnalysis {
        double curvature_sensitivity;    // d(precision) / d(curvature)
        double accuracy_sensitivity;      // d(precision) / d(epsilon)
        bool is_stable;                   // Small perturbations don't change structure
        
        std::vector<std::string> unstable_nodes;  // Nodes with high sensitivity
    };
    
    StabilityAnalysis analyze_stability(double perturbation = 0.01) {
        StabilityAnalysis result;
        result.is_stable = true;
        
        // Compute baseline persistence diagram
        PersistenceDiagram baseline = compute_persistence_diagram();
        
        // Perturb curvatures and recompute
        for (auto& [name, node] : const_cast<ComputationGraph&>(graph).nodes) {
            double original_curvature = node->curvature;
            
            // Increase curvature
            node->curvature *= (1.0 + perturbation);
            PersistenceDiagram perturbed_up = compute_persistence_diagram();
            
            // Decrease curvature
            node->curvature = original_curvature * (1.0 - perturbation);
            PersistenceDiagram perturbed_down = compute_persistence_diagram();
            
            // Restore original
            node->curvature = original_curvature;
            
            // Compute bottleneck distances
            double dist_up = baseline.bottleneck_distance(perturbed_up);
            double dist_down = baseline.bottleneck_distance(perturbed_down);
            
            double sensitivity = std::max(dist_up, dist_down) / perturbation;
            
            if (sensitivity > 10.0) {
                result.unstable_nodes.push_back(name);
                result.is_stable = false;
            }
            
            result.curvature_sensitivity = std::max(result.curvature_sensitivity, sensitivity);
        }
        
        // TODO: Similar analysis for epsilon perturbations
        result.accuracy_sensitivity = 0.0;
        
        return result;
    }
    
private:
    void generate_epsilon_filtration(double eps_start, double eps_end, int num_points) {
        epsilon_values.clear();
        
        // Geometric sequence
        double log_start = std::log10(eps_start);
        double log_end = std::log10(eps_end);
        double step = (log_end - log_start) / (num_points - 1);
        
        for (int i = 0; i < num_points; ++i) {
            double log_eps = log_start + i * step;
            epsilon_values.push_back(std::pow(10.0, log_eps));
        }
    }
    
    double compute_precision_variance_at_node(const std::string& node_name) const {
        if (!graph.nodes.count(node_name)) {
            return 0.0;
        }
        
        auto neighbors = graph.get_neighbors(node_name);
        if (neighbors.empty()) {
            return 0.0;
        }
        
        double node_prec = static_cast<double>(graph.nodes.at(node_name)->min_precision_bits);
        
        double sum_squared_diff = 0.0;
        int count = 0;
        
        for (const auto& neighbor : neighbors) {
            if (graph.nodes.count(neighbor)) {
                double neighbor_prec = static_cast<double>(graph.nodes.at(neighbor)->min_precision_bits);
                double diff = node_prec - neighbor_prec;
                sum_squared_diff += diff * diff;
                count++;
            }
        }
        
        if (count == 0) {
            return 0.0;
        }
        
        return std::sqrt(sum_squared_diff / count);
    }
};

} // namespace sheaf
} // namespace hnf
