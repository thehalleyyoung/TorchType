#pragma once

#include "computation_graph.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <set>
#include <memory>
#include <optional>

namespace hnf {
namespace sheaf {

// Precision assignment: maps nodes to precision (in bits)
using PrecisionAssignment = std::unordered_map<std::string, int>;

// Open set in the cover (represented as set of node names)
using OpenSet = std::unordered_set<std::string>;

// Precision section over an open set
// Maps nodes in the open set to their precision
struct PrecisionSection {
    OpenSet support;  // The open set this section is defined over
    PrecisionAssignment assignment;  // Precision for each node
    
    PrecisionSection() = default;
    PrecisionSection(const OpenSet& s, const PrecisionAssignment& a)
        : support(s), assignment(a) {}
    
    // Restrict this section to a subset
    PrecisionSection restrict_to(const OpenSet& subset) const {
        PrecisionSection restricted;
        restricted.support = subset;
        
        for (const auto& node : subset) {
            if (assignment.count(node)) {
                restricted.assignment[node] = assignment.at(node);
            }
        }
        
        return restricted;
    }
    
    // Check compatibility with another section on intersection
    bool compatible_with(const PrecisionSection& other, int tolerance = 0) const {
        OpenSet intersection;
        for (const auto& node : support) {
            if (other.support.count(node)) {
                intersection.insert(node);
            }
        }
        
        for (const auto& node : intersection) {
            if (!assignment.count(node) || !other.assignment.count(node)) {
                return false;
            }
            
            int p1 = assignment.at(node);
            int p2 = other.assignment.at(node);
            
            if (std::abs(p1 - p2) > tolerance) {
                return false;
            }
        }
        
        return true;
    }
};

// Open cover of the computation graph
class OpenCover {
public:
    std::vector<OpenSet> sets;
    const ComputationGraph& graph;
    
    OpenCover(const ComputationGraph& g) : graph(g) {}
    
    // Build star cover: each open set contains a node and its neighbors
    static OpenCover star_cover(const ComputationGraph& graph) {
        OpenCover cover(graph);
        
        for (const auto& [name, node] : graph.nodes) {
            OpenSet star_set;
            star_set.insert(name);
            
            auto neighbors = graph.get_neighbors(name);
            star_set.insert(neighbors.begin(), neighbors.end());
            
            cover.sets.push_back(star_set);
        }
        
        return cover;
    }
    
    // Build path cover: overlapping windows along computation paths
    static OpenCover path_cover(const ComputationGraph& graph, int window_size = 3) {
        OpenCover cover(graph);
        
        const auto& order = graph.topological_order();
        
        for (size_t i = 0; i < order.size(); ++i) {
            OpenSet path_set;
            
            // Add window of nodes
            for (int j = static_cast<int>(i) - window_size + 1; 
                 j <= static_cast<int>(i) + window_size - 1 && j >= 0 && j < static_cast<int>(order.size()); 
                 ++j) {
                path_set.insert(order[j]);
            }
            
            if (!path_set.empty()) {
                cover.sets.push_back(path_set);
            }
        }
        
        return cover;
    }
    
    // Get intersection of two open sets
    static OpenSet intersection(const OpenSet& U, const OpenSet& V) {
        OpenSet result;
        for (const auto& node : U) {
            if (V.count(node)) {
                result.insert(node);
            }
        }
        return result;
    }
    
    // Get all pairwise intersections (for Čech complex)
    std::vector<std::pair<int, int>> get_intersections() const {
        std::vector<std::pair<int, int>> result;
        
        for (size_t i = 0; i < sets.size(); ++i) {
            for (size_t j = i + 1; j < sets.size(); ++j) {
                auto inter = intersection(sets[i], sets[j]);
                if (!inter.empty()) {
                    result.emplace_back(i, j);
                }
            }
        }
        
        return result;
    }
    
    // Get all triple intersections (for checking cocycle condition)
    std::vector<std::tuple<int, int, int>> get_triple_intersections() const {
        std::vector<std::tuple<int, int, int>> result;
        
        for (size_t i = 0; i < sets.size(); ++i) {
            for (size_t j = i + 1; j < sets.size(); ++j) {
                for (size_t k = j + 1; k < sets.size(); ++k) {
                    auto inter_ij = intersection(sets[i], sets[j]);
                    auto inter_ijk = intersection(inter_ij, sets[k]);
                    
                    if (!inter_ijk.empty()) {
                        result.emplace_back(i, j, k);
                    }
                }
            }
        }
        
        return result;
    }
};

// Cocycle: assigns precision gap to each edge in the cover
// Represents obstruction in H^1
struct Cocycle {
    // Maps (i, j) where i < j to precision gap on U_i ∩ U_j
    std::map<std::pair<int, int>, int> values;
    
    Cocycle() = default;
    
    int operator()(int i, int j) const {
        auto key = std::make_pair(std::min(i, j), std::max(i, j));
        if (values.count(key)) {
            return values.at(key);
        }
        return 0;
    }
    
    void set(int i, int j, int value) {
        auto key = std::make_pair(std::min(i, j), std::max(i, j));
        values[key] = value;
    }
    
    // Check cocycle condition: ω_ij + ω_jk + ω_ki = 0 on triple intersections
    bool satisfies_cocycle_condition(const OpenCover& cover) const {
        auto triples = cover.get_triple_intersections();
        
        for (const auto& [i, j, k] : triples) {
            int omega_ij = (*this)(i, j);
            int omega_jk = (*this)(j, k);
            int omega_ki = (*this)(k, i);
            
            // Note: omega_ki has opposite sign when accessed as (i,k)
            int sum = omega_ij + omega_jk - (*this)(i, k);
            
            if (sum != 0) {
                return false;
            }
        }
        
        return true;
    }
    
    // L1 norm of the cocycle (for minimization)
    int l1_norm() const {
        int sum = 0;
        for (const auto& [_, val] : values) {
            sum += std::abs(val);
        }
        return sum;
    }
};

// Precision Sheaf: the sheaf P_G^ε from Section 4.4 of the paper
class PrecisionSheaf {
public:
    const ComputationGraph& graph;
    OpenCover cover;
    double target_accuracy;
    
    // C^0: sections over each open set
    std::vector<std::vector<PrecisionSection>> C0;
    
    // C^1: sections over pairwise intersections
    std::map<std::pair<int, int>, std::vector<PrecisionSection>> C1;
    
    PrecisionSheaf(
        const ComputationGraph& g,
        double eps,
        const OpenCover& c
    ) : graph(g), target_accuracy(eps), cover(c) {
        compute_local_sections();
    }
    
private:
    // Compute all valid precision sections over each open set
    void compute_local_sections() {
        C0.resize(cover.sets.size());
        
        // For each open set, compute valid precision assignments
        for (size_t i = 0; i < cover.sets.size(); ++i) {
            const auto& U = cover.sets[i];
            
            // For now, generate one section per reasonable precision level
            // In practice, we'd enumerate all valid assignments
            std::vector<int> precisions = {7, 10, 16, 23, 32, 52};
            
            for (int p : precisions) {
                PrecisionSection section;
                section.support = U;
                
                // Check if this precision is sufficient for all nodes in U
                bool valid = true;
                for (const auto& node_name : U) {
                    if (!graph.nodes.count(node_name)) continue;
                    
                    const auto& node = graph.nodes.at(node_name);
                    
                    // Use mutable copy to compute min precision
                    auto node_copy = *node;
                    node_copy.compute_min_precision(target_accuracy);
                    
                    if (p >= node_copy.min_precision_bits) {
                        section.assignment[node_name] = p;
                    } else {
                        valid = false;
                        break;
                    }
                }
                
                if (valid && !section.assignment.empty()) {
                    C0[i].push_back(section);
                }
            }
        }
        
        // Compute C^1: compatible sections on intersections
        auto intersections = cover.get_intersections();
        for (const auto& [i, j] : intersections) {
            auto inter = OpenCover::intersection(cover.sets[i], cover.sets[j]);
            
            std::vector<PrecisionSection> sections;
            
            // A section on intersection must come from compatible sections on U_i and U_j
            for (const auto& sec_i : C0[i]) {
                for (const auto& sec_j : C0[j]) {
                    if (sec_i.compatible_with(sec_j, 0)) {
                        // They agree on intersection
                        sections.push_back(sec_i.restrict_to(inter));
                    }
                }
            }
            
            C1[{i, j}] = sections;
        }
    }
    
public:
    // Compute H^0: global sections (kernel of d^0)
    std::vector<PrecisionAssignment> compute_H0() const {
        std::vector<PrecisionAssignment> global_sections;
        
        if (C0.empty()) return global_sections;
        
        // A global section is a choice of section from each C0[i]
        // such that they all agree on intersections
        
        // Use backtracking to find compatible assignments
        std::vector<int> choice(cover.sets.size(), -1);
        
        std::function<void(int)> backtrack = [&](int idx) {
            if (idx == static_cast<int>(cover.sets.size())) {
                // Found a complete assignment - extract it
                PrecisionAssignment assignment;
                for (size_t i = 0; i < cover.sets.size(); ++i) {
                    if (choice[i] >= 0 && choice[i] < static_cast<int>(C0[i].size())) {
                        const auto& section = C0[i][choice[i]];
                        for (const auto& [node, prec] : section.assignment) {
                            assignment[node] = prec;
                        }
                    }
                }
                global_sections.push_back(assignment);
                return;
            }
            
            // Try each section for this open set
            for (int s = 0; s < static_cast<int>(C0[idx].size()); ++s) {
                choice[idx] = s;
                
                // Check compatibility with all previous choices
                bool compatible = true;
                for (int prev = 0; prev < idx; ++prev) {
                    if (choice[prev] < 0) continue;
                    
                    // Check if sections are compatible
                    if (!C0[prev][choice[prev]].compatible_with(C0[idx][s], 0)) {
                        compatible = false;
                        break;
                    }
                }
                
                if (compatible) {
                    backtrack(idx + 1);
                }
            }
            
            choice[idx] = -1;
        };
        
        backtrack(0);
        return global_sections;
    }
    
    // Compute H^1: obstruction cocycles
    std::vector<Cocycle> compute_H1() const {
        std::vector<Cocycle> cocycles;
        
        // A 1-cocycle assigns an integer to each edge (i,j)
        // such that the cocycle condition holds on triple intersections
        
        auto intersections = cover.get_intersections();
        
        if (intersections.empty()) {
            return cocycles;
        }
        
        // Build the cocycle space using linear algebra
        // Variables: one for each intersection
        int n_edges = static_cast<int>(intersections.size());
        
        // Build constraint matrix for cocycle condition
        // Each triple intersection gives a constraint
        auto triples = cover.get_triple_intersections();
        
        Eigen::MatrixXi A = Eigen::MatrixXi::Zero(static_cast<int>(triples.size()), n_edges);
        
        std::map<std::pair<int, int>, int> edge_to_idx;
        for (int e = 0; e < n_edges; ++e) {
            auto [i, j] = intersections[e];
            edge_to_idx[{i, j}] = e;
        }
        
        for (size_t t = 0; t < triples.size(); ++t) {
            auto [i, j, k] = triples[t];
            
            // Constraint: ω_ij + ω_jk - ω_ik = 0
            if (edge_to_idx.count({i, j})) {
                A(static_cast<int>(t), edge_to_idx[{i, j}]) = 1;
            }
            if (edge_to_idx.count({j, k})) {
                A(static_cast<int>(t), edge_to_idx[{j, k}]) = 1;
            }
            if (edge_to_idx.count({i, k})) {
                A(static_cast<int>(t), edge_to_idx[{i, k}]) = -1;
            }
        }
        
        // For now, return a simple obstruction cocycle based on min precision differences
        Cocycle obstruction;
        for (const auto& [i, j] : intersections) {
            auto inter = OpenCover::intersection(cover.sets[i], cover.sets[j]);
            
            if (inter.empty()) continue;
            
            // Compute minimal precision needed on this intersection
            int min_prec_i = 23;  // default
            int min_prec_j = 23;
            
            for (const auto& node_name : inter) {
                if (!graph.nodes.count(node_name)) continue;
                
                auto node_copy = *graph.nodes.at(node_name);
                node_copy.compute_min_precision(target_accuracy);
                
                min_prec_i = std::max(min_prec_i, node_copy.min_precision_bits);
                min_prec_j = std::max(min_prec_j, node_copy.min_precision_bits);
            }
            
            // Gap represents precision incompatibility
            int gap = 0;
            if (!C1.count({i, j}) || C1.at({i, j}).empty()) {
                gap = min_prec_i - min_prec_j;
            }
            
            obstruction.set(i, j, gap);
        }
        
        if (obstruction.l1_norm() > 0) {
            cocycles.push_back(obstruction);
        }
        
        return cocycles;
    }
    
    // Check if sheaf has global sections
    bool has_global_sections() const {
        return !compute_H0().empty();
    }
    
    // Get the obstruction to global sections
    std::optional<Cocycle> get_obstruction() const {
        auto H0 = compute_H0();
        if (!H0.empty()) {
            return std::nullopt;  // No obstruction
        }
        
        auto H1 = compute_H1();
        if (!H1.empty()) {
            return H1[0];  // Return first obstruction
        }
        
        return std::nullopt;
    }
    
    // Get the cover for this sheaf
    const OpenCover& get_cover() const {
        return cover;
    }
    
    // Get section over an open set (simplified version for advanced_sheaf_theory)
    std::vector<PrecisionSection> get_section(const OpenSet& open_set) const {
        // Find which cover index this corresponds to
        for (size_t i = 0; i < cover.sets.size(); ++i) {
            if (cover.sets[i] == open_set) {
                if (i < C0.size()) {
                    return C0[i];
                }
            }
        }
        return {};  // Empty if not found
    }
};

} // namespace sheaf
} // namespace hnf
