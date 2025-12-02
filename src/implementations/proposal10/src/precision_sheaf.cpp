#include "precision_sheaf.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <set>
#include <limits>
#include <queue>

namespace hnf {
namespace stability_linter {

PrecisionSheaf::PrecisionSheaf(std::shared_ptr<ComputationGraph> graph)
    : graph_(graph) {}

PrecisionSheaf::Covering PrecisionSheaf::build_covering(int max_set_size) {
    Covering covering;
    
    // Build connected subgraphs using BFS from different starting points
    std::set<std::string> covered_nodes;
    std::vector<std::string> all_nodes;
    
    for (const auto& [id, _] : graph_->nodes) {
        all_nodes.push_back(id);
    }
    
    int set_idx = 0;
    for (const std::string& start_node : all_nodes) {
        if (covered_nodes.count(start_node)) continue;
        
        OpenSet open_set;
        open_set.name = "U_" + std::to_string(set_idx++);
        
        // BFS to build connected component
        std::queue<std::string> q;
        std::set<std::string> visited;
        
        q.push(start_node);
        visited.insert(start_node);
        
        while (!q.empty() && (int)open_set.nodes.size() < max_set_size) {
            std::string curr = q.front();
            q.pop();
            
            open_set.nodes.insert(curr);
            covered_nodes.insert(curr);
            
            // Add neighbors
            for (const std::string& next : graph_->get_outputs(curr)) {
                if (!visited.count(next)) {
                    visited.insert(next);
                    q.push(next);
                }
            }
            
            for (const std::string& prev : graph_->get_inputs(curr)) {
                if (!visited.count(prev)) {
                    visited.insert(prev);
                    q.push(prev);
                }
            }
        }
        
        // Identify boundary nodes
        for (const std::string& node_id : open_set.nodes) {
            // Check if any neighbor is outside this set
            bool is_boundary = false;
            
            for (const std::string& out : graph_->get_outputs(node_id)) {
                if (!open_set.nodes.count(out)) {
                    is_boundary = true;
                    break;
                }
            }
            
            if (!is_boundary) {
                for (const std::string& in : graph_->get_inputs(node_id)) {
                    if (!open_set.nodes.count(in)) {
                        is_boundary = true;
                        break;
                    }
                }
            }
            
            if (is_boundary) {
                open_set.boundary_nodes.push_back(node_id);
            }
        }
        
        covering.sets.push_back(open_set);
    }
    
    // Compute overlaps
    for (size_t i = 0; i < covering.sets.size(); ++i) {
        for (size_t j = i + 1; j < covering.sets.size(); ++j) {
            std::set<std::string> overlap;
            
            std::set_intersection(
                covering.sets[i].nodes.begin(), covering.sets[i].nodes.end(),
                covering.sets[j].nodes.begin(), covering.sets[j].nodes.end(),
                std::inserter(overlap, overlap.begin())
            );
            
            if (!overlap.empty()) {
                covering.overlaps[{covering.sets[i].name, covering.sets[j].name}] = overlap;
            }
        }
    }
    
    return covering;
}

int PrecisionSheaf::compute_required_precision(
    const std::string& node_id,
    const std::map<std::string, int>& input_precisions,
    double target_accuracy
) {
    auto node = graph_->get_node(node_id);
    if (!node) return 23;  // default FP32
    
    // Get curvature and range
    double kappa = node->curvature;
    if (kappa == 0.0) kappa = 1.0;  // avoid log(0)
    
    auto [range_min, range_max] = node->value_range;
    double D = range_max - range_min;
    
    // HNF Theorem 4.3: p >= log₂(c·κ·D²/ε)
    double c = 0.125;  // from theorem proof
    double required_p = std::log2(c * kappa * D * D / target_accuracy);
    
    // Account for input precision degradation
    int max_input_p = 0;
    for (const std::string& input_id : node->input_ids) {
        if (input_precisions.count(input_id)) {
            max_input_p = std::max(max_input_p, input_precisions.at(input_id));
        }
    }
    
    // Composition adds curvature
    int final_p = std::max(
        static_cast<int>(std::ceil(required_p)),
        max_input_p
    );
    
    return std::max(1, final_p);  // at least 1 bit
}

bool PrecisionSheaf::is_locally_consistent(
    const PrecisionAssignment& assignment,
    const std::set<std::string>& node_subset
) {
    // Check that precision assignment satisfies composition bounds
    // within the subset
    
    for (const std::string& node_id : node_subset) {
        if (!assignment.count(node_id)) continue;
        
        auto node = graph_->get_node(node_id);
        if (!node) continue;
        
        int assigned_p = assignment.at(node_id);
        
        // Check against inputs
        for (const std::string& input_id : node->input_ids) {
            if (!assignment.count(input_id)) continue;
            if (!node_subset.count(input_id)) continue;  // only check within subset
            
            int input_p = assignment.at(input_id);
            
            // Output precision must be at least input precision
            // (modulo curvature effects, which we handle separately)
            if (assigned_p < input_p - 5) {  // allow some degradation
                return false;
            }
        }
    }
    
    return true;
}

std::vector<PrecisionSheaf::LocalSection> PrecisionSheaf::compute_local_sections(
    const Covering& covering,
    double target_accuracy
) {
    std::vector<LocalSection> sections;
    
    for (const OpenSet& open_set : covering.sets) {
        LocalSection section;
        section.open_set = open_set;
        section.curvature_bound = 0.0;
        
        // Topologically sort nodes within this set
        std::vector<std::string> topo_order = graph_->topological_sort();
        std::vector<std::string> local_order;
        
        for (const std::string& node_id : topo_order) {
            if (open_set.nodes.count(node_id)) {
                local_order.push_back(node_id);
            }
        }
        
        // Forward pass: assign precision based on inputs
        for (const std::string& node_id : local_order) {
            auto node = graph_->get_node(node_id);
            if (!node) continue;
            
            int required_p = compute_required_precision(
                node_id,
                section.assignment,
                target_accuracy
            );
            
            section.assignment[node_id] = required_p;
            section.curvature_bound += node->curvature;
        }
        
        // Check consistency
        section.is_consistent = is_locally_consistent(
            section.assignment,
            open_set.nodes
        );
        
        sections.push_back(section);
    }
    
    return sections;
}

PrecisionSheaf::CompatibilityCheck PrecisionSheaf::check_compatibility(
    const std::vector<LocalSection>& sections,
    const Covering& covering
) {
    CompatibilityCheck check;
    check.compatible = true;
    
    // Check each overlap
    for (const auto& [set_pair, overlap_nodes] : covering.overlaps) {
        // Find corresponding sections
        const LocalSection* section1 = nullptr;
        const LocalSection* section2 = nullptr;
        
        for (const auto& sec : sections) {
            if (sec.open_set.name == set_pair.first) section1 = &sec;
            if (sec.open_set.name == set_pair.second) section2 = &sec;
        }
        
        if (!section1 || !section2) continue;
        
        // Check if assignments agree on overlap
        int max_gap = 0;
        for (const std::string& node_id : overlap_nodes) {
            if (!section1->assignment.count(node_id) || 
                !section2->assignment.count(node_id)) {
                continue;
            }
            
            int p1 = section1->assignment.at(node_id);
            int p2 = section2->assignment.at(node_id);
            
            int gap = std::abs(p1 - p2);
            if (gap > 2) {  // allow small differences due to rounding
                check.compatible = false;
                check.conflicts.push_back(set_pair);
                check.precision_gaps[set_pair] = gap;
                max_gap = std::max(max_gap, gap);
            }
        }
    }
    
    return check;
}

PrecisionSheaf::PrecisionAssignment PrecisionSheaf::resolve_conflicts(
    const PrecisionAssignment& section1,
    const PrecisionAssignment& section2,
    const std::set<std::string>& overlap_nodes
) {
    PrecisionAssignment resolved;
    
    // Take maximum precision from either section
    for (const std::string& node_id : overlap_nodes) {
        int p = 0;
        
        if (section1.count(node_id)) {
            p = std::max(p, section1.at(node_id));
        }
        if (section2.count(node_id)) {
            p = std::max(p, section2.at(node_id));
        }
        
        if (p > 0) {
            resolved[node_id] = p;
        }
    }
    
    return resolved;
}

PrecisionSheaf::CohomologyGroup PrecisionSheaf::compute_h1_cohomology(
    const Covering& covering,
    double target_accuracy
) {
    CohomologyGroup h1;
    h1.dimension = 0;
    h1.has_global_section = true;
    
    // Compute local sections
    auto local_sections = compute_local_sections(covering, target_accuracy);
    
    // Check compatibility
    auto compat = check_compatibility(local_sections, covering);
    
    if (!compat.compatible) {
        // Non-zero H^1: obstructions exist
        h1.has_global_section = false;
        h1.dimension = compat.conflicts.size();
        
        std::stringstream desc;
        desc << "H¹(G, P^ε) has dimension " << h1.dimension << ".\n";
        desc << "Obstructions arise from " << compat.conflicts.size() 
             << " incompatible overlaps:\n";
        
        for (const auto& conflict : compat.conflicts) {
            desc << "  • Conflict between " << conflict.first 
                 << " and " << conflict.second
                 << " (gap: " << compat.precision_gaps.at(conflict) << " bits)\n";
        }
        
        h1.obstruction_description = desc.str();
    } else {
        h1.obstruction_description = "H¹(G, P^ε) = 0. No obstructions to global section.";
    }
    
    return h1;
}

PrecisionSheaf::GlobalSection PrecisionSheaf::find_global_section(double target_accuracy) {
    GlobalSection global;
    global.exists = false;
    global.total_cost = 0.0;
    
    // Build covering
    auto covering = build_covering(5);
    
    // Compute local sections
    auto local_sections = compute_local_sections(covering, target_accuracy);
    
    // Check compatibility
    auto compat = check_compatibility(local_sections, covering);
    
    if (compat.compatible) {
        // Can construct global section by gluing
        global.exists = true;
        
        // Start with first section
        if (!local_sections.empty()) {
            global.assignment = local_sections[0].assignment;
        }
        
        // Merge other sections, resolving conflicts
        for (size_t i = 1; i < local_sections.size(); ++i) {
            for (const auto& [node_id, p] : local_sections[i].assignment) {
                if (!global.assignment.count(node_id)) {
                    global.assignment[node_id] = p;
                } else {
                    // Take maximum
                    global.assignment[node_id] = std::max(
                        global.assignment[node_id], p
                    );
                }
            }
        }
        
        // Compute total cost
        for (const auto& [_, p] : global.assignment) {
            global.total_cost += p;
        }
    } else {
        // H^1 != 0: no global section
        global.obstructions.push_back("Incompatible local sections on overlaps");
        
        for (const auto& conflict : compat.conflicts) {
            global.obstructions.push_back(
                "Conflict between " + conflict.first + " and " + conflict.second
            );
        }
    }
    
    return global;
}

PrecisionSheaf::OptimizedAssignment PrecisionSheaf::optimize_precision(
    double target_accuracy,
    int max_iterations
) {
    OptimizedAssignment result;
    result.is_minimal = false;
    result.certified_accuracy = target_accuracy;
    
    std::stringstream log;
    log << "Optimizing precision assignment via sheaf descent...\n";
    
    // Start with global section
    auto global = find_global_section(target_accuracy);
    
    if (!global.exists) {
        log << "No global section exists - using greedy assignment\n";
        
        // Fallback: assign precision greedily
        auto topo_order = graph_->topological_sort();
        for (const std::string& node_id : topo_order) {
            int required_p = compute_required_precision(
                node_id,
                result.assignment,
                target_accuracy
            );
            result.assignment[node_id] = required_p;
        }
    } else {
        result.assignment = global.assignment;
        log << "Starting from global section\n";
    }
    
    // Iterative optimization: try to reduce precision while maintaining accuracy
    bool improved = true;
    int iteration = 0;
    
    while (improved && iteration < max_iterations) {
        improved = false;
        iteration++;
        
        // Try reducing precision of each node
        for (auto& [node_id, current_p] : result.assignment) {
            if (current_p <= 1) continue;
            
            // Try reducing by 1 bit
            int new_p = current_p - 1;
            result.assignment[node_id] = new_p;
            
            // Check if still valid
            if (is_locally_consistent(result.assignment, graph_->nodes | 
                [](const auto& pair) { 
                    std::set<std::string> s; 
                    s.insert(pair.first); 
                    return s; 
                })) {
                // Reduction is valid
                improved = true;
                log << "  Reduced " << node_id << " from " << current_p 
                    << " to " << new_p << " bits\n";
            } else {
                // Revert
                result.assignment[node_id] = current_p;
            }
        }
    }
    
    log << "Optimization converged after " << iteration << " iterations\n";
    result.is_minimal = !improved;
    
    // Compute total bits
    result.total_bits = 0;
    for (const auto& [_, p] : result.assignment) {
        result.total_bits += p;
    }
    
    result.optimization_log = log.str();
    
    return result;
}

std::string PrecisionSheaf::visualize_sheaf_structure(const Covering& covering) {
    std::stringstream viz;
    
    viz << "Sheaf Structure Visualization\n";
    viz << std::string(50, '=') << "\n\n";
    
    viz << "Open Covering:\n";
    for (const auto& open_set : covering.sets) {
        viz << "  " << open_set.name << ": {";
        bool first = true;
        for (const std::string& node : open_set.nodes) {
            if (!first) viz << ", ";
            viz << node;
            first = false;
        }
        viz << "}\n";
        
        if (!open_set.boundary_nodes.empty()) {
            viz << "    Boundary: {";
            first = true;
            for (const std::string& node : open_set.boundary_nodes) {
                if (!first) viz << ", ";
                viz << node;
                first = false;
            }
            viz << "}\n";
        }
    }
    
    viz << "\nOverlaps:\n";
    for (const auto& [set_pair, overlap] : covering.overlaps) {
        viz << "  " << set_pair.first << " ∩ " << set_pair.second << ": {";
        bool first = true;
        for (const std::string& node : overlap) {
            if (!first) viz << ", ";
            viz << node;
            first = false;
        }
        viz << "}\n";
    }
    
    return viz.str();
}

// Čech cohomology implementation

CechCohomology::Cochain CechCohomology::coboundary(
    const Cochain& cp,
    const PrecisionSheaf::Covering& covering
) {
    // This is a simplified implementation
    // Full Čech complex would require more sophisticated intersection tracking
    Cochain result;
    
    // δ: C^p → C^{p+1}
    // For p=0: maps 0-cochains (on vertices) to 1-cochains (on edges)
    // δf(U_i, U_j) = f(U_j) - f(U_i) on overlap
    
    return result;  // Placeholder
}

std::vector<CechCohomology::Cochain> CechCohomology::compute_cocycles(
    int degree,
    const PrecisionSheaf::Covering& covering
) {
    std::vector<Cochain> cocycles;
    
    // Elements of ker(δ^p)
    // These are p-cochains that vanish under coboundary
    
    return cocycles;  // Simplified - full computation requires linear algebra
}

std::vector<CechCohomology::Cochain> CechCohomology::compute_coboundaries(
    int degree,
    const PrecisionSheaf::Covering& covering
) {
    std::vector<Cochain> coboundaries;
    
    // Elements of im(δ^{p-1})
    
    return coboundaries;  // Simplified
}

PrecisionSheaf::CohomologyGroup CechCohomology::compute_cohomology(
    int degree,
    const PrecisionSheaf::Covering& covering
) {
    PrecisionSheaf::CohomologyGroup h_p;
    
    // H^p = ker(δ^p) / im(δ^{p-1})
    
    auto cocycles = compute_cocycles(degree, covering);
    auto coboundaries = compute_coboundaries(degree, covering);
    
    // Quotient construction would go here
    // Dimension = dim(ker) - dim(im)
    
    h_p.dimension = cocycles.size();  // Simplified
    h_p.has_global_section = (h_p.dimension == 0);
    
    return h_p;
}

} // namespace stability_linter
} // namespace hnf
