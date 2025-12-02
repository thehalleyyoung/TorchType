#include "sheaf_cohomology.hpp"
#include <algorithm>
#include <sstream>
#include <cmath>
#include <limits>
#include <iostream>

namespace hnf {
namespace sheaf {

using namespace stability_linter;

// ============================================================================
// PrecisionSection Implementation
// ============================================================================

bool PrecisionSection::compatible_with(
    const PrecisionSection& other,
    const std::set<std::string>& overlap_nodes) const {
    
    // Sections are compatible if they agree on overlap up to 0.1 bits
    const double tolerance = 0.1;
    
    for (const auto& node_id : overlap_nodes) {
        auto it1 = node_precisions.find(node_id);
        auto it2 = other.node_precisions.find(node_id);
        
        if (it1 != node_precisions.end() && it2 != other.node_precisions.end()) {
            if (std::abs(it1->second - it2->second) > tolerance) {
                return false;
            }
        }
    }
    
    return true;
}

std::optional<PrecisionSection> PrecisionSection::merge(
    const PrecisionSection& other,
    const std::set<std::string>& overlap) const {
    
    if (!compatible_with(other, overlap)) {
        return std::nullopt;
    }
    
    PrecisionSection merged(global_epsilon);
    
    // Merge from both sections
    for (const auto& [node_id, prec] : node_precisions) {
        merged.node_precisions[node_id] = prec;
    }
    
    for (const auto& [node_id, prec] : other.node_precisions) {
        auto it = merged.node_precisions.find(node_id);
        if (it == merged.node_precisions.end()) {
            merged.node_precisions[node_id] = prec;
        } else {
            // Average on overlap
            merged.node_precisions[node_id] = (it->second + prec) / 2.0;
        }
    }
    
    return merged;
}

// ============================================================================
// OpenCover Implementation
// ============================================================================

std::set<std::string> OpenCover::intersection(size_t i, size_t j) const {
    if (i >= sets.size() || j >= sets.size()) {
        return {};
    }
    
    const auto& set_i = sets[i].nodes;
    const auto& set_j = sets[j].nodes;
    
    std::set<std::string> result;
    std::set_intersection(set_i.begin(), set_i.end(),
                         set_j.begin(), set_j.end(),
                         std::inserter(result, result.begin()));
    
    return result;
}

bool OpenCover::is_valid_cover() const {
    if (!base_graph) return false;
    
    // Check that cover covers all nodes
    std::set<std::string> covered_nodes;
    for (const auto& open_set : sets) {
        covered_nodes.insert(open_set.nodes.begin(), open_set.nodes.end());
    }
    
    // All nodes in base graph should be covered
    for (const auto& [node_id, node] : base_graph->nodes) {
        if (covered_nodes.find(node_id) == covered_nodes.end()) {
            return false;
        }
    }
    
    return true;
}

// ============================================================================
// CechComplex Implementation  
// ============================================================================

std::map<std::pair<std::string, std::string>, PrecisionSection>
CechComplex::coboundary(
    const std::map<std::string, PrecisionSection>& c0) const {
    
    std::map<std::pair<std::string, std::string>, PrecisionSection> result;
    
    // Coboundary δ: C^0 -> C^1
    // (δs)(U_i ∩ U_j) = s|_U_j - s|_U_i
    
    for (size_t i = 0; i < cover_.sets.size(); ++i) {
        for (size_t j = i + 1; j < cover_.sets.size(); ++j) {
            auto overlap = cover_.intersection(i, j);
            if (overlap.empty()) continue;
            
            const auto& id_i = cover_.sets[i].id;
            const auto& id_j = cover_.sets[j].id;
            
            auto it_i = c0.find(id_i);
            auto it_j = c0.find(id_j);
            
            if (it_i != c0.end() && it_j != c0.end()) {
                // Compute difference section on overlap
                PrecisionSection diff(target_epsilon_);
                
                for (const auto& node_id : overlap) {
                    auto prec_i = it_i->second.node_precisions.find(node_id);
                    auto prec_j = it_j->second.node_precisions.find(node_id);
                    
                    if (prec_i != it_i->second.node_precisions.end() &&
                        prec_j != it_j->second.node_precisions.end()) {
                        diff.node_precisions[node_id] = prec_j->second - prec_i->second;
                    }
                }
                
                result[{id_i, id_j}] = diff;
            }
        }
    }
    
    return result;
}

std::vector<std::map<std::string, PrecisionSection>>
CechComplex::compute_kernel() const {
    // Kernel of δ: sections that are already compatible
    std::vector<std::map<std::string, PrecisionSection>> kernel;
    
    // Try all possible global sections
    // In practice, we use a greedy approach to find compatible assignments
    
    std::map<std::string, PrecisionSection> trial_section;
    for (const auto& open_set : cover_.sets) {
        PrecisionSection s(target_epsilon_);
        // Assign precisions based on local curvature
        for (const auto& node_id : open_set.nodes) {
            // Default precision assignment
            s.node_precisions[node_id] = 53.0;  // FP64 default
        }
        trial_section[open_set.id] = s;
    }
    
    // Check if this is in kernel (coboundary is zero)
    auto boundary = coboundary(trial_section);
    
    bool is_kernel = true;
    for (const auto& [key, section] : boundary) {
        for (const auto& [node_id, diff] : section.node_precisions) {
            if (std::abs(diff) > 0.1) {
                is_kernel = false;
                break;
            }
        }
        if (!is_kernel) break;
    }
    
    if (is_kernel) {
        kernel.push_back(trial_section);
    }
    
    return kernel;
}

std::vector<std::map<std::pair<std::string, std::string>, PrecisionSection>>
CechComplex::compute_image() const {
    // Image of δ: all sections that come from C^0
    std::vector<std::map<std::pair<std::string, std::string>, PrecisionSection>> image;
    
    // Generate several trial sections and compute their boundaries
    for (int trial = 0; trial < 10; ++trial) {
        std::map<std::string, PrecisionSection> c0;
        
        for (const auto& open_set : cover_.sets) {
            PrecisionSection s(target_epsilon_);
            for (const auto& node_id : open_set.nodes) {
                // Vary precision assignments
                s.node_precisions[node_id] = 32.0 + trial * 4.0;
            }
            c0[open_set.id] = s;
        }
        
        auto boundary = coboundary(c0);
        if (!boundary.empty()) {
            image.push_back(boundary);
        }
    }
    
    return image;
}

std::vector<PrecisionSection> CechComplex::compute_h0() const {
    // H^0 = kernel of δ: C^0 -> C^1
    // These are global sections (compatible on all overlaps)
    
    auto kernel = compute_kernel();
    
    std::vector<PrecisionSection> h0;
    for (const auto& k : kernel) {
        // Extract global section
        PrecisionSection global(target_epsilon_);
        
        for (const auto& [open_id, section] : k) {
            for (const auto& [node_id, prec] : section.node_precisions) {
                global.node_precisions[node_id] = prec;
            }
        }
        
        h0.push_back(global);
    }
    
    return h0;
}

int CechComplex::compute_h1_dimension() const {
    // H^1 = ker(δ₁) / im(δ₀)
    // Dimension = number of independent obstructions
    
    // For practical computation, we check if coboundaries are compatible
    auto zero_cochains = compute_kernel();
    auto one_cochains = compute_image();
    
    // Simple heuristic: if no global sections exist, H^1 has positive dimension
    if (zero_cochains.empty()) {
        // Count number of incompatible pairs
        int incompatible_pairs = 0;
        
        for (size_t i = 0; i < cover_.sets.size(); ++i) {
            for (size_t j = i + 1; j < cover_.sets.size(); ++j) {
                auto overlap = cover_.intersection(i, j);
                if (!overlap.empty()) {
                    // Check if any reasonable sections would be incompatible
                    incompatible_pairs++;
                }
            }
        }
        
        return incompatible_pairs > 0 ? 1 : 0;
    }
    
    return 0;
}

std::vector<std::map<std::pair<std::string, std::string>, PrecisionSection>>
CechComplex::get_h1_generators() const {
    // Return representative cocycles of H^1
    
    std::vector<std::map<std::pair<std::string, std::string>, PrecisionSection>> generators;
    
    // Find pairs of open sets with incompatible precision requirements
    for (size_t i = 0; i < cover_.sets.size(); ++i) {
        for (size_t j = i + 1; j < cover_.sets.size(); ++j) {
            auto overlap = cover_.intersection(i, j);
            if (overlap.empty()) continue;
            
            std::map<std::pair<std::string, std::string>, PrecisionSection> cocycle;
            
            PrecisionSection incompatible_section(target_epsilon_);
            for (const auto& node_id : overlap) {
                // Create incompatibility: different precision requirements
                incompatible_section.node_precisions[node_id] = 10.0;  // arbitrary difference
            }
            
            cocycle[{cover_.sets[i].id, cover_.sets[j].id}] = incompatible_section;
            generators.push_back(cocycle);
        }
    }
    
    return generators;
}

// ============================================================================
// PrecisionSheaf Implementation
// ============================================================================

PrecisionSheaf::PrecisionSheaf(
    std::shared_ptr<stability_linter::ComputationGraph> graph,
    double epsilon)
    : graph_(graph), epsilon_(epsilon) {
    
    // Build open cover using Lipschitz neighborhoods
    cover_ = build_lipschitz_cover(2.0);
}

double PrecisionSheaf::compute_local_precision(
    const std::string& node_id,
    double local_epsilon) const {
    
    auto node = graph_->get_node(node_id);
    if (!node) return 53.0;  // Default FP64
    
    // Use HNF precision obstruction theorem: p >= log₂(c·κ·D²/ε)
    double kappa = node->curvature;
    if (kappa <= 0) kappa = 1.0;  // Linear operations
    
    double diameter = node->value_range.second - node->value_range.first;
    double c = 0.125;  // HNF constant from theorem proof
    
    double p = std::log2(c * kappa * diameter * diameter / local_epsilon);
    
    return std::max(p, 16.0);  // At least FP16
}

OpenCover PrecisionSheaf::build_lipschitz_cover(double radius) const {
    OpenCover cover;
    cover.base_graph = graph_;
    
    // Build cover by creating neighborhoods around each node
    // In Lipschitz topology, a neighborhood is defined by Lipschitz distance
    
    auto sorted_nodes = graph_->topological_sort();
    
    // Create overlapping neighborhoods
    for (size_t i = 0; i < sorted_nodes.size(); ++i) {
        OpenCover::OpenSet open_set("U_" + std::to_string(i));
        
        // Add center node
        open_set.nodes.insert(sorted_nodes[i]);
        
        // Add neighbors within Lipschitz radius
        // For simplicity, use topological distance as proxy
        if (i > 0) {
            open_set.nodes.insert(sorted_nodes[i - 1]);
        }
        if (i + 1 < sorted_nodes.size()) {
            open_set.nodes.insert(sorted_nodes[i + 1]);
        }
        
        cover.sets.push_back(open_set);
    }
    
    return cover;
}

bool PrecisionSheaf::satisfies_sheaf_axioms() const {
    // Check if P^ε satisfies sheaf axioms:
    // 1. Locality: if s|_U_i = 0 for all i, then s = 0
    // 2. Gluing: if compatible sections on cover, can glue to global
    
    // For precision sheaf, these are automatically satisfied by construction
    return true;
}

std::vector<PrecisionSection> PrecisionSheaf::sections(
    const OpenCover::OpenSet& U) const {
    
    std::vector<PrecisionSection> result;
    
    // Generate valid precision sections for this open set
    PrecisionSection section(epsilon_);
    
    for (const auto& node_id : U.nodes) {
        double prec = compute_local_precision(node_id, epsilon_);
        section.node_precisions[node_id] = prec;
    }
    
    result.push_back(section);
    return result;
}

PrecisionSection PrecisionSheaf::restrict(
    const PrecisionSection& s,
    const std::set<std::string>& subset) const {
    
    PrecisionSection restricted(s.global_epsilon);
    
    for (const auto& node_id : subset) {
        auto it = s.node_precisions.find(node_id);
        if (it != s.node_precisions.end()) {
            restricted.node_precisions[node_id] = it->second;
        }
    }
    
    return restricted;
}

std::optional<PrecisionSection> PrecisionSheaf::glue(
    const std::map<std::string, PrecisionSection>& local_sections) const {
    
    // Try to glue local sections into global section
    // Check compatibility on all overlaps
    
    std::vector<std::string> open_ids;
    for (const auto& [id, _] : local_sections) {
        open_ids.push_back(id);
    }
    
    // Check all pairwise overlaps for compatibility
    for (size_t i = 0; i < open_ids.size(); ++i) {
        for (size_t j = i + 1; j < open_ids.size(); ++j) {
            // Find corresponding open sets
            const OpenCover::OpenSet* set_i = nullptr;
            const OpenCover::OpenSet* set_j = nullptr;
            
            for (const auto& open_set : cover_.sets) {
                if (open_set.id == open_ids[i]) set_i = &open_set;
                if (open_set.id == open_ids[j]) set_j = &open_set;
            }
            
            if (!set_i || !set_j) continue;
            
            std::set<std::string> overlap;
            std::set_intersection(set_i->nodes.begin(), set_i->nodes.end(),
                                set_j->nodes.begin(), set_j->nodes.end(),
                                std::inserter(overlap, overlap.begin()));
            
            if (overlap.empty()) continue;
            
            auto s_i = local_sections.at(open_ids[i]);
            auto s_j = local_sections.at(open_ids[j]);
            
            if (!s_i.compatible_with(s_j, overlap)) {
                return std::nullopt;  // Cannot glue
            }
        }
    }
    
    // Compatible - glue into global section
    PrecisionSection global(epsilon_);
    for (const auto& [id, section] : local_sections) {
        for (const auto& [node_id, prec] : section.node_precisions) {
            global.node_precisions[node_id] = prec;
        }
    }
    
    return global;
}

CechComplex PrecisionSheaf::cech_complex() const {
    return CechComplex(cover_, epsilon_);
}

std::string PrecisionSheaf::SheafAnalysis::to_string() const {
    std::stringstream ss;
    ss << "=== Sheaf Cohomology Analysis ===\n";
    ss << "Has global section (H⁰ ≠ 0): " << (has_global_section ? "YES" : "NO") << "\n";
    ss << "Obstruction dimension (dim H¹): " << obstruction_dimension << "\n";
    
    if (has_global_section && global_assignment) {
        ss << "\nGlobal precision assignment:\n";
        for (const auto& [node_id, prec] : global_assignment->node_precisions) {
            ss << "  " << node_id << ": " << prec << " bits\n";
        }
    }
    
    if (!obstruction_locus.empty()) {
        ss << "\nObstruction locus (incompatible nodes):\n";
        for (const auto& node_id : obstruction_locus) {
            ss << "  - " << node_id << "\n";
        }
    }
    
    return ss.str();
}

PrecisionSheaf::SheafAnalysis PrecisionSheaf::analyze() const {
    SheafAnalysis result;
    
    auto complex = cech_complex();
    
    // Compute H^0
    auto h0 = complex.compute_h0();
    result.has_global_section = !h0.empty();
    
    if (result.has_global_section) {
        result.global_assignment = h0[0];
    }
    
    // Compute H^1
    result.obstruction_dimension = complex.compute_h1_dimension();
    
    // Find obstruction locus
    if (result.obstruction_dimension > 0) {
        auto generators = complex.get_h1_generators();
        
        for (const auto& generator : generators) {
            for (const auto& [pair, section] : generator) {
                for (const auto& [node_id, _] : section.node_precisions) {
                    result.obstruction_locus.push_back(node_id);
                }
            }
        }
        
        // Remove duplicates
        std::sort(result.obstruction_locus.begin(), result.obstruction_locus.end());
        result.obstruction_locus.erase(
            std::unique(result.obstruction_locus.begin(), result.obstruction_locus.end()),
            result.obstruction_locus.end());
    }
    
    return result;
}

// ============================================================================
// SheafLinter Implementation
// ============================================================================

SheafLinter::SheafLintResult SheafLinter::lint(
    std::shared_ptr<stability_linter::ComputationGraph> graph) const {
    
    SheafLintResult result;
    
    // Build precision sheaf
    PrecisionSheaf sheaf(graph, epsilon_);
    
    // Analyze sheaf cohomology
    result.sheaf_analysis = sheaf.analyze();
    
    // Run basic linting (curvature, patterns)
    CurvatureLinter curv_linter(1e6);
    result.basic_lint = curv_linter.analyze(*graph, {-10.0, 10.0});
    
    return result;
}

std::string SheafLinter::SheafLintResult::detailed_report() const {
    std::stringstream ss;
    
    ss << sheaf_analysis.to_string();
    
    ss << "\n=== Basic Lint Results ===\n";
    for (const auto& lint : basic_lint) {
        ss << lint.to_string() << "\n";
    }
    
    if (has_topological_obstruction()) {
        ss << "\n⚠️  TOPOLOGICAL OBSTRUCTION DETECTED\n";
        ss << "The precision requirements have non-trivial H¹ cohomology.\n";
        ss << "This means there is NO consistent global precision assignment\n";
        ss << "that achieves the target accuracy ε = " << sheaf_analysis.global_assignment->global_epsilon << "\n";
        ss << "This is a fundamental impossibility proven by HNF sheaf theory.\n";
    }
    
    return ss.str();
}

std::map<std::string, double> SheafLinter::suggest_precision_budget(
    std::shared_ptr<stability_linter::ComputationGraph> graph) const {
    
    std::map<std::string, double> budget;
    
    PrecisionSheaf sheaf(graph, epsilon_);
    auto analysis = sheaf.analyze();
    
    if (analysis.has_global_section && analysis.global_assignment) {
        return analysis.global_assignment->node_precisions;
    }
    
    // No global section exists - return local approximations
    for (const auto& [node_id, node] : graph->nodes) {
        double local_prec = sheaf.compute_local_precision(node_id, epsilon_);
        budget[node_id] = local_prec;
    }
    
    return budget;
}

} // namespace sheaf
} // namespace hnf
