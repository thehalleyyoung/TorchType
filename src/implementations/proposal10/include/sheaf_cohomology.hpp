#pragma once

#include "stability_linter.hpp"
#include <map>
#include <set>
#include <vector>
#include <memory>
#include <optional>

/**
 * Sheaf Cohomology for Precision Analysis
 * 
 * Based on HNF Section 4.3: "The precision constraints form a sheaf over
 * the space of computations, and this sheaf carries homotopy-theoretic
 * invariants that classify when computations can be accurately implemented."
 * 
 * This module implements:
 * 1. Precision presheaf P_G^ε construction (HNF Definition 4.5)
 * 2. Sheaf cohomology H¹(G; P_G^ε) computation (HNF Theorem 4.7)
 * 3. Obstruction detection for global precision assignment
 */

namespace hnf {
namespace sheaf {

/**
 * Precision Presheaf Section
 * 
 * A section assigns a precision requirement (in bits) to each node
 * in a subgraph, subject to compatibility conditions.
 */
struct PrecisionSection {
    std::map<std::string, double> node_precisions;  // node_id -> required bits
    double global_epsilon;  // Target accuracy
    
    PrecisionSection(double eps = 1e-6) : global_epsilon(eps) {}
    
    // Check if this section is compatible with another on their overlap
    bool compatible_with(const PrecisionSection& other,
                        const std::set<std::string>& overlap_nodes) const;
    
    // Merge two compatible sections
    std::optional<PrecisionSection> merge(const PrecisionSection& other,
                                         const std::set<std::string>& overlap) const;
};

/**
 * Open Cover of Computation Graph
 * 
 * A collection of overlapping subgraphs forming an open cover
 * in the Lipschitz topology (HNF Definition 4.4)
 */
struct OpenCover {
    struct OpenSet {
        std::string id;
        std::set<std::string> nodes;
        std::shared_ptr<stability_linter::ComputationGraph> subgraph;
        
        OpenSet(const std::string& _id) : id(_id) {}
    };
    
    std::vector<OpenSet> sets;
    std::shared_ptr<stability_linter::ComputationGraph> base_graph;
    
    // Get intersection of two open sets
    std::set<std::string> intersection(size_t i, size_t j) const;
    
    // Check if this is a valid open cover
    bool is_valid_cover() const;
};

/**
 * Cech Cohomology Complex
 * 
 * Computes the Čech cohomology groups for the precision sheaf.
 * H⁰ = global sections, H¹ = obstructions to gluing
 */
class CechComplex {
private:
    OpenCover cover_;
    double target_epsilon_;
    
    // C^0: sections on individual opens
    std::map<std::string, PrecisionSection> zero_cochains_;
    
    // C^1: sections on pairwise intersections
    std::map<std::pair<std::string, std::string>, PrecisionSection> one_cochains_;
    
    // Coboundary map δ: C^0 -> C^1
    std::map<std::pair<std::string, std::string>, PrecisionSection> 
    coboundary(const std::map<std::string, PrecisionSection>& c0) const;
    
    // Compute kernel of δ
    std::vector<std::map<std::string, PrecisionSection>> compute_kernel() const;
    
    // Compute image of δ  
    std::vector<std::map<std::pair<std::string, std::string>, PrecisionSection>> 
    compute_image() const;
    
public:
    CechComplex(const OpenCover& cover, double epsilon)
        : cover_(cover), target_epsilon_(epsilon) {}
    
    // Compute H^0(G; P^ε) - global precision assignments
    std::vector<PrecisionSection> compute_h0() const;
    
    // Compute H^1(G; P^ε) - obstructions to global assignment
    // Returns dimension of obstruction space
    int compute_h1_dimension() const;
    
    // Get representative cocycles of H^1
    std::vector<std::map<std::pair<std::string, std::string>, PrecisionSection>>
    get_h1_generators() const;
};

/**
 * Precision Sheaf
 * 
 * The sheaf P_G^ε of precision requirements over computation graph G.
 * HNF Definition 4.5: "For each open U ⊆ G, P^ε(U) is the set of
 * precision assignments that achieve ε-accuracy on U."
 */
class PrecisionSheaf {
private:
    std::shared_ptr<stability_linter::ComputationGraph> graph_;
    double epsilon_;
    OpenCover cover_;
    
public:
    // Compute local precision requirement for a node
    double compute_local_precision(const std::string& node_id,
                                   double local_epsilon) const;
    
    // Build open cover using Lipschitz balls
    OpenCover build_lipschitz_cover(double radius) const;
    
public:
    PrecisionSheaf(std::shared_ptr<stability_linter::ComputationGraph> graph,
                   double epsilon);
    
    // Check if sheaf axioms are satisfied
    bool satisfies_sheaf_axioms() const;
    
    // Compute sections over an open set
    std::vector<PrecisionSection> sections(const OpenCover::OpenSet& U) const;
    
    // Restriction map: sections on U -> sections on V ⊂ U
    PrecisionSection restrict(const PrecisionSection& s,
                             const std::set<std::string>& subset) const;
    
    // Gluing: if compatible sections on cover, glue to global
    std::optional<PrecisionSection> glue(
        const std::map<std::string, PrecisionSection>& local_sections) const;
    
    // Compute cohomology
    CechComplex cech_complex() const;
    
    // Main result: check if global precision assignment exists
    struct SheafAnalysis {
        bool has_global_section;  // H^0 ≠ 0
        int obstruction_dimension;  // dim H^1
        std::optional<PrecisionSection> global_assignment;  // if exists
        std::vector<std::string> obstruction_locus;  // nodes causing obstructions
        
        std::string to_string() const;
    };
    
    SheafAnalysis analyze() const;
};

/**
 * Sheaf-Theoretic Linter
 * 
 * Extends the basic stability linter with sheaf cohomology analysis.
 * Detects when precision requirements have topological obstructions.
 */
class SheafLinter {
private:
    double epsilon_;
    double lipschitz_radius_;
    
public:
    SheafLinter(double epsilon = 1e-6, double radius = 1.0)
        : epsilon_(epsilon), lipschitz_radius_(radius) {}
    
    struct SheafLintResult {
        PrecisionSheaf::SheafAnalysis sheaf_analysis;
        std::vector<stability_linter::LintResult> basic_lint;
        
        // Enhanced diagnostics
        bool has_topological_obstruction() const {
            return sheaf_analysis.obstruction_dimension > 0;
        }
        
        std::string detailed_report() const;
    };
    
    SheafLintResult lint(std::shared_ptr<stability_linter::ComputationGraph> graph) const;
    
    // Suggest precision budget allocation that respects sheaf structure
    std::map<std::string, double> suggest_precision_budget(
        std::shared_ptr<stability_linter::ComputationGraph> graph) const;
};

} // namespace sheaf
} // namespace hnf
