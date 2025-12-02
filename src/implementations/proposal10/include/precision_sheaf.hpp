#pragma once

#include "stability_linter.hpp"
#include <vector>
#include <map>
#include <string>
#include <memory>
#include <set>

namespace hnf {
namespace stability_linter {

// Sheaf-theoretic precision analysis from HNF Section 4.4
class PrecisionSheaf {
public:
    // A precision assignment assigns required bits to each node
    using PrecisionAssignment = std::map<std::string, int>;
    
    // An open set in the computation graph (a connected subgraph)
    struct OpenSet {
        std::string name;
        std::set<std::string> nodes;
        std::vector<std::string> boundary_nodes;  // nodes with edges to outside
    };
    
    // Local precision section over an open set
    struct LocalSection {
        OpenSet open_set;
        PrecisionAssignment assignment;
        bool is_consistent;  // satisfies local composition bounds
        double curvature_bound;
    };
    
    // A covering of the computation graph
    struct Covering {
        std::vector<OpenSet> sets;
        std::map<std::pair<std::string, std::string>, std::set<std::string>> overlaps;
    };
    
    // Sheaf cohomology group H^1(G, P^ε)
    struct CohomologyGroup {
        int dimension;  // obstruction space dimension
        std::vector<PrecisionAssignment> cocycles;  // elements of H^1
        bool has_global_section;  // true if H^1 = 0
        std::string obstruction_description;
    };
    
    explicit PrecisionSheaf(std::shared_ptr<ComputationGraph> graph);
    
    // Build a covering of the computation graph
    Covering build_covering(int max_set_size = 5);
    
    // Compute local precision sections for each open set
    std::vector<LocalSection> compute_local_sections(
        const Covering& covering,
        double target_accuracy
    );
    
    // Check if local sections are compatible on overlaps
    struct CompatibilityCheck {
        bool compatible;
        std::vector<std::pair<std::string, std::string>> conflicts;
        std::map<std::pair<std::string, std::string>, int> precision_gaps;
    };
    
    CompatibilityCheck check_compatibility(
        const std::vector<LocalSection>& sections,
        const Covering& covering
    );
    
    // Compute sheaf cohomology H^1
    CohomologyGroup compute_h1_cohomology(
        const Covering& covering,
        double target_accuracy
    );
    
    // Find global precision assignment (if exists)
    struct GlobalSection {
        bool exists;
        PrecisionAssignment assignment;
        double total_cost;  // sum of bits required
        std::vector<std::string> obstructions;  // empty if exists
    };
    
    GlobalSection find_global_section(double target_accuracy);
    
    // Optimize precision assignment using sheaf descent
    struct OptimizedAssignment {
        PrecisionAssignment assignment;
        double total_bits;
        double certified_accuracy;
        bool is_minimal;  // locally minimal in bit count
        std::string optimization_log;
    };
    
    OptimizedAssignment optimize_precision(
        double target_accuracy,
        int max_iterations = 100
    );
    
    // Visualize sheaf structure (generate description)
    std::string visualize_sheaf_structure(const Covering& covering);
    
private:
    std::shared_ptr<ComputationGraph> graph_;
    
    // Helper: check if precision assignment is locally consistent
    bool is_locally_consistent(
        const PrecisionAssignment& assignment,
        const std::set<std::string>& node_subset
    );
    
    // Helper: compute minimum precision for composition
    int compute_required_precision(
        const std::string& node_id,
        const std::map<std::string, int>& input_precisions,
        double target_accuracy
    );
    
    // Helper: resolve conflicts in overlapping sections
    PrecisionAssignment resolve_conflicts(
        const PrecisionAssignment& section1,
        const PrecisionAssignment& section2,
        const std::set<std::string>& overlap_nodes
    );
};

// Čech cohomology computation for precision sheaf
class CechCohomology {
public:
    // p-cochain: function from p-fold intersections to precision assignments
    using Cochain = std::map<std::vector<std::string>, PrecisionSheaf::PrecisionAssignment>;
    
    // Coboundary operator δ: C^p → C^{p+1}
    static Cochain coboundary(const Cochain& cp, const PrecisionSheaf::Covering& covering);
    
    // Compute kernel of coboundary (cocycles)
    static std::vector<Cochain> compute_cocycles(
        int degree,
        const PrecisionSheaf::Covering& covering
    );
    
    // Compute image of coboundary (coboundaries)
    static std::vector<Cochain> compute_coboundaries(
        int degree,
        const PrecisionSheaf::Covering& covering
    );
    
    // Compute cohomology H^p = ker(δ^p) / im(δ^{p-1})
    static PrecisionSheaf::CohomologyGroup compute_cohomology(
        int degree,
        const PrecisionSheaf::Covering& covering
    );
};

} // namespace stability_linter
} // namespace hnf
