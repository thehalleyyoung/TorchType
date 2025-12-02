#ifndef ADVANCED_SHEAF_THEORY_H
#define ADVANCED_SHEAF_THEORY_H

#include <vector>
#include <map>
#include <set>
#include <memory>
#include <functional>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include "computation_graph.h"
#include "precision_sheaf.h"

// Advanced sheaf-theoretic constructions for precision analysis
// Based on HNF Paper Section 4.4 and algebraic topology

namespace hnf {

// Forward declarations - use types from sheaf namespace
using namespace sheaf;

// Spectral Sequence for computing sheaf cohomology via filtrations
class SpectralSequence {
public:
    struct Page {
        int p, q;  // bidegree
        std::map<std::pair<int,int>, Eigen::MatrixXd> E;  // E^{p,q}_r
        std::map<std::pair<int,int>, Eigen::MatrixXd> d;  // differentials
        
        Page(int r_val) : r(r_val) {}
        
        int r;  // page number
        
        // Get E^{p,q}_r
        Eigen::MatrixXd get_entry(int p, int q) const {
            auto it = E.find({p, q});
            return it != E.end() ? it->second : Eigen::MatrixXd();
        }
        
        // Set E^{p,q}_r
        void set_entry(int p, int q, const Eigen::MatrixXd& val) {
            E[{p, q}] = val;
        }
        
        // Compute next page via homology
        std::shared_ptr<Page> compute_next() const;
    };
    
    SpectralSequence(const ComputationGraph& graph, 
                    const std::vector<std::set<std::string>>& filtration);
    
    // Compute E_2 page from Čech complex
    void compute_E2();
    
    // Converge spectral sequence
    void converge(int max_pages = 10);
    
    // Extract limit H^n
    Eigen::MatrixXd get_limit_cohomology(int n) const;
    
    // Detect if mixed precision is forced by higher cohomology
    bool has_cohomological_obstruction() const;
    
    // Get critical nodes from spectral sequence
    std::vector<std::string> get_critical_nodes() const;
    
private:
    const ComputationGraph& graph_;
    std::vector<std::set<std::string>> filtration_;
    std::vector<std::shared_ptr<Page>> pages_;
    bool converged_;
    int convergence_page_;
    
    // Build filtration spectral sequence
    void build_filtration_spectral_sequence();
};

// Derived Functors for computing sheaf cohomology
class DerivedFunctorComputer {
public:
    DerivedFunctorComputer(const ComputationGraph& graph) : graph_(graph) {}
    
    // Right derived functors of global sections functor
    // R^i Γ(G, P) = H^i(G, P)
    std::vector<Eigen::MatrixXd> compute_derived_functors(
        const PrecisionSheaf& sheaf,
        int max_degree = 3
    );
    
    // Compute using injective resolutions
    std::vector<Eigen::MatrixXd> via_injective_resolution(
        const PrecisionSheaf& sheaf
    );
    
    // Compute using Čech resolution
    std::vector<Eigen::MatrixXd> via_cech_resolution(
        const PrecisionSheaf& sheaf
    );
    
    // Verify they agree (fundamental theorem)
    bool verify_agreement(double tol = 1e-10);
    
private:
    const ComputationGraph& graph_;
    std::vector<Eigen::MatrixXd> injective_result_;
    std::vector<Eigen::MatrixXd> cech_result_;
};

// Descent theory for precision sheaves
class DescentTheory {
public:
    struct DescentDatum {
        // Data on double overlaps U_i ∩ U_j
        std::map<std::pair<int,int>, Eigen::MatrixXd> data;
        
        // Cocycle condition: φ_ij ∘ φ_jk = φ_ik on triple overlaps
        std::map<std::tuple<int,int,int>, bool> cocycle_satisfied;
        
        bool is_effective() const;  // Can be descended to global data
    };
    
    DescentTheory(const ComputationGraph& graph) : graph_(graph) {}
    
    // Check if precision sheaf satisfies descent
    bool satisfies_descent(const PrecisionSheaf& sheaf) const;
    
    // Effective descent: reconstruct global from local
    std::map<std::string, int> descend(const DescentDatum& datum);
    
    // Obstruction to descent lives in H^2
    Eigen::MatrixXd compute_descent_obstruction(const DescentDatum& datum);
    
    // Faithfully flat descent for precision requirements
    bool is_faithfully_flat_cover(const std::vector<std::set<std::string>>& cover) const;
    
private:
    const ComputationGraph& graph_;
    
    bool check_cocycle_condition(
        const DescentDatum& datum,
        int i, int j, int k
    ) const;
};

// Sheafification functor
class Sheafification {
public:
    Sheafification(const ComputationGraph& graph) : graph_(graph) {}
    
    // Given presheaf P, compute its sheafification P^+
    // This is left adjoint to the forgetful functor
    PrecisionSheaf sheafify(const PrecisionSheaf& presheaf);
    
    // Check if already a sheaf (gluing axiom)
    bool is_sheaf(const PrecisionSheaf& P) const;
    
    // Universal property: map from P to P^+ is initial
    bool verify_universal_property(const PrecisionSheaf& P) const;
    
private:
    const ComputationGraph& graph_;
    
    // Gluing axiom: compatible sections glue uniquely
    bool check_gluing(
        const PrecisionSheaf& P,
        const std::vector<OpenSet>& cover
    ) const;
};

// Higher direct images for compositions
class HigherDirectImage {
public:
    HigherDirectImage(const ComputationGraph& source,
                     const ComputationGraph& target,
                     const std::function<std::string(std::string)>& morphism)
        : source_(source), target_(target), f_(morphism) {}
    
    // R^i f_* (sheaf on source) → sheaf on target
    PrecisionSheaf compute(const PrecisionSheaf& F, int i);
    
    // Leray spectral sequence: E_2^{p,q} = H^p(target, R^q f_* F) => H^{p+q}(source, F)
    SpectralSequence leray_spectral_sequence(const PrecisionSheaf& F);
    
private:
    const ComputationGraph& source_;
    const ComputationGraph& target_;
    std::function<std::string(std::string)> f_;
};

// Grothendieck topology for precision requirements
class GrothendieckTopology {
public:
    struct Sieve {
        std::string object;  // Node in computation graph
        std::set<std::string> covering_morphisms;  // Edges covering this node
        
        bool is_covering(const ComputationGraph& graph) const;
    };
    
    GrothendieckTopology(const ComputationGraph& graph) : graph_(graph) {}
    
    // Canonical topology: covers are jointly surjective families
    std::vector<Sieve> canonical_topology() const;
    
    // Check if collection of sieves forms a Grothendieck topology
    bool is_grothendieck_topology(const std::vector<Sieve>& sieves) const;
    
    // Sheaves in this topology
    bool is_sheaf_in_topology(const PrecisionSheaf& P,
                             const std::vector<Sieve>& topology) const;
    
private:
    const ComputationGraph& graph_;
    
    bool pullback_stability(const Sieve& S) const;
    bool local_character(const Sieve& S) const;
};

// Étale cohomology for precision analysis
class EtaleCohomology {
public:
    EtaleCohomology(const ComputationGraph& graph) : graph_(graph) {}
    
    // Étale site: smooth covers with discrete fibers
    struct EtaleCover {
        std::string base;
        std::vector<std::string> cover_elements;
        std::map<std::string, int> fiber_sizes;  // Should all be 1 for étale
        
        bool is_etale() const;
    };
    
    // Compute étale cohomology H^i_et(G, P)
    std::vector<Eigen::MatrixXd> compute_etale_cohomology(
        const PrecisionSheaf& sheaf,
        int max_degree = 3
    );
    
    // Compare with Zariski cohomology
    bool verify_comparison_theorem(const PrecisionSheaf& sheaf);
    
private:
    const ComputationGraph& graph_;
};

// Cup product structure on cohomology
class CupProduct {
public:
    CupProduct(const ComputationGraph& graph) : graph_(graph) {}
    
    // H^p × H^q → H^{p+q}
    Eigen::MatrixXd compute_cup_product(
        const Eigen::MatrixXd& alpha,  // p-cocycle
        const Eigen::MatrixXd& beta,   // q-cocycle
        int p, int q
    ) const;
    
    // Graded ring structure
    struct CohomologyRing {
        std::map<int, Eigen::MatrixXd> generators;  // One per degree
        std::map<std::tuple<int,int,int>, Eigen::MatrixXd> products;  // (i,j) → i+j
        
        // Ring axioms
        bool verify_associativity() const;
        bool verify_commutativity() const;
        bool verify_unit() const;
    };
    
    CohomologyRing compute_ring_structure(const PrecisionSheaf& sheaf);
    
    // Poincaré duality for precision graphs
    bool verify_poincare_duality() const;
    
private:
    const ComputationGraph& graph_;
    
    // Alexander-Whitney map for computing cup products
    Eigen::MatrixXd alexander_whitney_map(
        const Eigen::MatrixXd& alpha,
        const Eigen::MatrixXd& beta
    ) const;
};

// Verdier duality for precision sheaves
class VerdierDuality {
public:
    VerdierDuality(const ComputationGraph& graph) : graph_(graph) {}
    
    // Dualizing complex
    struct DualizingComplex {
        std::vector<Eigen::MatrixXd> components;
        std::vector<Eigen::MatrixXd> differentials;
        
        int dimension() const { return components.size(); }
    };
    
    // Compute dualizing complex for computation graph
    DualizingComplex compute_dualizing_complex();
    
    // Duality isomorphism H^i(G, F) ≅ H^{n-i}(G, F^*)^*
    bool verify_duality(const PrecisionSheaf& sheaf, int i);
    
    // Dual sheaf
    PrecisionSheaf compute_dual_sheaf(const PrecisionSheaf& sheaf);
    
private:
    const ComputationGraph& graph_;
};

// Local-to-global principles for precision
class LocalToGlobalPrinciple {
public:
    LocalToGlobalPrinciple(const ComputationGraph& graph) : graph_(graph) {}
    
    // If precision works locally everywhere, does it work globally?
    struct LocalGlobalResult {
        bool local_existence;   // ∃ local precision assignments
        bool global_existence;  // ∃ global precision assignment
        Eigen::MatrixXd obstruction;  // Lives in H^1
        
        std::string diagnosis() const;
    };
    
    LocalGlobalResult analyze(double target_accuracy);
    
    // Hasse principle for precision: local-global fails iff H^1 ≠ 0
    bool satisfies_hasse_principle(double target_accuracy);
    
    // Find minimal obstructions
    std::vector<std::pair<std::string, std::string>> find_minimal_obstructions();
    
private:
    const ComputationGraph& graph_;
};

// Perverse sheaves for precision analysis
class PerverseSheaves {
public:
    PerverseSheaves(const ComputationGraph& graph) : graph_(graph) {}
    
    // t-structure on derived category
    struct TStructure {
        std::function<bool(int)> is_in_D_leq_0;
        std::function<bool(int)> is_in_D_geq_0;
        
        bool verify_axioms() const;
    };
    
    // Perverse t-structure
    TStructure perverse_t_structure();
    
    // IC sheaf (intersection cohomology sheaf) for stratified precision
    PrecisionSheaf compute_IC_sheaf();
    
    // Decomposition theorem
    std::vector<PrecisionSheaf> decomposition_theorem();
    
private:
    const ComputationGraph& graph_;
};

} // namespace hnf

#endif // ADVANCED_SHEAF_THEORY_H
