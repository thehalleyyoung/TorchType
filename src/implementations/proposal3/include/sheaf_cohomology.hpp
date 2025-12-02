#pragma once

#include "attention_types.hpp"
#include <torch/torch.h>
#include <vector>
#include <map>
#include <memory>

namespace hnf {
namespace attention {

/**
 * Sheaf Cohomology for Precision Analysis
 * 
 * From HNF Paper Section 4 (Precision Sheaf):
 * 
 * Over a computation graph G, precision requirements form a presheaf P_G^ε.
 * The failure of this presheaf to be a sheaf—measured by its cohomology
 * H^1(G; P_G^ε)—detects obstructions to uniform precision assignment.
 * 
 * This implementation:
 * 1. Constructs the precision presheaf over a transformer architecture
 * 2. Computes sheaf cohomology groups H^0 and H^1
 * 3. Detects global obstructions to precision assignment
 * 4. Provides certificates for precision impossibility
 */

/**
 * Represents a vertex (computation node) in the graph
 */
struct ComputationVertex {
    std::string name;
    int layer_id;
    int head_id;  // -1 for non-attention nodes
    double local_curvature;
    double required_precision_bits;
    torch::Tensor local_error_bound;
    
    ComputationVertex(const std::string& n, int l, int h = -1)
        : name(n), layer_id(l), head_id(h),
          local_curvature(0.0), required_precision_bits(0.0) {}
};

/**
 * Represents an edge (data flow) between vertices
 */
struct ComputationEdge {
    int from_vertex;
    int to_vertex;
    double lipschitz_constant;
    torch::Tensor error_propagation;
    
    ComputationEdge(int f, int t, double L)
        : from_vertex(f), to_vertex(t), lipschitz_constant(L) {}
};

/**
 * The computation graph representing a transformer architecture
 */
class ComputationGraph {
public:
    ComputationGraph() = default;
    
    int add_vertex(const ComputationVertex& v) {
        vertices_.push_back(v);
        return vertices_.size() - 1;
    }
    
    void add_edge(const ComputationEdge& e) {
        edges_.push_back(e);
        // Build adjacency for efficient traversal
        if (adjacency_.find(e.from_vertex) == adjacency_.end()) {
            adjacency_[e.from_vertex] = std::vector<int>();
        }
        adjacency_[e.from_vertex].push_back(edges_.size() - 1);
    }
    
    const std::vector<ComputationVertex>& vertices() const { return vertices_; }
    const std::vector<ComputationEdge>& edges() const { return edges_; }
    const std::map<int, std::vector<int>>& adjacency() const { return adjacency_; }
    
    std::vector<int> get_outgoing_edges(int vertex_id) const {
        auto it = adjacency_.find(vertex_id);
        if (it != adjacency_.end()) {
            return it->second;
        }
        return std::vector<int>();
    }
    
    size_t num_vertices() const { return vertices_.size(); }
    size_t num_edges() const { return edges_.size(); }
    
private:
    std::vector<ComputationVertex> vertices_;
    std::vector<ComputationEdge> edges_;
    std::map<int, std::vector<int>> adjacency_;  // vertex_id -> edge_ids
};

/**
 * Section of the precision sheaf over an open set
 * 
 * For an open set U in the computation graph, a section assigns
 * precision requirements to each vertex in U, compatible with
 * error propagation along edges.
 */
struct PrecisionSection {
    std::vector<int> vertex_ids;  // Vertices in this open set
    std::vector<double> precision_assignments;  // Bits required at each vertex
    double consistency_error;  // Measures failure to be a section
    
    PrecisionSection() : consistency_error(0.0) {}
};

/**
 * Cohomology computation for the precision sheaf
 */
class SheafCohomology {
public:
    explicit SheafCohomology(const ComputationGraph& graph)
        : graph_(graph) {}
    
    /**
     * Compute H^0: Global sections (consistent precision assignments)
     * 
     * Returns vector of precision assignments that satisfy:
     * For all edges (u,v): precision[v] >= L_{uv} * precision[u] + roundoff
     * 
     * If H^0 is empty, no consistent global precision assignment exists
     * at the given accuracy target.
     */
    std::vector<PrecisionSection> compute_H0(
        double target_accuracy,
        const HardwareModel& hardware
    );
    
    /**
     * Compute H^1: Obstructions to global sections
     * 
     * Non-zero H^1 indicates that local precision requirements cannot
     * be satisfied globally. This is the "precision obstruction group".
     * 
     * Returns: dimension of H^1 and representative cocycles
     */
    struct CohomologyResult {
        int h0_dimension;  // Number of independent global sections
        int h1_dimension;  // Dimension of obstruction space
        std::vector<PrecisionSection> h0_generators;
        std::vector<std::vector<double>> h1_cocycles;  // Obstructions
        double minimal_precision;  // Minimum bits to achieve target accuracy
        std::vector<std::string> obstruction_reasons;
    };
    
    CohomologyResult compute_cohomology(
        double target_accuracy,
        const HardwareModel& hardware
    );
    
    /**
     * Check if a precision assignment forms a valid section
     * 
     * A section is valid if error propagation along all edges
     * maintains the target accuracy.
     */
    double check_section_consistency(
        const std::vector<double>& precision_assignment,
        double target_accuracy
    ) const;
    
    /**
     * Compute the presheaf on an open cover
     * 
     * Constructs local sections on each open set in the cover,
     * then checks gluing conditions.
     */
    std::vector<PrecisionSection> compute_presheaf_on_cover(
        const std::vector<std::vector<int>>& open_cover,
        double target_accuracy,
        const HardwareModel& hardware
    );
    
    /**
     * Detect cycles in precision requirements that cause obstructions
     * 
     * A cycle u1 -> u2 -> ... -> un -> u1 with product of Lipschitz
     * constants > 1 creates an obstruction: precision requirements
     * grow unboundedly.
     */
    std::vector<std::vector<int>> find_obstruction_cycles() const;
    
    /**
     * Minimal precision assignment satisfying all constraints
     * 
     * Solves the linear program:
     * minimize sum(precision[v])
     * subject to: precision[v] >= L_{uv} * precision[u] + roundoff for all edges
     *             precision[v] >= local_requirement[v] for all vertices
     */
    std::vector<double> compute_minimal_precision(
        double target_accuracy,
        const HardwareModel& hardware
    ) const;
    
    /**
     * Visualize the precision sheaf structure
     * 
     * Returns a string representation suitable for graphviz
     */
    std::string to_graphviz() const;
    
private:
    const ComputationGraph& graph_;
    
    // Helper: Compute local precision requirement at a vertex
    double compute_local_precision(
        const ComputationVertex& vertex,
        double target_accuracy,
        const HardwareModel& hardware
    ) const;
    
    // Helper: Propagate precision requirement backward through edge
    double propagate_precision_backward(
        const ComputationEdge& edge,
        double downstream_precision,
        const HardwareModel& hardware
    ) const;
    
    // Helper: Check if a cycle forms a precision obstruction
    bool is_obstruction_cycle(const std::vector<int>& cycle) const;
};

/**
 * Multi-layer precision propagation using sheaf theory
 * 
 * Implements the full HNF framework for compositional error analysis:
 * 
 * From Theorem 3.1 (Stability Composition):
 * Φ_{f_n ∘ ... ∘ f_1}(ε) ≤ Σ_i (Π_{j>i} L_j) · Φ_i(ε_i)
 * 
 * This class:
 * 1. Builds computation graph from transformer architecture
 * 2. Computes curvature at each vertex
 * 3. Propagates precision requirements using sheaf cohomology
 * 4. Detects global obstructions
 * 5. Suggests minimal precision assignment
 */
class MultiLayerPrecisionAnalyzer {
public:
    MultiLayerPrecisionAnalyzer() = default;
    
    /**
     * Build computation graph from transformer layer specifications
     */
    void build_graph_from_transformer(
        int num_layers,
        int num_heads,
        int hidden_dim,
        int seq_len,
        double temperature = 1.0
    );
    
    /**
     * Populate curvature and Lipschitz data from actual tensors
     */
    void populate_from_weights(
        const std::vector<torch::Tensor>& Q_weights,  // Per-layer Q matrices
        const std::vector<torch::Tensor>& K_weights,
        const std::vector<torch::Tensor>& V_weights,
        const std::vector<torch::Tensor>& ffn_weights
    );
    
    /**
     * Run full precision analysis
     */
    SheafCohomology::CohomologyResult analyze_precision(
        double target_accuracy,
        const HardwareModel& hardware
    );
    
    /**
     * Get the computation graph
     */
    const ComputationGraph& graph() const { return graph_; }
    
    /**
     * Generate a detailed report
     */
    struct AnalysisReport {
        SheafCohomology::CohomologyResult cohomology;
        std::vector<std::string> layer_diagnoses;
        std::vector<double> per_layer_precision;
        double total_error_bound;
        bool is_achievable_with_hardware;
        std::vector<std::string> recommendations;
    };
    
    AnalysisReport generate_report(
        double target_accuracy,
        const HardwareModel& hardware
    );
    
private:
    ComputationGraph graph_;
    int num_layers_;
    int num_heads_;
    int hidden_dim_;
    int seq_len_;
    double temperature_;
};

} // namespace attention
} // namespace hnf
