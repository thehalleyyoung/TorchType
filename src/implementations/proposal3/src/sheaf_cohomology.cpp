#include "sheaf_cohomology.hpp"
#include "attention_curvature.hpp"
#include <algorithm>
#include <cmath>
#include <sstream>
#include <limits>
#include <queue>
#include <set>
#include <iostream>

namespace hnf {
namespace attention {

// Compute H^0: Global sections
std::vector<PrecisionSection> SheafCohomology::compute_H0(
    double target_accuracy,
    const HardwareModel& hardware
) {
    std::vector<PrecisionSection> global_sections;
    
    // Try to find a consistent global precision assignment
    // Start with minimal local requirements and propagate forward
    std::vector<double> precision(graph_.num_vertices(), 0.0);
    
    // Initialize with local requirements
    for (size_t i = 0; i < graph_.num_vertices(); ++i) {
        precision[i] = compute_local_precision(
            graph_.vertices()[i],
            target_accuracy,
            hardware
        );
    }
    
    // Forward propagation: ensure downstream vertices have enough precision
    // Iterate until convergence (or detect divergence)
    bool changed = true;
    int max_iterations = 100;
    int iteration = 0;
    
    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;
        
        for (const auto& edge : graph_.edges()) {
            double required_downstream = 
                edge.lipschitz_constant * precision[edge.from_vertex] + 
                hardware.machine_epsilon() * 10.0;  // Roundoff term
            
            if (precision[edge.to_vertex] < required_downstream) {
                precision[edge.to_vertex] = required_downstream;
                changed = true;
            }
        }
    }
    
    // Check if we converged
    if (iteration >= max_iterations) {
        // No global section exists - precision requirements diverge
        return global_sections;  // Empty
    }
    
    // Verify this is actually a valid section
    double consistency = check_section_consistency(precision, target_accuracy);
    
    if (consistency < target_accuracy) {
        // Valid global section found
        PrecisionSection section;
        section.vertex_ids.resize(graph_.num_vertices());
        for (size_t i = 0; i < graph_.num_vertices(); ++i) {
            section.vertex_ids[i] = i;
        }
        section.precision_assignments = precision;
        section.consistency_error = consistency;
        global_sections.push_back(section);
    }
    
    return global_sections;
}

// Compute full cohomology
SheafCohomology::CohomologyResult SheafCohomology::compute_cohomology(
    double target_accuracy,
    const HardwareModel& hardware
) {
    CohomologyResult result;
    result.h0_dimension = 0;
    result.h1_dimension = 0;
    result.minimal_precision = std::numeric_limits<double>::infinity();
    
    // Compute H^0
    auto h0_sections = compute_H0(target_accuracy, hardware);
    result.h0_dimension = h0_sections.size();
    result.h0_generators = h0_sections;
    
    if (h0_sections.empty()) {
        // No global sections exist - compute obstructions
        
        // Find obstruction cycles
        auto cycles = find_obstruction_cycles();
        result.h1_dimension = cycles.size();
        
        for (const auto& cycle : cycles) {
            std::ostringstream oss;
            oss << "Precision obstruction cycle: ";
            double cycle_lipschitz = 1.0;
            
            for (size_t i = 0; i < cycle.size(); ++i) {
                int v1 = cycle[i];
                int v2 = cycle[(i + 1) % cycle.size()];
                
                // Find edge from v1 to v2
                for (const auto& edge : graph_.edges()) {
                    if (edge.from_vertex == v1 && edge.to_vertex == v2) {
                        cycle_lipschitz *= edge.lipschitz_constant;
                        oss << graph_.vertices()[v1].name << " -> ";
                        break;
                    }
                }
            }
            oss << " (product of Lipschitz constants = " << cycle_lipschitz << ")";
            result.obstruction_reasons.push_back(oss.str());
            
            // Create a cocycle representing this obstruction
            std::vector<double> cocycle(graph_.num_vertices(), 0.0);
            for (int v : cycle) {
                cocycle[v] = 1.0;
            }
            result.h1_cocycles.push_back(cocycle);
        }
    }
    
    // Compute minimal precision assignment (even if obstructed, find best effort)
    auto minimal_precision = compute_minimal_precision(target_accuracy, hardware);
    
    if (!minimal_precision.empty()) {
        double max_precision = *std::max_element(
            minimal_precision.begin(),
            minimal_precision.end()
        );
        result.minimal_precision = max_precision;
    }
    
    return result;
}

// Check section consistency
double SheafCohomology::check_section_consistency(
    const std::vector<double>& precision_assignment,
    double target_accuracy
) const {
    double max_inconsistency = 0.0;
    
    // Check each edge
    for (const auto& edge : graph_.edges()) {
        double from_prec = precision_assignment[edge.from_vertex];
        double to_prec = precision_assignment[edge.to_vertex];
        
        // Required precision at destination
        double required = edge.lipschitz_constant * std::pow(2.0, -from_prec);
        
        // Actual precision at destination
        double actual = std::pow(2.0, -to_prec);
        
        // Inconsistency
        double inconsistency = std::abs(actual - required);
        max_inconsistency = std::max(max_inconsistency, inconsistency);
    }
    
    return max_inconsistency;
}

// Find obstruction cycles
std::vector<std::vector<int>> SheafCohomology::find_obstruction_cycles() const {
    std::vector<std::vector<int>> cycles;
    
    // Use DFS to find cycles
    std::vector<bool> visited(graph_.num_vertices(), false);
    std::vector<bool> on_stack(graph_.num_vertices(), false);
    std::vector<int> parent(graph_.num_vertices(), -1);
    
    std::function<void(int)> dfs = [&](int v) {
        visited[v] = true;
        on_stack[v] = true;
        
        auto outgoing = graph_.get_outgoing_edges(v);
        for (int edge_id : outgoing) {
            const auto& edge = graph_.edges()[edge_id];
            int next = edge.to_vertex;
            
            if (!visited[next]) {
                parent[next] = v;
                dfs(next);
            } else if (on_stack[next]) {
                // Found a cycle - trace it back
                std::vector<int> cycle;
                int curr = v;
                while (curr != next && curr != -1) {
                    cycle.push_back(curr);
                    curr = parent[curr];
                }
                cycle.push_back(next);
                
                // Check if it's an obstruction cycle
                if (is_obstruction_cycle(cycle)) {
                    cycles.push_back(cycle);
                }
            }
        }
        
        on_stack[v] = false;
    };
    
    for (size_t v = 0; v < graph_.num_vertices(); ++v) {
        if (!visited[v]) {
            dfs(v);
        }
    }
    
    return cycles;
}

// Check if cycle is an obstruction
bool SheafCohomology::is_obstruction_cycle(const std::vector<int>& cycle) const {
    if (cycle.empty()) return false;
    
    double lipschitz_product = 1.0;
    
    for (size_t i = 0; i < cycle.size(); ++i) {
        int v1 = cycle[i];
        int v2 = cycle[(i + 1) % cycle.size()];
        
        // Find edge from v1 to v2
        bool found = false;
        for (const auto& edge : graph_.edges()) {
            if (edge.from_vertex == v1 && edge.to_vertex == v2) {
                lipschitz_product *= edge.lipschitz_constant;
                found = true;
                break;
            }
        }
        
        if (!found) {
            return false;  // Not a valid cycle in the graph
        }
    }
    
    // Obstruction if product > 1 (precision grows around the cycle)
    return lipschitz_product > 1.0 + 1e-6;
}

// Compute minimal precision
std::vector<double> SheafCohomology::compute_minimal_precision(
    double target_accuracy,
    const HardwareModel& hardware
) const {
    std::vector<double> precision(graph_.num_vertices(), 0.0);
    
    // Initialize with local requirements
    for (size_t i = 0; i < graph_.num_vertices(); ++i) {
        precision[i] = compute_local_precision(
            graph_.vertices()[i],
            target_accuracy,
            hardware
        );
    }
    
    // Use Bellman-Ford-like algorithm to find minimal consistent assignment
    // This is a shortest path problem in log-precision space
    bool changed = true;
    int max_iterations = graph_.num_vertices();
    int iteration = 0;
    
    while (changed && iteration < max_iterations) {
        changed = false;
        iteration++;
        
        for (const auto& edge : graph_.edges()) {
            double required_downstream = propagate_precision_backward(
                edge,
                precision[edge.to_vertex],
                hardware
            );
            
            if (precision[edge.from_vertex] < required_downstream) {
                precision[edge.from_vertex] = required_downstream;
                changed = true;
            }
        }
    }
    
    return precision;
}

// Compute local precision requirement
double SheafCohomology::compute_local_precision(
    const ComputationVertex& vertex,
    double target_accuracy,
    const HardwareModel& hardware
) const {
    // From HNF Theorem 4.1: p >= log2(c * κ * D^2 / ε)
    double curvature = vertex.local_curvature;
    if (curvature < 1e-10) {
        curvature = 1.0;  // Minimum curvature for linear operations
    }
    
    double diameter = 10.0;  // Estimate based on typical attention range
    double c_constant = 1.0;
    
    double precision_bits = std::log2(
        c_constant * curvature * diameter * diameter / target_accuracy
    );
    
    return std::max(precision_bits, 0.0);
}

// Propagate precision backward
double SheafCohomology::propagate_precision_backward(
    const ComputationEdge& edge,
    double downstream_precision,
    const HardwareModel& hardware
) const {
    // From HNF error propagation:
    // error_in = (error_out - roundoff) / L
    // precision_in = -log2(error_in)
    
    double downstream_error = std::pow(2.0, -downstream_precision);
    double roundoff = hardware.machine_epsilon() * 2.0;
    
    double upstream_error = (downstream_error - roundoff) / edge.lipschitz_constant;
    upstream_error = std::max(upstream_error, hardware.machine_epsilon());
    
    return -std::log2(upstream_error);
}

// Compute presheaf on cover
std::vector<PrecisionSection> SheafCohomology::compute_presheaf_on_cover(
    const std::vector<std::vector<int>>& open_cover,
    double target_accuracy,
    const HardwareModel& hardware
) {
    std::vector<PrecisionSection> sections;
    
    for (const auto& open_set : open_cover) {
        PrecisionSection section;
        section.vertex_ids = open_set;
        section.precision_assignments.resize(open_set.size());
        
        // Compute local precision for each vertex in the open set
        for (size_t i = 0; i < open_set.size(); ++i) {
            int v_id = open_set[i];
            section.precision_assignments[i] = compute_local_precision(
                graph_.vertices()[v_id],
                target_accuracy,
                hardware
            );
        }
        
        // Check consistency within this open set
        section.consistency_error = 0.0;
        for (const auto& edge : graph_.edges()) {
            if (std::find(open_set.begin(), open_set.end(), edge.from_vertex) != open_set.end() &&
                std::find(open_set.begin(), open_set.end(), edge.to_vertex) != open_set.end()) {
                
                size_t from_idx = std::find(open_set.begin(), open_set.end(), edge.from_vertex) - open_set.begin();
                size_t to_idx = std::find(open_set.begin(), open_set.end(), edge.to_vertex) - open_set.begin();
                
                double from_error = std::pow(2.0, -section.precision_assignments[from_idx]);
                double to_error = std::pow(2.0, -section.precision_assignments[to_idx]);
                double propagated_error = edge.lipschitz_constant * from_error + hardware.machine_epsilon();
                
                section.consistency_error = std::max(
                    section.consistency_error,
                    std::abs(to_error - propagated_error)
                );
            }
        }
        
        sections.push_back(section);
    }
    
    return sections;
}

// Generate graphviz representation
std::string SheafCohomology::to_graphviz() const {
    std::ostringstream oss;
    oss << "digraph PrecisionSheaf {\n";
    oss << "  rankdir=TB;\n";
    oss << "  node [shape=box, style=rounded];\n\n";
    
    // Vertices
    for (size_t i = 0; i < graph_.num_vertices(); ++i) {
        const auto& v = graph_.vertices()[i];
        oss << "  v" << i << " [label=\"" << v.name << "\\n"
            << "Layer: " << v.layer_id;
        if (v.head_id >= 0) {
            oss << ", Head: " << v.head_id;
        }
        oss << "\\nκ=" << v.local_curvature
            << "\\np=" << v.required_precision_bits << " bits\"];\n";
    }
    
    oss << "\n";
    
    // Edges
    for (const auto& e : graph_.edges()) {
        oss << "  v" << e.from_vertex << " -> v" << e.to_vertex
            << " [label=\"L=" << e.lipschitz_constant << "\"];\n";
    }
    
    oss << "}\n";
    return oss.str();
}

// MultiLayerPrecisionAnalyzer implementation

void MultiLayerPrecisionAnalyzer::build_graph_from_transformer(
    int num_layers,
    int num_heads,
    int hidden_dim,
    int seq_len,
    double temperature
) {
    num_layers_ = num_layers;
    num_heads_ = num_heads;
    hidden_dim_ = hidden_dim;
    seq_len_ = seq_len;
    temperature_ = temperature;
    
    graph_ = ComputationGraph();
    
    // Build graph structure
    std::vector<int> input_vertices;
    
    // Input embedding
    ComputationVertex input_vertex("input_embedding", 0);
    int input_id = graph_.add_vertex(input_vertex);
    input_vertices.push_back(input_id);
    
    // For each transformer layer
    for (int layer = 0; layer < num_layers; ++layer) {
        std::vector<int> layer_heads;
        
        // Multi-head attention
        for (int head = 0; head < num_heads; ++head) {
            std::ostringstream name;
            name << "attention_L" << layer << "_H" << head;
            ComputationVertex attn_vertex(name.str(), layer, head);
            int attn_id = graph_.add_vertex(attn_vertex);
            layer_heads.push_back(attn_id);
            
            // Connect from previous layer
            for (int prev_id : input_vertices) {
                ComputationEdge edge(prev_id, attn_id, 1.0);  // Will be updated
                graph_.add_edge(edge);
            }
        }
        
        // Attention concatenation and projection
        ComputationVertex concat_vertex("attention_concat_L" + std::to_string(layer), layer);
        int concat_id = graph_.add_vertex(concat_vertex);
        
        for (int head_id : layer_heads) {
            ComputationEdge edge(head_id, concat_id, 1.0);
            graph_.add_edge(edge);
        }
        
        // Feed-forward network
        ComputationVertex ffn_vertex("ffn_L" + std::to_string(layer), layer);
        int ffn_id = graph_.add_vertex(ffn_vertex);
        
        ComputationEdge edge(concat_id, ffn_id, 1.0);
        graph_.add_edge(edge);
        
        // Layer normalization
        ComputationVertex norm_vertex("layernorm_L" + std::to_string(layer), layer);
        int norm_id = graph_.add_vertex(norm_vertex);
        
        ComputationEdge edge2(ffn_id, norm_id, 1.0);
        graph_.add_edge(edge2);
        
        // Update input for next layer
        input_vertices = {norm_id};
    }
    
    // Output projection
    ComputationVertex output_vertex("output_projection", num_layers);
    int output_id = graph_.add_vertex(output_vertex);
    
    for (int prev_id : input_vertices) {
        ComputationEdge edge(prev_id, output_id, 1.0);
        graph_.add_edge(edge);
    }
}

void MultiLayerPrecisionAnalyzer::populate_from_weights(
    const std::vector<torch::Tensor>& Q_weights,
    const std::vector<torch::Tensor>& K_weights,
    const std::vector<torch::Tensor>& V_weights,
    const std::vector<torch::Tensor>& ffn_weights
) {
    // Populate curvature and Lipschitz constants from actual weights
    int head_dim = hidden_dim_ / num_heads_;
    
    for (size_t i = 0; i < graph_.vertices().size(); ++i) {
        auto& vertex = const_cast<ComputationVertex&>(graph_.vertices()[i]);
        
        if (vertex.head_id >= 0 && vertex.layer_id < static_cast<int>(Q_weights.size())) {
            // Attention vertex - compute curvature
            const auto& Q = Q_weights[vertex.layer_id];
            const auto& K = K_weights[vertex.layer_id];
            
            // Extract this head's Q and K
            int start_dim = vertex.head_id * head_dim;
            int end_dim = start_dim + head_dim;
            
            auto Q_head = Q.index({torch::indexing::Slice(), 
                                   torch::indexing::Slice(start_dim, end_dim)});
            auto K_head = K.index({torch::indexing::Slice(), 
                                   torch::indexing::Slice(start_dim, end_dim)});
            
            // Compute curvature
            double Q_norm = Q_head.norm().item<double>();
            double K_norm = K_head.norm().item<double>();
            
            // Estimate max logit
            double max_logit = Q_norm * K_norm / std::sqrt(head_dim) / temperature_;
            
            // HNF curvature formula
            vertex.local_curvature = 0.5 * Q_norm * K_norm / std::sqrt(head_dim) 
                                   * std::exp(2.0 * max_logit / std::sqrt(head_dim));
            
            // Required precision from HNF Theorem 4.1
            vertex.required_precision_bits = std::log2(
                vertex.local_curvature * seq_len_ * seq_len_ / 1e-6
            );
        } else if (vertex.name.find("ffn") != std::string::npos && 
                   vertex.layer_id < static_cast<int>(ffn_weights.size())) {
            // Feed-forward vertex
            const auto& W = ffn_weights[vertex.layer_id];
            double W_norm = W.norm().item<double>();
            
            // FFN is nearly linear, low curvature
            vertex.local_curvature = W_norm * W_norm * 0.1;
            vertex.required_precision_bits = std::log2(
                vertex.local_curvature * hidden_dim_ / 1e-6
            );
        } else {
            // Other vertices - use default values
            vertex.local_curvature = 1.0;
            vertex.required_precision_bits = 32.0;
        }
    }
    
    // Update Lipschitz constants on edges
    for (size_t i = 0; i < graph_.edges().size(); ++i) {
        auto& edge = const_cast<ComputationEdge&>(graph_.edges()[i]);
        const auto& from_vertex = graph_.vertices()[edge.from_vertex];
        const auto& to_vertex = graph_.vertices()[edge.to_vertex];
        
        // Lipschitz constant is product of operator norms along the edge
        edge.lipschitz_constant = std::sqrt(from_vertex.local_curvature) * 
                                 std::sqrt(to_vertex.local_curvature);
        edge.lipschitz_constant = std::max(edge.lipschitz_constant, 1.0);
    }
}

SheafCohomology::CohomologyResult MultiLayerPrecisionAnalyzer::analyze_precision(
    double target_accuracy,
    const HardwareModel& hardware
) {
    SheafCohomology sheaf(graph_);
    return sheaf.compute_cohomology(target_accuracy, hardware);
}

MultiLayerPrecisionAnalyzer::AnalysisReport MultiLayerPrecisionAnalyzer::generate_report(
    double target_accuracy,
    const HardwareModel& hardware
) {
    AnalysisReport report;
    report.cohomology = analyze_precision(target_accuracy, hardware);
    
    // Per-layer diagnoses
    for (int layer = 0; layer < num_layers_; ++layer) {
        std::ostringstream oss;
        oss << "Layer " << layer << ": ";
        
        double max_curvature = 0.0;
        double max_precision = 0.0;
        
        for (const auto& vertex : graph_.vertices()) {
            if (vertex.layer_id == layer) {
                max_curvature = std::max(max_curvature, vertex.local_curvature);
                max_precision = std::max(max_precision, vertex.required_precision_bits);
            }
        }
        
        oss << "Max curvature = " << max_curvature << ", "
            << "Max precision required = " << max_precision << " bits";
        
        report.layer_diagnoses.push_back(oss.str());
        report.per_layer_precision.push_back(max_precision);
    }
    
    // Total error bound using HNF composition theorem
    report.total_error_bound = 0.0;
    if (!report.cohomology.h0_generators.empty()) {
        const auto& global_section = report.cohomology.h0_generators[0];
        
        // Error accumulates as product of Lipschitz constants
        double accumulated_lipschitz = 1.0;
        for (const auto& edge : graph_.edges()) {
            accumulated_lipschitz *= edge.lipschitz_constant;
        }
        
        report.total_error_bound = accumulated_lipschitz * std::pow(2.0, -report.cohomology.minimal_precision);
    }
    
    // Check if achievable with hardware
    report.is_achievable_with_hardware = 
        report.cohomology.minimal_precision <= hardware.precision_bits();
    
    // Recommendations
    if (!report.is_achievable_with_hardware) {
        std::ostringstream oss;
        oss << "Required precision (" << report.cohomology.minimal_precision 
            << " bits) exceeds hardware capability (" << hardware.precision_bits() << " bits).";
        report.recommendations.push_back(oss.str());
        report.recommendations.push_back("Consider: (1) Reducing model complexity, (2) Using higher precision hardware, (3) Increasing temperature");
    }
    
    if (report.cohomology.h1_dimension > 0) {
        report.recommendations.push_back("Precision obstruction detected! No consistent global precision assignment exists.");
        for (const auto& reason : report.cohomology.obstruction_reasons) {
            report.recommendations.push_back("  - " + reason);
        }
    }
    
    return report;
}

} // namespace attention
} // namespace hnf
