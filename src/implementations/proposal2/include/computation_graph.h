#pragma once

#include <torch/torch.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <functional>

namespace hnf {
namespace sheaf {

// Forward declarations
class ComputationNode;
class ComputationGraph;

// Represents a node in the computation graph
// Each node is an operation with curvature and Lipschitz bounds
class ComputationNode {
public:
    std::string name;
    std::string op_type;  // "matmul", "softmax", "relu", etc.
    
    // HNF numerical invariants (from paper Section 5)
    double curvature;         // κ^curv: curvature bound
    double lipschitz;         // L_f: Lipschitz constant
    double diameter;          // D: domain diameter
    
    // Precision requirements
    int min_precision_bits;   // p_min from Theorem 5.7
    int assigned_precision;   // Current precision assignment
    
    // Computational metadata
    std::vector<std::string> inputs;   // Input node names
    std::vector<std::string> outputs;  // Output node names
    
    // Error functional Φ_f(ε, H) for this operation
    std::function<double(double, int)> error_functional;
    
    // Shape information for tensor operations
    std::vector<int64_t> input_shape;
    std::vector<int64_t> output_shape;
    
    ComputationNode(
        const std::string& n,
        const std::string& op,
        double kappa = 0.0,
        double L = 1.0,
        double D = 1.0
    ) : name(n), 
        op_type(op),
        curvature(kappa),
        lipschitz(L),
        diameter(D),
        min_precision_bits(0),
        assigned_precision(32)
    {
        // Default error functional: Φ_f(ε, p) = L·ε + 2^(-p)
        error_functional = [L](double eps, int p) {
            return L * eps + std::pow(2.0, -p);
        };
    }
    
    // Compute minimum precision using Theorem 5.7 (Precision Obstruction)
    // p ≥ log₂(c·κ·D²/ε)
    void compute_min_precision(double target_eps, double c = 2.0) {
        if (curvature > 1e-10 && diameter > 1e-10) {
            min_precision_bits = static_cast<int>(std::ceil(
                std::log2(c * curvature * diameter * diameter / target_eps)
            ));
            // Ensure reasonable bounds
            min_precision_bits = std::max(7, std::min(min_precision_bits, 112));
        } else {
            // Linear or nearly-linear operations
            min_precision_bits = static_cast<int>(std::ceil(std::log2(1.0 / target_eps)));
        }
    }
    
    // Get required hardware precision (quantized to standard formats)
    int get_hardware_precision() const {
        if (min_precision_bits <= 7) return 7;   // bfloat16
        if (min_precision_bits <= 10) return 10; // float16
        if (min_precision_bits <= 23) return 23; // float32
        if (min_precision_bits <= 52) return 52; // float64
        return 112; // float128
    }
};

// Edge in the computation graph
struct ComputationEdge {
    std::string source;
    std::string target;
    
    // Precision compatibility constraint
    // |p_source - p_target| ≤ tolerance
    int tolerance;
    
    ComputationEdge(const std::string& s, const std::string& t, int tol = 0)
        : source(s), target(t), tolerance(tol) {}
    
    bool operator==(const ComputationEdge& other) const {
        return source == other.source && target == other.target;
    }
};

// Hash function for edges (for use in unordered_map)
struct EdgeHash {
    std::size_t operator()(const ComputationEdge& e) const {
        return std::hash<std::string>()(e.source) ^ 
               (std::hash<std::string>()(e.target) << 1);
    }
};

// Computation Graph: DAG representing a neural network or computation
class ComputationGraph {
public:
    std::unordered_map<std::string, std::shared_ptr<ComputationNode>> nodes;
    std::vector<ComputationEdge> edges;
    
    // Topological ordering cache
    mutable std::vector<std::string> topo_order;
    mutable bool topo_valid = false;
    
    ComputationGraph() = default;
    
    // Add a node to the graph
    void add_node(std::shared_ptr<ComputationNode> node) {
        nodes[node->name] = node;
        topo_valid = false;
    }
    
    // Add an edge between nodes
    void add_edge(const std::string& source, const std::string& target, int tolerance = 0) {
        edges.emplace_back(source, target, tolerance);
        
        // Update node connections
        if (nodes.count(source) && nodes.count(target)) {
            nodes[source]->outputs.push_back(target);
            nodes[target]->inputs.push_back(source);
        }
        
        topo_valid = false;
    }
    
    // Get neighbors of a node (both input and output)
    std::unordered_set<std::string> get_neighbors(const std::string& node_name) const {
        std::unordered_set<std::string> neighbors;
        
        if (!nodes.count(node_name)) return neighbors;
        
        const auto& node = nodes.at(node_name);
        neighbors.insert(node->inputs.begin(), node->inputs.end());
        neighbors.insert(node->outputs.begin(), node->outputs.end());
        
        return neighbors;
    }
    
    // Get all nodes reachable from a set of nodes
    std::unordered_set<std::string> get_reachable(
        const std::unordered_set<std::string>& start_nodes
    ) const {
        std::unordered_set<std::string> reachable = start_nodes;
        std::vector<std::string> queue(start_nodes.begin(), start_nodes.end());
        
        while (!queue.empty()) {
            std::string current = queue.back();
            queue.pop_back();
            
            auto neighbors = get_neighbors(current);
            for (const auto& neighbor : neighbors) {
                if (reachable.insert(neighbor).second) {
                    queue.push_back(neighbor);
                }
            }
        }
        
        return reachable;
    }
    
    // Extract subgraph containing specified nodes
    ComputationGraph subgraph(const std::unordered_set<std::string>& node_names) const {
        ComputationGraph sub;
        
        // Add nodes
        for (const auto& name : node_names) {
            if (nodes.count(name)) {
                sub.add_node(nodes.at(name));
            }
        }
        
        // Add edges that connect nodes in the subgraph
        for (const auto& edge : edges) {
            if (node_names.count(edge.source) && node_names.count(edge.target)) {
                sub.add_edge(edge.source, edge.target, edge.tolerance);
            }
        }
        
        return sub;
    }
    
    // Topological sort (for DAG traversal)
    const std::vector<std::string>& topological_order() const {
        if (topo_valid) return topo_order;
        
        topo_order.clear();
        std::unordered_map<std::string, int> in_degree;
        
        // Compute in-degrees
        for (const auto& [name, node] : nodes) {
            in_degree[name] = static_cast<int>(node->inputs.size());
        }
        
        // Find nodes with no incoming edges
        std::vector<std::string> queue;
        for (const auto& [name, degree] : in_degree) {
            if (degree == 0) {
                queue.push_back(name);
            }
        }
        
        // Kahn's algorithm
        while (!queue.empty()) {
            std::string current = queue.back();
            queue.pop_back();
            topo_order.push_back(current);
            
            if (!nodes.count(current)) continue;
            
            // Reduce in-degree of neighbors
            for (const auto& output : nodes.at(current)->outputs) {
                if (--in_degree[output] == 0) {
                    queue.push_back(output);
                }
            }
        }
        
        topo_valid = true;
        return topo_order;
    }
    
    // Check if graph is acyclic
    bool is_acyclic() const {
        topological_order();
        return topo_order.size() == nodes.size();
    }
    
    // Check if edge exists
    bool has_edge(const std::string& source, const std::string& target) const {
        for (const auto& edge : edges) {
            if (edge.source == source && edge.target == target) {
                return true;
            }
        }
        return false;
    }
    
    // Get all node names
    std::vector<std::string> get_nodes() const {
        std::vector<std::string> node_names;
        node_names.reserve(nodes.size());
        for (const auto& [name, _] : nodes) {
            node_names.push_back(name);
        }
        return node_names;
    }
    
    // Get node by name
    std::shared_ptr<ComputationNode> get_node(const std::string& name) const {
        auto it = nodes.find(name);
        if (it != nodes.end()) {
            return it->second;
        }
        return nullptr;
    }
    
    // Compute global Lipschitz constant (product along longest path)
    double global_lipschitz() const {
        if (!is_acyclic()) return std::numeric_limits<double>::infinity();
        
        const auto& order = topological_order();
        std::unordered_map<std::string, double> max_lip;
        
        // Initialize
        for (const auto& [name, _] : nodes) {
            max_lip[name] = 1.0;
        }
        
        // Forward pass
        for (const auto& name : order) {
            if (!nodes.count(name)) continue;
            const auto& node = nodes.at(name);
            
            double incoming_lip = 1.0;
            for (const auto& input : node->inputs) {
                incoming_lip = std::max(incoming_lip, max_lip[input]);
            }
            
            max_lip[name] = incoming_lip * node->lipschitz;
        }
        
        // Return maximum
        double result = 1.0;
        for (const auto& [_, lip] : max_lip) {
            result = std::max(result, lip);
        }
        return result;
    }
    
    // Compute curvature for the entire graph (max over nodes)
    double global_curvature() const {
        double max_curv = 0.0;
        for (const auto& [_, node] : nodes) {
            max_curv = std::max(max_curv, node->curvature);
        }
        return max_curv;
    }
    
    // Get all input nodes (no predecessors)
    std::vector<std::string> input_nodes() const {
        std::vector<std::string> inputs;
        for (const auto& [name, node] : nodes) {
            if (node->inputs.empty()) {
                inputs.push_back(name);
            }
        }
        return inputs;
    }
    
    // Get all output nodes (no successors)
    std::vector<std::string> output_nodes() const {
        std::vector<std::string> outputs;
        for (const auto& [name, node] : nodes) {
            if (node->outputs.empty()) {
                outputs.push_back(name);
            }
        }
        return outputs;
    }
};

} // namespace sheaf
} // namespace hnf
