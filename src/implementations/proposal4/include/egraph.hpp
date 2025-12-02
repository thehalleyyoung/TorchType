#pragma once

#include "graph_ir.hpp"
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <queue>
#include <functional>

namespace hnf {
namespace rewriter {

// Equality saturation using E-graphs
// Based on "egg: Easy, Efficient, and Extensible E-graphs" (Willsey et al., 2021)
// This implements the core E-graph data structure for equality saturation

// Represents an equivalence class ID
using EClassId = size_t;

// E-node: operation + children (which are e-class IDs)
struct ENode {
    OpType op;
    std::vector<EClassId> children;
    NodeAttrs attrs;
    
    bool operator==(const ENode& other) const {
        return op == other.op && children == other.children;
    }
    
    struct Hash {
        size_t operator()(const ENode& n) const {
            size_t h = std::hash<int>{}(static_cast<int>(n.op));
            for (auto c : n.children) {
                h ^= std::hash<size_t>{}(c) + 0x9e3779b9 + (h << 6) + (h >> 2);
            }
            return h;
        }
    };
};

// E-class: represents an equivalence class of expressions
struct EClass {
    EClassId id;
    std::vector<ENode> nodes;  // All equivalent expressions
    EClassId find_id;  // For union-find
    
    // Metadata for extraction
    double best_cost;
    std::optional<ENode> best_node;
    
    EClass(EClassId id_) : id(id_), find_id(id_), best_cost(std::numeric_limits<double>::infinity()) {}
};

// E-graph: the main equality saturation data structure
class EGraph {
public:
    EGraph() : next_class_id_(0) {}
    
    // Add a node to the e-graph, returns its e-class ID
    EClassId add_node(const ENode& node) {
        // Check if this exact node already exists (hashcons)
        auto it = hashcons_.find(node);
        if (it != hashcons_.end()) {
            return find(it->second);
        }
        
        // Create new e-class
        EClassId new_id = next_class_id_++;
        classes_[new_id] = EClass(new_id);
        classes_[new_id].nodes.push_back(node);
        hashcons_[node] = new_id;
        
        return new_id;
    }
    
    // Add an entire graph, returns root e-class ID
    EClassId add_graph(const Graph& g) {
        std::unordered_map<std::string, EClassId> node_to_eclass;
        
        // Process in topological order
        auto order = g.topological_order();
        
        for (const auto& node_id : order) {
            auto node_ptr = g.get_node(node_id);
            if (!node_ptr) continue;
            
            // Handle special cases
            if (node_ptr->op == OpType::INPUT || node_ptr->op == OpType::CONSTANT) {
                // Create e-class for input
                ENode enode;
                enode.op = node_ptr->op;
                enode.attrs = node_ptr->attrs;
                node_to_eclass[node_id] = add_node(enode);
                continue;
            }
            
            // Build e-node with children
            ENode enode;
            enode.op = node_ptr->op;
            enode.attrs = node_ptr->attrs;
            
            for (const auto& input_id : node_ptr->inputs) {
                auto it = node_to_eclass.find(input_id);
                if (it != node_to_eclass.end()) {
                    enode.children.push_back(find(it->second));
                }
            }
            
            node_to_eclass[node_id] = add_node(enode);
        }
        
        // Return root e-class
        if (!g.outputs.empty()) {
            auto it = node_to_eclass.find(g.outputs[0]);
            if (it != node_to_eclass.end()) {
                return find(it->second);
            }
        }
        
        return 0;
    }
    
    // Union two e-classes (make them equivalent)
    void merge(EClassId a, EClassId b) {
        EClassId root_a = find(a);
        EClassId root_b = find(b);
        
        if (root_a == root_b) return;
        
        // Union by rank: merge smaller into larger
        if (classes_[root_a].nodes.size() < classes_[root_b].nodes.size()) {
            std::swap(root_a, root_b);
        }
        
        // Merge root_b into root_a
        classes_[root_a].find_id = root_b;
        
        // Combine nodes
        for (const auto& node : classes_[root_b].nodes) {
            classes_[root_a].nodes.push_back(node);
        }
        
        // Update hashcons
        for (const auto& node : classes_[root_b].nodes) {
            hashcons_[node] = root_a;
        }
        
        pending_updates_.insert(root_a);
    }
    
    // Find representative of e-class (with path compression)
    EClassId find(EClassId id) {
        if (!classes_.count(id)) return id;
        
        if (classes_[id].find_id != id) {
            classes_[id].find_id = find(classes_[id].find_id);
        }
        
        return classes_[id].find_id;
    }
    
    // Apply a rewrite rule to the entire e-graph
    template<typename RewriteFn>
    bool apply_rewrites(const RewriteFn& rewrite_fn) {
        bool modified = false;
        
        // Collect all e-classes
        std::vector<EClassId> all_classes;
        for (const auto& [id, eclass] : classes_) {
            if (find(id) == id) {  // Only root classes
                all_classes.push_back(id);
            }
        }
        
        // Try to apply rewrites to each e-class
        for (EClassId class_id : all_classes) {
            const auto& eclass = classes_[find(class_id)];
            
            for (const auto& node : eclass.nodes) {
                // Try to rewrite this node
                auto rewrites = rewrite_fn(node, *this);
                
                for (const auto& rewrite : rewrites) {
                    EClassId new_id = add_node(rewrite);
                    if (new_id != find(class_id)) {
                        merge(class_id, new_id);
                        modified = true;
                    }
                }
            }
        }
        
        // Rebuild hashcons after merges
        if (modified) {
            rebuild();
        }
        
        return modified;
    }
    
    // Equality saturation: apply rewrites until fixed point
    template<typename RewriteFn>
    void saturate(const RewriteFn& rewrite_fn, size_t max_iterations = 100) {
        for (size_t iter = 0; iter < max_iterations; ++iter) {
            bool modified = apply_rewrites(rewrite_fn);
            if (!modified) {
                std::cout << "Saturation complete after " << iter << " iterations\n";
                return;
            }
        }
        
        std::cout << "Warning: Saturation did not converge after " << max_iterations << " iterations\n";
    }
    
    // Extract best expression from an e-class using a cost function
    template<typename CostFn>
    Graph extract(EClassId root_id, const CostFn& cost_fn) {
        // Bottom-up dynamic programming to find minimum cost expression
        std::unordered_map<EClassId, std::pair<double, ENode>> best;
        std::unordered_set<EClassId> visited;
        
        std::function<double(EClassId)> compute_cost = [&](EClassId id) -> double {
            id = find(id);
            
            if (visited.count(id)) {
                return best[id].first;
            }
            visited.insert(id);
            
            const auto& eclass = classes_[id];
            double min_cost = std::numeric_limits<double>::infinity();
            ENode best_node;
            
            for (const auto& node : eclass.nodes) {
                double cost = cost_fn(node);
                
                // Add cost of children
                for (EClassId child : node.children) {
                    cost += compute_cost(child);
                }
                
                if (cost < min_cost) {
                    min_cost = cost;
                    best_node = node;
                }
            }
            
            best[id] = {min_cost, best_node};
            return min_cost;
        };
        
        compute_cost(find(root_id));
        
        // Build graph from best nodes
        Graph result;
        std::unordered_map<EClassId, std::string> eclass_to_node_id;
        size_t counter = 0;
        
        std::function<std::string(EClassId)> build_graph = [&](EClassId id) -> std::string {
            id = find(id);
            
            auto it = eclass_to_node_id.find(id);
            if (it != eclass_to_node_id.end()) {
                return it->second;
            }
            
            const auto& [cost, node] = best[id];
            std::string node_id = "n" + std::to_string(counter++);
            
            std::vector<std::string> input_ids;
            for (EClassId child : node.children) {
                input_ids.push_back(build_graph(child));
            }
            
            if (node.op == OpType::INPUT) {
                result.add_input(node_id);
            } else {
                result.add_node(node_id, node.op, input_ids, node.attrs);
            }
            
            eclass_to_node_id[id] = node_id;
            return node_id;
        };
        
        std::string root_node_id = build_graph(find(root_id));
        result.add_output(root_node_id);
        
        return result;
    }
    
    // Get all expressions in an e-class
    const std::vector<ENode>& get_eclass_nodes(EClassId id) const {
        auto it = classes_.find(find(id));
        if (it != classes_.end()) {
            return it->second.nodes;
        }
        static const std::vector<ENode> empty;
        return empty;
    }
    
    // Check if two e-class IDs are equivalent
    bool equivalent(EClassId a, EClassId b) const {
        return find(a) == find(b);
    }
    
    // Get size of e-graph (number of e-classes)
    size_t size() const {
        size_t count = 0;
        for (const auto& [id, eclass] : classes_) {
            if (find(id) == id) count++;
        }
        return count;
    }
    
    // Get total number of e-nodes
    size_t num_nodes() const {
        size_t count = 0;
        for (const auto& [id, eclass] : classes_) {
            if (find(id) == id) {
                count += eclass.nodes.size();
            }
        }
        return count;
    }
    
private:
    std::unordered_map<EClassId, EClass> classes_;
    std::unordered_map<ENode, EClassId, ENode::Hash> hashcons_;
    std::unordered_set<EClassId> pending_updates_;
    EClassId next_class_id_;
    
    // Rebuild hashcons after merges
    void rebuild() {
        hashcons_.clear();
        
        for (auto& [id, eclass] : classes_) {
            if (find(id) != id) continue;
            
            for (const auto& node : eclass.nodes) {
                // Canonicalize children
                ENode canonical_node = node;
                for (auto& child : canonical_node.children) {
                    child = find(child);
                }
                
                hashcons_[canonical_node] = id;
            }
        }
        
        pending_updates_.clear();
    }
};

// Curvature-based cost function for extraction
class CurvatureCostFunction {
public:
    explicit CurvatureCostFunction(const std::unordered_map<std::string, TensorStats>& stats)
        : stats_(stats) {}
    
    double operator()(const ENode& node) const {
        // Base cost: favor simpler operations
        double base_cost = 1.0;
        
        // Add curvature penalty
        // We need to estimate curvature without knowing exact statistics
        // Use conservative estimates based on operation type
        
        switch (node.op) {
            case OpType::EXP:
                return base_cost + 1000.0;  // High curvature
            
            case OpType::LOG:
                return base_cost + 100.0;   // Medium curvature
            
            case OpType::DIV:
                return base_cost + 50.0;    // Can be unstable
            
            case OpType::SOFTMAX:
                return base_cost + 5000.0;  // Very high curvature (naive)
            
            case OpType::STABLE_SOFTMAX:
            case OpType::LOGSUMEXP:
                return base_cost + 1.0;     // Low curvature (stable)
            
            case OpType::ADD:
            case OpType::SUB:
            case OpType::MUL:
                return base_cost;           // Linear operations
            
            case OpType::MATMUL:
                return base_cost + 10.0;    // Depends on condition number
            
            default:
                return base_cost;
        }
    }
    
private:
    const std::unordered_map<std::string, TensorStats>& stats_;
};

// Rewrite rules for equality saturation
class SaturationRules {
public:
    // Returns a list of possible rewrites for a given e-node
    static std::vector<ENode> apply(const ENode& node, const EGraph& egraph) {
        std::vector<ENode> rewrites;
        
        // log(exp(x)) -> x
        if (node.op == OpType::LOG && node.children.size() == 1) {
            const auto& child_nodes = egraph.get_eclass_nodes(node.children[0]);
            for (const auto& child : child_nodes) {
                if (child.op == OpType::EXP && child.children.size() == 1) {
                    // Return identity: just the child's child
                    ENode identity;
                    identity.op = OpType::IDENTITY;
                    identity.children = child.children;
                    rewrites.push_back(identity);
                }
            }
        }
        
        // exp(log(x)) -> x
        if (node.op == OpType::EXP && node.children.size() == 1) {
            const auto& child_nodes = egraph.get_eclass_nodes(node.children[0]);
            for (const auto& child : child_nodes) {
                if (child.op == OpType::LOG && child.children.size() == 1) {
                    ENode identity;
                    identity.op = OpType::IDENTITY;
                    identity.children = child.children;
                    rewrites.push_back(identity);
                }
            }
        }
        
        // exp(x) / sum(exp(x)) -> stable_softmax(x)
        if (node.op == OpType::DIV && node.children.size() == 2) {
            const auto& numerator_nodes = egraph.get_eclass_nodes(node.children[0]);
            const auto& denominator_nodes = egraph.get_eclass_nodes(node.children[1]);
            
            for (const auto& num : numerator_nodes) {
                if (num.op == OpType::EXP && num.children.size() == 1) {
                    for (const auto& denom : denominator_nodes) {
                        if (denom.op == OpType::SUM && denom.children.size() == 1) {
                            // Check if denom's child is also exp(same input)
                            if (egraph.equivalent(denom.children[0], node.children[0])) {
                                ENode stable_softmax;
                                stable_softmax.op = OpType::STABLE_SOFTMAX;
                                stable_softmax.children = num.children;
                                rewrites.push_back(stable_softmax);
                            }
                        }
                    }
                }
            }
        }
        
        // log(sum(exp(x))) -> logsumexp(x)
        if (node.op == OpType::LOG && node.children.size() == 1) {
            const auto& child_nodes = egraph.get_eclass_nodes(node.children[0]);
            for (const auto& child : child_nodes) {
                if (child.op == OpType::SUM && child.children.size() == 1) {
                    const auto& sum_child_nodes = egraph.get_eclass_nodes(child.children[0]);
                    for (const auto& sum_child : sum_child_nodes) {
                        if (sum_child.op == OpType::EXP && sum_child.children.size() == 1) {
                            ENode logsumexp;
                            logsumexp.op = OpType::LOGSUMEXP;
                            logsumexp.children = sum_child.children;
                            rewrites.push_back(logsumexp);
                        }
                    }
                }
            }
        }
        
        // Associativity: (a + b) + c -> a + (b + c)
        if (node.op == OpType::ADD && node.children.size() == 2) {
            const auto& left_nodes = egraph.get_eclass_nodes(node.children[0]);
            for (const auto& left : left_nodes) {
                if (left.op == OpType::ADD && left.children.size() == 2) {
                    // (a + b) + c -> a + (b + c)
                    ENode inner;
                    inner.op = OpType::ADD;
                    inner.children = {left.children[1], node.children[1]};
                    
                    // This would need to add inner to egraph first
                    // For now, skip reassociation rules
                }
            }
        }
        
        // Commutativity: a + b -> b + a
        if (node.op == OpType::ADD || node.op == OpType::MUL) {
            if (node.children.size() == 2) {
                ENode commuted;
                commuted.op = node.op;
                commuted.children = {node.children[1], node.children[0]};
                rewrites.push_back(commuted);
            }
        }
        
        return rewrites;
    }
};

} // namespace rewriter
} // namespace hnf
