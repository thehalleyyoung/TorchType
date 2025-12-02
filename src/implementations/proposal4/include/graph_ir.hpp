#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <optional>
#include <functional>
#include <cmath>
#include <sstream>
#include <iostream>

namespace hnf {
namespace rewriter {

// OpType enumeration covering all operations we need
enum class OpType {
    // Arithmetic
    ADD, SUB, MUL, DIV, NEG, ABS,
    // Transcendental
    EXP, LOG, SQRT, POW, LOG1P, EXPM1,
    // Matrix operations
    MATMUL, TRANSPOSE,
    // Reductions
    SUM, MAX, MIN, MEAN, KAHAN_SUM,
    // Activations
    RELU, SIGMOID, TANH, SOFTMAX, LOG_SOFTMAX, GELU, SWIGLU,
    // Normalization
    LAYER_NORM, BATCH_NORM, RMS_NORM,
    // Composite operations
    LOGSUMEXP, STABLE_SOFTMAX,
    // Attention operations
    FLASH_ATTENTION, SCALED_DOT_PRODUCT_ATTENTION,
    // Compensated arithmetic
    COMPENSATED_DOT,
    // Special
    CONSTANT, INPUT, OUTPUT, IDENTITY
};

// Convert OpType to string for debugging
inline std::string optype_to_string(OpType op) {
    switch (op) {
        case OpType::ADD: return "add";
        case OpType::SUB: return "sub";
        case OpType::MUL: return "mul";
        case OpType::DIV: return "div";
        case OpType::NEG: return "neg";
        case OpType::ABS: return "abs";
        case OpType::EXP: return "exp";
        case OpType::LOG: return "log";
        case OpType::SQRT: return "sqrt";
        case OpType::POW: return "pow";
        case OpType::LOG1P: return "log1p";
        case OpType::EXPM1: return "expm1";
        case OpType::MATMUL: return "matmul";
        case OpType::TRANSPOSE: return "transpose";
        case OpType::SUM: return "sum";
        case OpType::MAX: return "max";
        case OpType::MIN: return "min";
        case OpType::MEAN: return "mean";
        case OpType::KAHAN_SUM: return "kahan_sum";
        case OpType::RELU: return "relu";
        case OpType::SIGMOID: return "sigmoid";
        case OpType::TANH: return "tanh";
        case OpType::SOFTMAX: return "softmax";
        case OpType::LOG_SOFTMAX: return "log_softmax";
        case OpType::GELU: return "gelu";
        case OpType::SWIGLU: return "swiglu";
        case OpType::LAYER_NORM: return "layer_norm";
        case OpType::BATCH_NORM: return "batch_norm";
        case OpType::RMS_NORM: return "rms_norm";
        case OpType::LOGSUMEXP: return "logsumexp";
        case OpType::STABLE_SOFTMAX: return "stable_softmax";
        case OpType::FLASH_ATTENTION: return "flash_attention";
        case OpType::SCALED_DOT_PRODUCT_ATTENTION: return "scaled_dot_product_attention";
        case OpType::COMPENSATED_DOT: return "compensated_dot";
        case OpType::CONSTANT: return "constant";
        case OpType::INPUT: return "input";
        case OpType::OUTPUT: return "output";
        case OpType::IDENTITY: return "identity";
        default: return "unknown";
    }
}

// Statistics about tensor values (used for curvature computation)
struct TensorStats {
    double min_val;
    double max_val;
    double mean_val;
    double std_val;
    double condition_number;
    std::vector<int> shape;
    
    TensorStats() : min_val(0), max_val(0), mean_val(0), std_val(0), condition_number(1.0) {}
    
    double range() const { return max_val - min_val; }
};

// Node attributes (e.g., axis for reduction operations)
struct NodeAttrs {
    std::unordered_map<std::string, int> int_attrs;
    std::unordered_map<std::string, double> float_attrs;
    std::unordered_map<std::string, std::string> string_attrs;
    
    void set_int(const std::string& key, int val) { int_attrs[key] = val; }
    void set_float(const std::string& key, double val) { float_attrs[key] = val; }
    void set_string(const std::string& key, const std::string& val) { string_attrs[key] = val; }
    
    int get_int(const std::string& key, int default_val = 0) const {
        auto it = int_attrs.find(key);
        return it != int_attrs.end() ? it->second : default_val;
    }
    
    double get_float(const std::string& key, double default_val = 0.0) const {
        auto it = float_attrs.find(key);
        return it != float_attrs.end() ? it->second : default_val;
    }
    
    std::string get_string(const std::string& key, const std::string& default_val = "") const {
        auto it = string_attrs.find(key);
        return it != string_attrs.end() ? it->second : default_val;
    }
};

// Forward declaration
class Graph;

// Node in the computation graph
class Node {
public:
    std::string id;
    OpType op;
    std::vector<std::string> inputs;  // IDs of input nodes
    NodeAttrs attrs;
    
    Node(const std::string& id_, OpType op_, const std::vector<std::string>& inputs_ = {})
        : id(id_), op(op_), inputs(inputs_) {}
    
    Node(const std::string& id_, OpType op_, const std::vector<std::string>& inputs_, const NodeAttrs& attrs_)
        : id(id_), op(op_), inputs(inputs_), attrs(attrs_) {}
    
    std::string to_string() const {
        std::stringstream ss;
        ss << id << " = " << optype_to_string(op) << "(";
        for (size_t i = 0; i < inputs.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << inputs[i];
        }
        ss << ")";
        return ss.str();
    }
};

// Computation graph
class Graph {
private:
    std::unordered_map<std::string, std::shared_ptr<Node>> nodes_;
    std::vector<std::string> inputs_;
    std::vector<std::string> outputs_;
    int next_id_ = 0;
    
public:
    Graph() = default;
    
    Graph(const std::unordered_map<std::string, std::shared_ptr<Node>>& nodes,
          const std::vector<std::string>& inputs,
          const std::vector<std::string>& outputs)
        : nodes_(nodes), inputs_(inputs), outputs_(outputs) {}
    
    // Generate unique node ID
    std::string gen_id(const std::string& prefix = "n") {
        return prefix + std::to_string(next_id_++);
    }
    
    // Add a node to the graph
    void add_node(std::shared_ptr<Node> node) {
        nodes_[node->id] = node;
    }
    
    // Convenience method to add node with parameters
    void add_node(const std::string& id, OpType op, 
                  const std::vector<std::string>& inputs = {},
                  const NodeAttrs& attrs = NodeAttrs()) {
        auto node = std::make_shared<Node>(id, op, inputs, attrs);
        nodes_[id] = node;
    }
    
    // Add a constant node
    void add_constant(const std::string& id, double value) {
        auto node = std::make_shared<Node>(id, OpType::CONSTANT);
        node->attrs.set_float("value", value);
        nodes_[id] = node;
    }
    
    // Get node by ID
    std::shared_ptr<Node> get_node(const std::string& id) const {
        auto it = nodes_.find(id);
        return it != nodes_.end() ? it->second : nullptr;
    }
    
    // Check if node exists
    bool has_node(const std::string& id) const {
        return nodes_.find(id) != nodes_.end();
    }
    
    // Get all nodes
    const std::unordered_map<std::string, std::shared_ptr<Node>>& nodes() const {
        return nodes_;
    }
    
    // Get inputs
    const std::vector<std::string>& inputs() const { return inputs_; }
    void set_inputs(const std::vector<std::string>& ins) { inputs_ = ins; }
    void add_input(const std::string& id) { inputs_.push_back(id); }
    
    // Get outputs
    const std::vector<std::string>& outputs() const { return outputs_; }
    void set_outputs(const std::vector<std::string>& outs) { outputs_ = outs; }
    void add_output(const std::string& id) { outputs_.push_back(id); }
    
    // Topological sort
    std::vector<std::string> topological_order() const {
        std::vector<std::string> result;
        std::unordered_set<std::string> visited;
        std::unordered_set<std::string> in_stack;
        
        std::function<bool(const std::string&)> visit = [&](const std::string& id) -> bool {
            if (visited.count(id)) return true;
            if (in_stack.count(id)) return false; // Cycle detected
            
            in_stack.insert(id);
            auto node = get_node(id);
            if (node) {
                for (const auto& input_id : node->inputs) {
                    if (!visit(input_id)) return false;
                }
            }
            in_stack.erase(id);
            visited.insert(id);
            result.push_back(id);
            return true;
        };
        
        // Visit outputs first (reverse topological order)
        for (const auto& out_id : outputs_) {
            if (!visit(out_id)) {
                std::cerr << "Warning: Cycle detected in graph" << std::endl;
                return {};
            }
        }
        
        // Visit any remaining nodes
        for (const auto& [id, node] : nodes_) {
            visit(id);
        }
        
        return result;
    }
    
    // Extract subgraph containing given nodes
    Graph subgraph(const std::unordered_set<std::string>& node_ids) const {
        std::unordered_map<std::string, std::shared_ptr<Node>> sub_nodes;
        std::vector<std::string> sub_inputs;
        std::vector<std::string> sub_outputs;
        
        for (const auto& id : node_ids) {
            auto node = get_node(id);
            if (node) {
                sub_nodes[id] = node;
            }
        }
        
        // Find inputs: nodes referenced but not in subgraph
        std::unordered_set<std::string> referenced;
        for (const auto& [id, node] : sub_nodes) {
            for (const auto& input_id : node->inputs) {
                if (!node_ids.count(input_id)) {
                    sub_inputs.push_back(input_id);
                }
            }
        }
        
        // Outputs are the nodes in the subgraph that are referenced by output nodes
        for (const auto& out_id : outputs_) {
            if (node_ids.count(out_id)) {
                sub_outputs.push_back(out_id);
            }
        }
        
        return Graph(sub_nodes, sub_inputs, sub_outputs);
    }
    
    // Replace subgraph with new graph
    // old_ids: nodes to remove
    // new_graph: replacement graph
    // mapping: maps old_ids to new_graph's outputs
    Graph replace(const std::unordered_set<std::string>& old_ids, 
                  const Graph& new_graph,
                  const std::unordered_map<std::string, std::string>& mapping) const {
        
        std::unordered_map<std::string, std::shared_ptr<Node>> result_nodes;
        
        // Copy nodes from new_graph
        for (const auto& [id, node] : new_graph.nodes()) {
            result_nodes[id] = node;
        }
        
        // Copy nodes not in old_ids, updating their inputs
        for (const auto& [id, node] : nodes_) {
            if (old_ids.count(id)) continue;
            
            auto new_node = std::make_shared<Node>(*node);
            
            // Update inputs if they reference old nodes
            for (auto& input_id : new_node->inputs) {
                if (old_ids.count(input_id)) {
                    auto it = mapping.find(input_id);
                    if (it != mapping.end()) {
                        input_id = it->second;
                    }
                }
            }
            
            result_nodes[id] = new_node;
        }
        
        // Update outputs
        std::vector<std::string> result_outputs;
        for (const auto& out_id : outputs_) {
            if (old_ids.count(out_id)) {
                auto it = mapping.find(out_id);
                if (it != mapping.end()) {
                    result_outputs.push_back(it->second);
                }
            } else {
                result_outputs.push_back(out_id);
            }
        }
        
        return Graph(result_nodes, inputs_, result_outputs);
    }
    
    // Pretty print the graph
    std::string to_string() const {
        std::stringstream ss;
        ss << "Graph(\n";
        ss << "  Inputs: ";
        for (size_t i = 0; i < inputs_.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << inputs_[i];
        }
        ss << "\n  Outputs: ";
        for (size_t i = 0; i < outputs_.size(); ++i) {
            if (i > 0) ss << ", ";
            ss << outputs_[i];
        }
        ss << "\n  Nodes:\n";
        
        auto topo = topological_order();
        for (const auto& id : topo) {
            auto node = get_node(id);
            if (node) {
                ss << "    " << node->to_string() << "\n";
            }
        }
        ss << ")";
        return ss.str();
    }
    
    // Clone the graph
    Graph clone() const {
        std::unordered_map<std::string, std::shared_ptr<Node>> cloned_nodes;
        for (const auto& [id, node] : nodes_) {
            cloned_nodes[id] = std::make_shared<Node>(*node);
        }
        return Graph(cloned_nodes, inputs_, outputs_);
    }
};

} // namespace rewriter
} // namespace hnf
