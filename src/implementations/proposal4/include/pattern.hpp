#pragma once

#include "graph_ir.hpp"
#include <unordered_map>
#include <optional>
#include <functional>

namespace hnf {
namespace rewriter {

// Pattern for matching subgraphs
// Wildcards are represented as INPUT nodes with special names starting with "$"
class Pattern {
public:
    Graph pattern_graph;
    std::string root_id;  // The output node of the pattern
    
    Pattern(const Graph& g, const std::string& root) 
        : pattern_graph(g), root_id(root) {}
    
    // Try to match this pattern starting at a given node in the target graph
    // Returns mapping from pattern node IDs to target graph node IDs
    std::optional<std::unordered_map<std::string, std::string>> 
    match(const Graph& target, const std::string& start_id) const {
        
        std::unordered_map<std::string, std::string> mapping;
        std::unordered_map<std::string, std::string> wildcard_binding;
        
        if (!match_node(target, start_id, root_id, mapping, wildcard_binding)) {
            return std::nullopt;
        }
        
        return mapping;
    }
    
private:
    bool match_node(const Graph& target,
                   const std::string& target_id,
                   const std::string& pattern_id,
                   std::unordered_map<std::string, std::string>& mapping,
                   std::unordered_map<std::string, std::string>& wildcard_binding) const {
        
        auto pattern_node = pattern_graph.get_node(pattern_id);
        if (!pattern_node) return false;
        
        // Check if this is a wildcard (input with $ prefix)
        if (pattern_node->op == OpType::INPUT && pattern_id[0] == '$') {
            // Bind wildcard to target node
            auto it = wildcard_binding.find(pattern_id);
            if (it != wildcard_binding.end()) {
                // Wildcard already bound, check consistency
                return it->second == target_id;
            } else {
                wildcard_binding[pattern_id] = target_id;
                mapping[pattern_id] = target_id;
                return true;
            }
        }
        
        auto target_node = target.get_node(target_id);
        if (!target_node) return false;
        
        // Check operation type matches
        if (pattern_node->op != target_node->op) {
            return false;
        }
        
        // Check number of inputs matches
        if (pattern_node->inputs.size() != target_node->inputs.size()) {
            return false;
        }
        
        // Record mapping
        mapping[pattern_id] = target_id;
        
        // Recursively match inputs
        for (size_t i = 0; i < pattern_node->inputs.size(); ++i) {
            const auto& pattern_input = pattern_node->inputs[i];
            const auto& target_input = target_node->inputs[i];
            
            if (!match_node(target, target_input, pattern_input, mapping, wildcard_binding)) {
                return false;
            }
        }
        
        return true;
    }
};

// Pattern library for common numerical patterns
class PatternLibrary {
public:
    // log(exp(x)) pattern
    static Pattern log_exp_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP, std::vector<std::string>{"$x"});
        auto log_node = std::make_shared<Node>("log", OpType::LOG, std::vector<std::string>{"exp"});
        
        g.add_node(x_node);
        g.add_node(exp_node);
        g.add_node(log_node);
        g.set_outputs({"log"});
        
        return Pattern(g, "log");
    }
    
    // exp(log(x)) pattern
    static Pattern exp_log_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto log_node = std::make_shared<Node>("log", OpType::LOG, std::vector<std::string>{"$x"});
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP, std::vector<std::string>{"log"});
        
        g.add_node(x_node);
        g.add_node(log_node);
        g.add_node(exp_node);
        g.set_outputs({"exp"});
        
        return Pattern(g, "exp");
    }
    
    // log(sum(exp(x))) pattern (naive logsumexp)
    static Pattern naive_logsumexp_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP, std::vector<std::string>{"$x"});
        auto sum_node = std::make_shared<Node>("sum", OpType::SUM, std::vector<std::string>{"exp"});
        auto log_node = std::make_shared<Node>("log", OpType::LOG, std::vector<std::string>{"sum"});
        
        g.add_node(x_node);
        g.add_node(exp_node);
        g.add_node(sum_node);
        g.add_node(log_node);
        g.set_outputs({"log"});
        
        return Pattern(g, "log");
    }
    
    // exp(x) / sum(exp(x)) pattern (naive softmax)
    static Pattern naive_softmax_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP, std::vector<std::string>{"$x"});
        auto sum_node = std::make_shared<Node>("sum", OpType::SUM, std::vector<std::string>{"exp"});
        auto div_node = std::make_shared<Node>("div", OpType::DIV, 
                                               std::vector<std::string>{"exp", "sum"});
        
        g.add_node(x_node);
        g.add_node(exp_node);
        g.add_node(sum_node);
        g.add_node(div_node);
        g.set_outputs({"div"});
        
        return Pattern(g, "div");
    }
    
    // a - b where a and b are similar (for compensated operations)
    static Pattern cancellation_pattern() {
        Graph g;
        auto a_node = std::make_shared<Node>("$a", OpType::INPUT);
        auto b_node = std::make_shared<Node>("$b", OpType::INPUT);
        auto sub_node = std::make_shared<Node>("sub", OpType::SUB,
                                               std::vector<std::string>{"$a", "$b"});
        
        g.add_node(a_node);
        g.add_node(b_node);
        g.add_node(sub_node);
        g.set_outputs({"sub"});
        
        return Pattern(g, "sub");
    }
    
    // x^2 then sqrt pattern
    static Pattern square_sqrt_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto two_node = std::make_shared<Node>("$two", OpType::CONSTANT);
        auto pow_node = std::make_shared<Node>("pow", OpType::POW,
                                               std::vector<std::string>{"$x", "$two"});
        auto sqrt_node = std::make_shared<Node>("sqrt", OpType::SQRT,
                                                std::vector<std::string>{"pow"});
        
        g.add_node(x_node);
        g.add_node(two_node);
        g.add_node(pow_node);
        g.add_node(sqrt_node);
        g.set_outputs({"sqrt"});
        
        return Pattern(g, "sqrt");
    }
    
    // -log(softmax(x)) pattern (can be fused to cross-entropy)
    static Pattern negative_log_softmax_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto softmax_node = std::make_shared<Node>("softmax", OpType::SOFTMAX,
                                                   std::vector<std::string>{"$x"});
        auto log_node = std::make_shared<Node>("log", OpType::LOG,
                                               std::vector<std::string>{"softmax"});
        auto neg_node = std::make_shared<Node>("neg", OpType::NEG,
                                               std::vector<std::string>{"log"});
        
        g.add_node(x_node);
        g.add_node(softmax_node);
        g.add_node(log_node);
        g.add_node(neg_node);
        g.set_outputs({"neg"});
        
        return Pattern(g, "neg");
    }
};

} // namespace rewriter
} // namespace hnf
