#pragma once

#include "graph_ir.hpp"
#include "pattern.hpp"
#include "curvature.hpp"
#include <functional>
#include <memory>

namespace hnf {
namespace rewriter {

// Function that generates a replacement graph given a match
using ReplacementGenerator = std::function<Graph(const std::unordered_map<std::string, std::string>&)>;

// Condition function that checks if a rewrite should be applied
using RewriteCondition = std::function<bool(const Graph&, const std::unordered_map<std::string, std::string>&)>;

// Represents a rewrite rule
class RewriteRule {
public:
    std::string name;
    std::string description;
    Pattern pattern;
    ReplacementGenerator replacement_gen;
    RewriteCondition condition;
    
    RewriteRule(const std::string& name_,
                const std::string& desc_,
                const Pattern& pat,
                ReplacementGenerator rep_gen,
                RewriteCondition cond = nullptr)
        : name(name_), description(desc_), pattern(pat), 
          replacement_gen(rep_gen), condition(cond) {}
    
    // Try to apply this rule to a graph
    // Returns new graph if successful, nullopt otherwise
    std::optional<Graph> apply(const Graph& graph) const {
        // Try to match pattern at each node
        for (const auto& [node_id, node] : graph.nodes()) {
            auto match = pattern.match(graph, node_id);
            if (!match) continue;
            
            // Check condition if specified
            if (condition && !condition(graph, *match)) {
                continue;
            }
            
            // Generate replacement graph
            Graph replacement = replacement_gen(*match);
            
            // Build mapping from old nodes to replacement outputs
            std::unordered_map<std::string, std::string> mapping;
            
            // Map the pattern root to replacement output
            auto pattern_outputs = pattern.pattern_graph.outputs();
            auto replacement_outputs = replacement.outputs();
            
            if (!pattern_outputs.empty() && !replacement_outputs.empty()) {
                auto pattern_root_match = match->find(pattern.root_id);
                if (pattern_root_match != match->end()) {
                    mapping[pattern_root_match->second] = replacement_outputs[0];
                }
            }
            
            // Collect all matched nodes to remove
            std::unordered_set<std::string> to_remove;
            for (const auto& [pat_id, graph_id] : *match) {
                // Don't remove wildcard inputs
                if (pat_id[0] != '$') {
                    to_remove.insert(graph_id);
                }
            }
            
            // Replace subgraph
            return graph.replace(to_remove, replacement, mapping);
        }
        
        return std::nullopt;
    }
};

// Library of standard rewrite rules based on HNF theory
class RewriteRuleLibrary {
public:
    // Rule: log(exp(x)) → x
    static RewriteRule log_exp_cancel() {
        return RewriteRule(
            "log_exp_cancel",
            "Cancel log(exp(x)) to x",
            PatternLibrary::log_exp_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                // Find the wildcard binding
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_input(x_it->second);
                    g.add_output(x_it->second);
                }
                return g;
            }
        );
    }
    
    // Rule: exp(log(x)) → x
    static RewriteRule exp_log_cancel() {
        return RewriteRule(
            "exp_log_cancel",
            "Cancel exp(log(x)) to x",
            PatternLibrary::exp_log_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_input(x_it->second);
                    g.add_output(x_it->second);
                }
                return g;
            }
        );
    }
    
    // Rule: log(sum(exp(x))) → stable_logsumexp(x)
    static RewriteRule naive_to_stable_logsumexp() {
        return RewriteRule(
            "stable_logsumexp",
            "Replace naive logsumexp with stable version",
            PatternLibrary::naive_logsumexp_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    // Create stable logsumexp operation
                    // Stable version: max + log(sum(exp(x - max)))
                    auto max_node = std::make_shared<Node>(
                        "max_stable", OpType::MAX, std::vector<std::string>{x_it->second});
                    auto sub_node = std::make_shared<Node>(
                        "sub_stable", OpType::SUB, 
                        std::vector<std::string>{x_it->second, "max_stable"});
                    auto exp_node = std::make_shared<Node>(
                        "exp_stable", OpType::EXP, std::vector<std::string>{"sub_stable"});
                    auto sum_node = std::make_shared<Node>(
                        "sum_stable", OpType::SUM, std::vector<std::string>{"exp_stable"});
                    auto log_node = std::make_shared<Node>(
                        "log_stable", OpType::LOG, std::vector<std::string>{"sum_stable"});
                    auto add_node = std::make_shared<Node>(
                        "add_stable", OpType::ADD,
                        std::vector<std::string>{"max_stable", "log_stable"});
                    
                    g.add_node(max_node);
                    g.add_node(sub_node);
                    g.add_node(exp_node);
                    g.add_node(sum_node);
                    g.add_node(log_node);
                    g.add_node(add_node);
                    g.add_input(x_it->second);
                    g.add_output("add_stable");
                }
                return g;
            }
        );
    }
    
    // Rule: exp(x)/sum(exp(x)) → stable_softmax(x)
    static RewriteRule naive_to_stable_softmax() {
        return RewriteRule(
            "stable_softmax",
            "Replace naive softmax with stable version",
            PatternLibrary::naive_softmax_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    // Use built-in stable softmax operation
                    auto softmax_node = std::make_shared<Node>(
                        "softmax_stable", OpType::STABLE_SOFTMAX,
                        std::vector<std::string>{x_it->second});
                    
                    g.add_node(softmax_node);
                    g.add_input(x_it->second);
                    g.add_output("softmax_stable");
                }
                return g;
            }
        );
    }
    
    // Rule: -log(softmax(x)) → log_softmax(x)
    static RewriteRule negative_log_softmax_fusion() {
        return RewriteRule(
            "log_softmax_fusion",
            "Fuse -log(softmax(x)) to log_softmax(x)",
            PatternLibrary::negative_log_softmax_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    auto log_softmax_node = std::make_shared<Node>(
                        "log_softmax_fused", OpType::LOG_SOFTMAX,
                        std::vector<std::string>{x_it->second});
                    
                    g.add_node(log_softmax_node);
                    g.add_input(x_it->second);
                    g.add_output("log_softmax_fused");
                }
                return g;
            }
        );
    }
    
    // Rule: sqrt(x^2) → abs(x)  (represented as max(x, -x))
    static RewriteRule square_sqrt_to_abs() {
        return RewriteRule(
            "square_sqrt_abs",
            "Replace sqrt(x^2) with abs(x)",
            PatternLibrary::square_sqrt_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    // abs(x) = max(x, -x)
                    auto neg_node = std::make_shared<Node>(
                        "neg_abs", OpType::NEG, std::vector<std::string>{x_it->second});
                    auto max_node = std::make_shared<Node>(
                        "max_abs", OpType::MAX,
                        std::vector<std::string>{x_it->second, "neg_abs"});
                    
                    g.add_node(neg_node);
                    g.add_node(max_node);
                    g.add_input(x_it->second);
                    g.add_output("max_abs");
                }
                return g;
            },
            // Condition: only apply if exponent is 2
            [](const Graph& graph, const auto& match) {
                auto two_it = match.find("$two");
                if (two_it == match.end()) return false;
                auto node = graph.get_node(two_it->second);
                if (!node || node->op != OpType::CONSTANT) return false;
                return std::abs(node->attrs.get_float("value", 0.0) - 2.0) < 1e-6;
            }
        );
    }
    
    // Get all standard rewrite rules
    static std::vector<RewriteRule> get_all_rules() {
        return {
            log_exp_cancel(),
            exp_log_cancel(),
            naive_to_stable_logsumexp(),
            naive_to_stable_softmax(),
            negative_log_softmax_fusion(),
            square_sqrt_to_abs()
        };
    }
    
    // Get stability-focused rules (rules that reduce curvature)
    static std::vector<RewriteRule> get_stability_rules() {
        return {
            naive_to_stable_logsumexp(),
            naive_to_stable_softmax(),
            negative_log_softmax_fusion()
        };
    }
    
    // Get algebraic simplification rules
    static std::vector<RewriteRule> get_simplification_rules() {
        return {
            log_exp_cancel(),
            exp_log_cancel(),
            square_sqrt_to_abs()
        };
    }
};

} // namespace rewriter
} // namespace hnf
