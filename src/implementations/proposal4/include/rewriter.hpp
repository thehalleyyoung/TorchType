#pragma once

#include "graph_ir.hpp"
#include "curvature.hpp"
#include "rewrite_rules.hpp"
#include <queue>
#include <set>
#include <algorithm>

namespace hnf {
namespace rewriter {

// Result of a rewrite operation
struct RewriteResult {
    Graph graph;
    double curvature;
    std::vector<std::string> applied_rules;
    
    RewriteResult(const Graph& g, double c) : graph(g), curvature(c) {}
};

// Comparison for priority queue (min-heap by curvature)
struct RewriteResultCompare {
    bool operator()(const std::shared_ptr<RewriteResult>& a,
                   const std::shared_ptr<RewriteResult>& b) const {
        return a->curvature > b->curvature;  // Min-heap
    }
};

// Main graph rewriter using beam search
class GraphRewriter {
private:
    std::vector<RewriteRule> rules_;
    int max_iterations_;
    int beam_width_;
    
public:
    GraphRewriter(const std::vector<RewriteRule>& rules,
                 int max_iterations = 100,
                 int beam_width = 10)
        : rules_(rules), max_iterations_(max_iterations), beam_width_(beam_width) {}
    
    // Rewrite graph to minimize curvature using beam search
    RewriteResult rewrite(const Graph& initial_graph,
                         const std::unordered_map<std::string, TensorStats>& input_stats) {
        
        double initial_curv = CurvatureAnalyzer::total_curvature(initial_graph, input_stats);
        
        auto best = std::make_shared<RewriteResult>(initial_graph, initial_curv);
        
        // Priority queue for beam search (min-heap by curvature)
        std::priority_queue<std::shared_ptr<RewriteResult>,
                          std::vector<std::shared_ptr<RewriteResult>>,
                          RewriteResultCompare> beam;
        
        beam.push(best);
        
        // Track seen graphs to avoid cycles (using simple hash of structure)
        std::set<std::string> seen;
        seen.insert(graph_hash(initial_graph));
        
        for (int iter = 0; iter < max_iterations_; ++iter) {
            if (beam.empty()) break;
            
            // Expand beam
            std::vector<std::shared_ptr<RewriteResult>> candidates;
            
            // Process current beam
            std::vector<std::shared_ptr<RewriteResult>> current_beam;
            while (!beam.empty() && static_cast<int>(current_beam.size()) < beam_width_) {
                current_beam.push_back(beam.top());
                beam.pop();
            }
            
            // Try applying each rule to each graph in beam
            for (const auto& result : current_beam) {
                for (const auto& rule : rules_) {
                    auto new_graph_opt = rule.apply(result->graph);
                    
                    if (new_graph_opt) {
                        auto& new_graph = *new_graph_opt;
                        std::string hash = graph_hash(new_graph);
                        
                        if (seen.count(hash)) {
                            continue;  // Already seen this graph
                        }
                        seen.insert(hash);
                        
                        double new_curv = CurvatureAnalyzer::total_curvature(new_graph, input_stats);
                        
                        auto new_result = std::make_shared<RewriteResult>(new_graph, new_curv);
                        new_result->applied_rules = result->applied_rules;
                        new_result->applied_rules.push_back(rule.name);
                        
                        candidates.push_back(new_result);
                        
                        // Update best if improved
                        if (new_curv < best->curvature) {
                            best = new_result;
                        }
                    }
                }
            }
            
            if (candidates.empty()) {
                break;  // No more rewrites possible
            }
            
            // Sort candidates by curvature and keep top beam_width_
            std::sort(candidates.begin(), candidates.end(),
                     [](const auto& a, const auto& b) {
                         return a->curvature < b->curvature;
                     });
            
            // Rebuild beam with best candidates
            for (int i = 0; i < std::min(beam_width_, static_cast<int>(candidates.size())); ++i) {
                beam.push(candidates[i]);
            }
        }
        
        return *best;
    }
    
    // Greedy rewrite: apply rules in order until no more improvements
    RewriteResult rewrite_greedy(const Graph& initial_graph,
                                const std::unordered_map<std::string, TensorStats>& input_stats) {
        
        Graph current = initial_graph;
        double current_curv = CurvatureAnalyzer::total_curvature(current, input_stats);
        std::vector<std::string> applied_rules;
        
        bool improved = true;
        int iterations = 0;
        
        while (improved && iterations < max_iterations_) {
            improved = false;
            iterations++;
            
            for (const auto& rule : rules_) {
                auto new_graph_opt = rule.apply(current);
                
                if (new_graph_opt) {
                    double new_curv = CurvatureAnalyzer::total_curvature(*new_graph_opt, input_stats);
                    
                    if (new_curv < current_curv) {
                        current = *new_graph_opt;
                        current_curv = new_curv;
                        applied_rules.push_back(rule.name);
                        improved = true;
                        break;  // Restart with new graph
                    }
                }
            }
        }
        
        RewriteResult result(current, current_curv);
        result.applied_rules = applied_rules;
        return result;
    }
    
    // Compute difference between two graphs (for reporting)
    std::string diff(const Graph& original, const Graph& rewritten) const {
        std::stringstream ss;
        
        // Compare node counts
        ss << "Original nodes: " << original.nodes().size() << "\n";
        ss << "Rewritten nodes: " << rewritten.nodes().size() << "\n";
        
        // Find nodes that changed
        std::unordered_set<OpType> orig_ops, rewr_ops;
        for (const auto& [id, node] : original.nodes()) {
            orig_ops.insert(node->op);
        }
        for (const auto& [id, node] : rewritten.nodes()) {
            rewr_ops.insert(node->op);
        }
        
        ss << "\nOperation changes:\n";
        for (const auto& op : orig_ops) {
            if (!rewr_ops.count(op)) {
                ss << "  - Removed: " << optype_to_string(op) << "\n";
            }
        }
        for (const auto& op : rewr_ops) {
            if (!orig_ops.count(op)) {
                ss << "  + Added: " << optype_to_string(op) << "\n";
            }
        }
        
        return ss.str();
    }
    
private:
    // Simple hash of graph structure for cycle detection
    std::string graph_hash(const Graph& g) const {
        std::stringstream ss;
        
        auto topo = g.topological_order();
        for (const auto& node_id : topo) {
            auto node = g.get_node(node_id);
            if (node) {
                ss << optype_to_string(node->op) << ":";
                for (const auto& inp : node->inputs) {
                    ss << inp << ",";
                }
                ss << ";";
            }
        }
        
        return ss.str();
    }
};

// Utility: Create example graphs for testing
class GraphBuilder {
public:
    // Build naive softmax graph: exp(x) / sum(exp(x))
    static Graph naive_softmax(const std::string& input_id = "x") {
        Graph g;
        
        auto x_node = std::make_shared<Node>(input_id, OpType::INPUT);
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP,
                                              std::vector<std::string>{input_id});
        auto sum_node = std::make_shared<Node>("sum", OpType::SUM,
                                              std::vector<std::string>{"exp"});
        auto div_node = std::make_shared<Node>("output", OpType::DIV,
                                              std::vector<std::string>{"exp", "sum"});
        
        g.add_node(x_node);
        g.add_node(exp_node);
        g.add_node(sum_node);
        g.add_node(div_node);
        g.add_input(input_id);
        g.add_output("output");
        
        return g;
    }
    
    // Build naive logsumexp graph: log(sum(exp(x)))
    static Graph naive_logsumexp(const std::string& input_id = "x") {
        Graph g;
        
        auto x_node = std::make_shared<Node>(input_id, OpType::INPUT);
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP,
                                              std::vector<std::string>{input_id});
        auto sum_node = std::make_shared<Node>("sum", OpType::SUM,
                                              std::vector<std::string>{"exp"});
        auto log_node = std::make_shared<Node>("output", OpType::LOG,
                                              std::vector<std::string>{"sum"});
        
        g.add_node(x_node);
        g.add_node(exp_node);
        g.add_node(sum_node);
        g.add_node(log_node);
        g.add_input(input_id);
        g.add_output("output");
        
        return g;
    }
    
    // Build log(exp(x)) graph (should simplify to x)
    static Graph log_exp(const std::string& input_id = "x") {
        Graph g;
        
        auto x_node = std::make_shared<Node>(input_id, OpType::INPUT);
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP,
                                              std::vector<std::string>{input_id});
        auto log_node = std::make_shared<Node>("output", OpType::LOG,
                                              std::vector<std::string>{"exp"});
        
        g.add_node(x_node);
        g.add_node(exp_node);
        g.add_node(log_node);
        g.add_input(input_id);
        g.add_output("output");
        
        return g;
    }
    
    // Build cross-entropy pattern: -log(softmax(x))
    static Graph cross_entropy_pattern(const std::string& input_id = "x") {
        Graph g;
        
        auto x_node = std::make_shared<Node>(input_id, OpType::INPUT);
        auto softmax_node = std::make_shared<Node>("softmax", OpType::SOFTMAX,
                                                   std::vector<std::string>{input_id});
        auto log_node = std::make_shared<Node>("log", OpType::LOG,
                                              std::vector<std::string>{"softmax"});
        auto neg_node = std::make_shared<Node>("output", OpType::NEG,
                                              std::vector<std::string>{"log"});
        
        g.add_node(x_node);
        g.add_node(softmax_node);
        g.add_node(log_node);
        g.add_node(neg_node);
        g.add_input(input_id);
        g.add_output("output");
        
        return g;
    }
};

} // namespace rewriter
} // namespace hnf
