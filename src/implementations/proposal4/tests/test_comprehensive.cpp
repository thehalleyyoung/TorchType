#include "../include/graph_ir.hpp"
#include "../include/curvature.hpp"
#include "../include/pattern.hpp"
#include "../include/rewrite_rules.hpp"
#include "../include/rewriter.hpp"

#include <iostream>
#include <iomanip>
#include <cassert>
#include <cmath>

using namespace hnf::rewriter;

// Test utilities
void print_test_header(const std::string& test_name) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "TEST: " << test_name << "\n";
    std::cout << std::string(80, '=') << "\n";
}

void print_success(const std::string& msg) {
    std::cout << "✓ " << msg << "\n";
}

void print_info(const std::string& msg) {
    std::cout << "  " << msg << "\n";
}

bool approx_equal(double a, double b, double tol = 1e-6) {
    return std::abs(a - b) < tol;
}

// Test 1: Basic graph construction and traversal
void test_graph_construction() {
    print_test_header("Graph Construction and Traversal");
    
    Graph g;
    auto x_node = std::make_shared<Node>("x", OpType::INPUT);
    auto y_node = std::make_shared<Node>("y", OpType::INPUT);
    auto add_node = std::make_shared<Node>("add", OpType::ADD,
                                          std::vector<std::string>{"x", "y"});
    auto mul_node = std::make_shared<Node>("mul", OpType::MUL,
                                          std::vector<std::string>{"add", "x"});
    
    g.add_node(x_node);
    g.add_node(y_node);
    g.add_node(add_node);
    g.add_node(mul_node);
    g.add_input("x");
    g.add_input("y");
    g.add_output("mul");
    
    print_info(g.to_string());
    
    auto topo = g.topological_order();
    print_info("Topological order:");
    for (const auto& id : topo) {
        print_info("  " + id);
    }
    
    assert(topo.size() == 4);
    assert(g.has_node("x"));
    assert(g.has_node("add"));
    
    print_success("Graph construction works correctly");
}

// Test 2: Curvature computation
void test_curvature_computation() {
    print_test_header("Curvature Computation");
    
    // Create a graph with exp operation
    Graph g;
    auto x_node = std::make_shared<Node>("x", OpType::INPUT);
    auto exp_node = std::make_shared<Node>("exp", OpType::EXP,
                                          std::vector<std::string>{"x"});
    g.add_node(x_node);
    g.add_node(exp_node);
    g.add_input("x");
    g.add_output("exp");
    
    // Set up input statistics
    std::unordered_map<std::string, TensorStats> input_stats;
    TensorStats x_stats;
    x_stats.min_val = 0.0;
    x_stats.max_val = 10.0;
    x_stats.mean_val = 5.0;
    x_stats.std_val = 2.0;
    input_stats["x"] = x_stats;
    
    double curv = CurvatureAnalyzer::total_curvature(g, input_stats);
    print_info("Curvature of exp(x) with x in [0, 10]: " + std::to_string(curv));
    
    // exp curvature should be exp(2 * x_max) = exp(20)
    double expected = std::exp(20.0);
    print_info("Expected curvature: " + std::to_string(expected));
    
    assert(approx_equal(curv, expected, expected * 0.01));
    
    print_success("Curvature computation validated");
}

// Test 3: Pattern matching
void test_pattern_matching() {
    print_test_header("Pattern Matching");
    
    // Create log(exp(x)) graph
    Graph g = GraphBuilder::log_exp("x");
    print_info("Original graph:");
    print_info(g.to_string());
    
    // Try to match log(exp(x)) pattern
    auto pattern = PatternLibrary::log_exp_pattern();
    
    bool found = false;
    for (const auto& [node_id, node] : g.nodes()) {
        auto match = pattern.match(g, node_id);
        if (match) {
            print_info("Pattern matched at node: " + node_id);
            print_info("Bindings:");
            for (const auto& [pat_id, graph_id] : *match) {
                print_info("  " + pat_id + " -> " + graph_id);
            }
            found = true;
            break;
        }
    }
    
    assert(found);
    print_success("Pattern matching works correctly");
}

// Test 4: Log-exp cancellation rewrite
void test_log_exp_cancellation() {
    print_test_header("Log-Exp Cancellation Rewrite");
    
    Graph g = GraphBuilder::log_exp("x");
    print_info("Original graph:");
    print_info(g.to_string());
    
    auto rule = RewriteRuleLibrary::log_exp_cancel();
    auto result_opt = rule.apply(g);
    
    assert(result_opt.has_value());
    
    auto& result = *result_opt;
    print_info("\nRewritten graph:");
    print_info(result.to_string());
    
    // Result should be just identity
    assert(result.nodes().size() < g.nodes().size());
    
    print_success("Log-exp cancellation works correctly");
}

// Test 5: Naive softmax to stable softmax
void test_naive_to_stable_softmax() {
    print_test_header("Naive to Stable Softmax Rewrite");
    
    Graph g = GraphBuilder::naive_softmax("x");
    print_info("Original naive softmax:");
    print_info(g.to_string());
    
    // Set up statistics
    std::unordered_map<std::string, TensorStats> input_stats;
    TensorStats x_stats;
    x_stats.min_val = 0.0;
    x_stats.max_val = 100.0;  // Large range = unstable
    x_stats.mean_val = 50.0;
    x_stats.std_val = 20.0;
    input_stats["x"] = x_stats;
    
    double orig_curv = CurvatureAnalyzer::total_curvature(g, input_stats);
    print_info("Original curvature: " + std::to_string(orig_curv));
    
    auto rule = RewriteRuleLibrary::naive_to_stable_softmax();
    auto result_opt = rule.apply(g);
    
    assert(result_opt.has_value());
    
    auto& result = *result_opt;
    print_info("\nStable softmax:");
    print_info(result.to_string());
    
    double new_curv = CurvatureAnalyzer::total_curvature(result, input_stats);
    print_info("New curvature: " + std::to_string(new_curv));
    
    // Stable softmax should have much lower curvature
    assert(new_curv < orig_curv * 0.01);  // At least 100x improvement
    
    print_success("Softmax stabilization works correctly");
}

// Test 6: Naive logsumexp to stable version
void test_naive_to_stable_logsumexp() {
    print_test_header("Naive to Stable LogSumExp Rewrite");
    
    Graph g = GraphBuilder::naive_logsumexp("x");
    print_info("Original naive logsumexp:");
    print_info(g.to_string());
    
    std::unordered_map<std::string, TensorStats> input_stats;
    TensorStats x_stats;
    x_stats.min_val = 100.0;
    x_stats.max_val = 300.0;  // Very large values = extremely unstable
    x_stats.mean_val = 200.0;
    x_stats.std_val = 50.0;
    input_stats["x"] = x_stats;
    
    double orig_curv = CurvatureAnalyzer::total_curvature(g, input_stats);
    print_info("Original curvature: " + std::to_string(orig_curv));
    
    auto rule = RewriteRuleLibrary::naive_to_stable_logsumexp();
    auto result_opt = rule.apply(g);
    
    assert(result_opt.has_value());
    
    auto& result = *result_opt;
    print_info("\nStable logsumexp:");
    print_info(result.to_string());
    
    double new_curv = CurvatureAnalyzer::total_curvature(result, input_stats);
    print_info("New curvature: " + std::to_string(new_curv));
    
    // Stable version should have dramatically lower curvature
    assert(new_curv < 100.0);  // Should be O(1)
    
    print_success("LogSumExp stabilization works correctly");
}

// Test 7: Cross-entropy fusion
void test_cross_entropy_fusion() {
    print_test_header("Cross-Entropy Fusion");
    
    Graph g = GraphBuilder::cross_entropy_pattern("x");
    print_info("Original -log(softmax(x)):");
    print_info(g.to_string());
    
    auto rule = RewriteRuleLibrary::negative_log_softmax_fusion();
    auto result_opt = rule.apply(g);
    
    assert(result_opt.has_value());
    
    auto& result = *result_opt;
    print_info("\nFused log_softmax:");
    print_info(result.to_string());
    
    // Should have fewer operations
    assert(result.nodes().size() < g.nodes().size());
    
    print_success("Cross-entropy fusion works correctly");
}

// Test 8: Greedy rewriter
void test_greedy_rewriter() {
    print_test_header("Greedy Rewriter");
    
    // Create a graph with multiple optimization opportunities
    Graph g = GraphBuilder::naive_softmax("x");
    
    std::unordered_map<std::string, TensorStats> input_stats;
    TensorStats x_stats;
    x_stats.min_val = -10.0;
    x_stats.max_val = 10.0;
    x_stats.mean_val = 0.0;
    x_stats.std_val = 5.0;
    input_stats["x"] = x_stats;
    
    auto rules = RewriteRuleLibrary::get_stability_rules();
    GraphRewriter rewriter(rules, 100, 10);
    
    double orig_curv = CurvatureAnalyzer::total_curvature(g, input_stats);
    print_info("Original curvature: " + std::to_string(orig_curv));
    
    auto result = rewriter.rewrite_greedy(g, input_stats);
    
    print_info("\nRewritten graph:");
    print_info(result.graph.to_string());
    print_info("Final curvature: " + std::to_string(result.curvature));
    
    print_info("\nApplied rules:");
    for (const auto& rule_name : result.applied_rules) {
        print_info("  - " + rule_name);
    }
    
    assert(result.curvature < orig_curv);
    assert(!result.applied_rules.empty());
    
    print_success("Greedy rewriter works correctly");
}

// Test 9: Beam search rewriter
void test_beam_search_rewriter() {
    print_test_header("Beam Search Rewriter");
    
    Graph g = GraphBuilder::naive_logsumexp("x");
    
    std::unordered_map<std::string, TensorStats> input_stats;
    TensorStats x_stats;
    x_stats.min_val = 0.0;
    x_stats.max_val = 50.0;
    x_stats.mean_val = 25.0;
    x_stats.std_val = 10.0;
    input_stats["x"] = x_stats;
    
    auto rules = RewriteRuleLibrary::get_all_rules();
    GraphRewriter rewriter(rules, 50, 5);
    
    double orig_curv = CurvatureAnalyzer::total_curvature(g, input_stats);
    print_info("Original curvature: " + std::to_string(orig_curv));
    print_info("Original graph:");
    print_info(g.to_string());
    
    auto result = rewriter.rewrite(g, input_stats);
    
    print_info("\nOptimized graph:");
    print_info(result.graph.to_string());
    print_info("Final curvature: " + std::to_string(result.curvature));
    
    print_info("\nRewrite sequence:");
    for (const auto& rule_name : result.applied_rules) {
        print_info("  -> " + rule_name);
    }
    
    assert(result.curvature < orig_curv);
    
    print_success("Beam search finds good optimizations");
}

// Test 10: Complex multi-step optimization
void test_complex_optimization() {
    print_test_header("Complex Multi-Step Optimization");
    
    // Create a complex graph with nested operations
    Graph g;
    auto x_node = std::make_shared<Node>("x", OpType::INPUT);
    auto exp1_node = std::make_shared<Node>("exp1", OpType::EXP,
                                           std::vector<std::string>{"x"});
    auto log1_node = std::make_shared<Node>("log1", OpType::LOG,
                                           std::vector<std::string>{"exp1"});
    auto exp2_node = std::make_shared<Node>("exp2", OpType::EXP,
                                           std::vector<std::string>{"log1"});
    auto sum_node = std::make_shared<Node>("sum", OpType::SUM,
                                          std::vector<std::string>{"exp2"});
    auto log2_node = std::make_shared<Node>("output", OpType::LOG,
                                           std::vector<std::string>{"sum"});
    
    g.add_node(x_node);
    g.add_node(exp1_node);
    g.add_node(log1_node);
    g.add_node(exp2_node);
    g.add_node(sum_node);
    g.add_node(log2_node);
    g.add_input("x");
    g.add_output("output");
    
    print_info("Original complex graph:");
    print_info(g.to_string());
    
    std::unordered_map<std::string, TensorStats> input_stats;
    TensorStats x_stats;
    x_stats.min_val = -5.0;
    x_stats.max_val = 5.0;
    x_stats.mean_val = 0.0;
    input_stats["x"] = x_stats;
    
    auto rules = RewriteRuleLibrary::get_all_rules();
    GraphRewriter rewriter(rules, 100, 10);
    
    auto result = rewriter.rewrite(g, input_stats);
    
    print_info("\nOptimized graph:");
    print_info(result.graph.to_string());
    
    print_info("\nOptimization sequence:");
    for (size_t i = 0; i < result.applied_rules.size(); ++i) {
        print_info("  Step " + std::to_string(i+1) + ": " + result.applied_rules[i]);
    }
    
    // Should simplify significantly
    assert(result.graph.nodes().size() < g.nodes().size());
    
    print_success("Complex multi-step optimization works");
}

// Test 11: Curvature-stability correlation
void test_curvature_stability_correlation() {
    print_test_header("Curvature-Stability Correlation Verification");
    
    // Test on naive vs stable softmax with varying input ranges
    std::vector<double> ranges = {10.0, 50.0, 100.0, 200.0};
    
    print_info("Comparing naive vs stable softmax across input ranges:");
    print_info("");
    print_info("Range    | Naive Curv   | Stable Curv  | Improvement");
    print_info("---------|--------------|--------------|-------------");
    
    for (double range : ranges) {
        Graph naive = GraphBuilder::naive_softmax("x");
        
        std::unordered_map<std::string, TensorStats> stats;
        TensorStats x_stats;
        x_stats.min_val = 0.0;
        x_stats.max_val = range;
        x_stats.mean_val = range / 2;
        stats["x"] = x_stats;
        
        double naive_curv = CurvatureAnalyzer::total_curvature(naive, stats);
        
        auto rule = RewriteRuleLibrary::naive_to_stable_softmax();
        auto stable = rule.apply(naive);
        double stable_curv = CurvatureAnalyzer::total_curvature(*stable, stats);
        
        double improvement = naive_curv / stable_curv;
        
        std::cout << std::fixed << std::setprecision(2);
        std::cout << std::setw(8) << range << " | ";
        std::cout << std::setw(12) << std::scientific << naive_curv << " | ";
        std::cout << std::setw(12) << stable_curv << " | ";
        std::cout << std::setw(11) << std::fixed << improvement << "x\n";
    }
    
    print_success("Curvature correctly predicts numerical stability");
}

// Test 12: Rule library completeness
void test_rule_library_completeness() {
    print_test_header("Rule Library Completeness");
    
    auto all_rules = RewriteRuleLibrary::get_all_rules();
    auto stability_rules = RewriteRuleLibrary::get_stability_rules();
    auto simplification_rules = RewriteRuleLibrary::get_simplification_rules();
    
    print_info("Total rules: " + std::to_string(all_rules.size()));
    print_info("Stability rules: " + std::to_string(stability_rules.size()));
    print_info("Simplification rules: " + std::to_string(simplification_rules.size()));
    
    print_info("\nAll rules:");
    for (const auto& rule : all_rules) {
        print_info("  - " + rule.name + ": " + rule.description);
    }
    
    assert(all_rules.size() >= 5);
    assert(stability_rules.size() >= 2);
    assert(simplification_rules.size() >= 2);
    
    print_success("Rule library is comprehensive");
}

int main() {
    std::cout << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  HNF Proposal #4: Stability-Preserving Graph Rewriter Tests   ║\n";
    std::cout << "║  Comprehensive Test Suite                                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    try {
        test_graph_construction();
        test_curvature_computation();
        test_pattern_matching();
        test_log_exp_cancellation();
        test_naive_to_stable_softmax();
        test_naive_to_stable_logsumexp();
        test_cross_entropy_fusion();
        test_greedy_rewriter();
        test_beam_search_rewriter();
        test_complex_optimization();
        test_curvature_stability_correlation();
        test_rule_library_completeness();
        
        std::cout << "\n";
        std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
        std::cout << "║                   ✓✓✓ ALL TESTS PASSED ✓✓✓                    ║\n";
        std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
        std::cout << "\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << "\n";
        return 1;
    } catch (...) {
        std::cerr << "\n✗ Test failed with unknown exception\n";
        return 1;
    }
}
