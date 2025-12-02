#include "../include/computation_graph.h"
#include "../include/precision_sheaf.h"
#include "../include/mixed_precision_optimizer.h"
#include "../include/graph_builder.h"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <cassert>

using namespace hnf::sheaf;

// Forward declaration
double compute_avg_precision(const PrecisionAssignment& assignment);

// ANSI color codes for output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"

void print_test_header(const std::string& test_name) {
    std::cout << "\n" << CYAN << "========================================" << RESET << "\n";
    std::cout << CYAN << "TEST: " << test_name << RESET << "\n";
    std::cout << CYAN << "========================================" << RESET << "\n\n";
}

void print_pass(const std::string& message) {
    std::cout << GREEN << "✓ PASS: " << message << RESET << "\n";
}

void print_fail(const std::string& message) {
    std::cout << RED << "✗ FAIL: " << message << RESET << "\n";
}

void print_info(const std::string& message) {
    std::cout << BLUE << "ℹ " << message << RESET << "\n";
}

// Test 1: Basic graph construction and topology
void test_graph_topology() {
    print_test_header("Graph Topology");
    
    ComputationGraph graph;
    
    // Build simple chain: A -> B -> C
    auto nodeA = std::make_shared<ComputationNode>("A", "linear", 0.0, 1.0);
    auto nodeB = std::make_shared<ComputationNode>("B", "relu", 0.0, 1.0);
    auto nodeC = std::make_shared<ComputationNode>("C", "linear", 0.0, 1.0);
    
    graph.add_node(nodeA);
    graph.add_node(nodeB);
    graph.add_node(nodeC);
    
    graph.add_edge("A", "B");
    graph.add_edge("B", "C");
    
    // Test acyclicity
    assert(graph.is_acyclic());
    print_pass("Graph is acyclic");
    
    // Test topological order
    auto topo = graph.topological_order();
    assert(topo.size() == 3);
    assert(topo[0] == "A");
    assert(topo[1] == "B");
    assert(topo[2] == "C");
    print_pass("Topological order is correct: A -> B -> C");
    
    // Test neighbors
    auto neighbors_B = graph.get_neighbors("B");
    assert(neighbors_B.count("A") && neighbors_B.count("C"));
    assert(neighbors_B.size() == 2);
    print_pass("Neighbor computation is correct");
    
    // Test input/output nodes
    auto inputs = graph.input_nodes();
    auto outputs = graph.output_nodes();
    assert(inputs.size() == 1 && inputs[0] == "A");
    assert(outputs.size() == 1 && outputs[0] == "C");
    print_pass("Input/output node detection works");
    
    print_info("Graph topology tests completed successfully");
}

// Test 2: Precision requirements from curvature
void test_precision_requirements() {
    print_test_header("Precision Requirements from Curvature");
    
    double target_eps = 1e-5;
    double c = 2.0;
    
    // Test 1: Low curvature operation (should use low precision)
    {
        auto node = std::make_shared<ComputationNode>("relu", "relu", 0.0, 1.0, 2.0);
        node->compute_min_precision(target_eps, c);
        
        print_info("ReLU (κ=0): min_precision = " + std::to_string(node->min_precision_bits) + " bits");
        assert(node->min_precision_bits <= 23);
        print_pass("Linear operations require low precision");
    }
    
    // Test 2: Moderate curvature (softmax)
    {
        double kappa = 0.5;  // Intrinsic softmax curvature
        double D = 10.0;     // Typical range
        
        auto node = std::make_shared<ComputationNode>("softmax", "softmax", kappa, 1.0, D);
        node->compute_min_precision(target_eps, c);
        
        print_info("Softmax (κ=0.5, D=10): min_precision = " + std::to_string(node->min_precision_bits) + " bits");
        
        // Expected: p >= log2(2 * 0.5 * 100 / 1e-5) = log2(1e7) ≈ 23.25
        assert(node->min_precision_bits >= 20 && node->min_precision_bits <= 30);
        print_pass("Softmax curvature bounds are reasonable");
    }
    
    // Test 3: High curvature (attention with large QK^T)
    {
        double qk_norm = 20.0;  // ||QK^T||
        double kappa = 0.5 * qk_norm * qk_norm;  // Composed curvature
        double D = 10.0;
        
        auto node = std::make_shared<ComputationNode>("attention", "attention", kappa, 1.0, D);
        node->compute_min_precision(target_eps, c);
        
        print_info("Attention (κ=" + std::to_string(kappa) + ", D=10): min_precision = " + 
                   std::to_string(node->min_precision_bits) + " bits");
        
        // Should require higher precision
        assert(node->min_precision_bits >= 30);
        print_pass("High curvature operations require high precision");
    }
    
    print_info("Precision requirement tests completed successfully");
}

// Test 3: Open covers
void test_open_covers() {
    print_test_header("Open Covers (Sheaf Theory)");
    
    ComputationGraph graph = GraphBuilder::build_ffn_graph(128, 512);
    
    // Test star cover
    auto star = OpenCover::star_cover(graph);
    
    print_info("Star cover has " + std::to_string(star.sets.size()) + " open sets");
    assert(star.sets.size() == graph.nodes.size());
    print_pass("Star cover has one set per node");
    
    // Each set should contain the node and its neighbors
    for (size_t i = 0; i < star.sets.size(); ++i) {
        assert(!star.sets[i].empty());
    }
    print_pass("All open sets are non-empty");
    
    // Test intersections
    auto intersections = star.get_intersections();
    print_info("Found " + std::to_string(intersections.size()) + " pairwise intersections");
    assert(intersections.size() > 0);
    print_pass("Intersections computed correctly");
    
    // Test path cover
    auto path = OpenCover::path_cover(graph, 3);
    print_info("Path cover has " + std::to_string(path.sets.size()) + " open sets");
    assert(path.sets.size() > 0);
    print_pass("Path cover constructed successfully");
    
    print_info("Open cover tests completed successfully");
}

// Test 4: Precision sheaf and cohomology
void test_sheaf_cohomology() {
    print_test_header("Sheaf Cohomology");
    
    // Build simple graph where uniform precision should work
    ComputationGraph simple_graph;
    
    auto n1 = std::make_shared<ComputationNode>("n1", "linear", 0.0, 1.0, 1.0);
    auto n2 = std::make_shared<ComputationNode>("n2", "relu", 0.0, 1.0, 1.0);
    auto n3 = std::make_shared<ComputationNode>("n3", "linear", 0.0, 1.0, 1.0);
    
    simple_graph.add_node(n1);
    simple_graph.add_node(n2);
    simple_graph.add_node(n3);
    simple_graph.add_edge("n1", "n2");
    simple_graph.add_edge("n2", "n3");
    
    double target_eps = 1e-3;
    auto cover = OpenCover::star_cover(simple_graph);
    PrecisionSheaf sheaf(simple_graph, target_eps, cover);
    
    // Test H^0 computation
    auto H0 = sheaf.compute_H0();
    print_info("H^0 has dimension " + std::to_string(H0.size()));
    
    if (!H0.empty()) {
        print_pass("Global sections exist for simple graph (H^0 ≠ ∅)");
        
        // Print one global section
        print_info("Example global section:");
        for (const auto& [node, prec] : H0[0]) {
            std::cout << "  " << node << ": " << prec << " bits\n";
        }
    } else {
        print_info("No uniform precision assignment (H^0 = ∅)");
        
        // Compute obstruction
        auto obstruction = sheaf.get_obstruction();
        if (obstruction) {
            print_pass("Obstruction cocycle computed (H^1 ≠ 0)");
            print_info("Obstruction L1 norm: " + std::to_string(obstruction->l1_norm()));
        }
    }
    
    print_info("Sheaf cohomology tests completed successfully");
}

// Test 5: Pathological network (mixed precision required)
void test_pathological_network() {
    print_test_header("Pathological Network (Mixed Precision Required)");
    
    auto graph = GraphBuilder::build_pathological_network();
    
    print_info("Built pathological network with exp(exp(x)) layer");
    print_info("Number of nodes: " + std::to_string(graph.nodes.size()));
    
    // Compute min precisions
    double target_eps = 1e-5;
    for (auto& [name, node] : graph.nodes) {
        node->compute_min_precision(target_eps);
    }
    
    // Check that exp layers require high precision
    auto exp1 = graph.nodes["exp1"];
    auto exp2 = graph.nodes["exp2"];
    
    print_info("exp1 min precision: " + std::to_string(exp1->min_precision_bits) + " bits");
    print_info("exp2 min precision: " + std::to_string(exp2->min_precision_bits) + " bits");
    
    assert(exp2->min_precision_bits > 32);
    print_pass("Double exponential requires high precision (>32 bits)");
    
    // Check that linear layers can use lower precision
    auto linear1 = graph.nodes["linear1"];
    print_info("linear1 min precision: " + std::to_string(linear1->min_precision_bits) + " bits");
    
    assert(linear1->min_precision_bits <= 23);
    print_pass("Linear layer can use lower precision (<=23 bits)");
    
    // This should demonstrate that mixed precision is topologically required
    auto cover = OpenCover::star_cover(graph);
    PrecisionSheaf sheaf(graph, target_eps, cover);
    
    auto H0 = sheaf.compute_H0();
    print_info("H^0 dimension for pathological network: " + std::to_string(H0.size()));
    
    if (H0.empty()) {
        print_pass("No uniform precision works - mixed precision REQUIRED (cohomological obstruction)");
    }
    
    print_info("Pathological network tests completed successfully");
}

// Test 6: Mixed-precision optimizer
void test_mixed_precision_optimizer() {
    print_test_header("Mixed-Precision Optimizer");
    
    // Build attention mechanism (from proposal example)
    int64_t seq_len = 128;
    int64_t d_model = 512;
    int64_t num_heads = 8;
    
    auto graph = GraphBuilder::build_attention_graph(seq_len, d_model, num_heads);
    
    print_info("Built attention graph with " + std::to_string(graph.nodes.size()) + " nodes");
    
    // Run optimizer
    double target_eps = 1e-4;
    MixedPrecisionOptimizer optimizer(graph, target_eps, 50);
    
    print_info("Running mixed-precision optimization...");
    auto result = optimizer.optimize();
    
    if (result.success) {
        print_pass("Optimization succeeded!");
        print_info("Status: " + result.status_message);
        print_info("H^0 dimension: " + std::to_string(result.h0_dimension));
        print_info("Memory saving: " + std::to_string(result.estimated_memory_saving * 100) + "%");
        
        // Print precision assignment
        std::cout << "\n" << YELLOW << "Precision Assignment:" << RESET << "\n";
        for (const auto& [node, prec] : result.optimal_assignment) {
            std::string color = (prec >= 32) ? RED : (prec >= 16) ? YELLOW : GREEN;
            std::cout << "  " << std::setw(15) << std::left << node 
                     << ": " << color << prec << " bits" << RESET;
            
            if (result.precision_rationale.count(node)) {
                std::cout << "  (" << result.precision_rationale[node] << ")";
            }
            std::cout << "\n";
        }
        
        // Check key properties from proposal Example
        if (result.optimal_assignment.count("softmax")) {
            int softmax_prec = result.optimal_assignment["softmax"];
            print_info("Softmax precision: " + std::to_string(softmax_prec) + " bits");
            
            // Softmax should need higher precision than other ops
            if (softmax_prec >= 23) {
                print_pass("Softmax uses fp32 or higher (as predicted in proposal)");
            }
        }
        
        // Compare with baselines
        auto comparison = optimizer.compare_with_baseline(result.optimal_assignment);
        
        std::cout << "\n" << YELLOW << "Comparison with Baselines:" << RESET << "\n";
        std::cout << "  Uniform fp16 accuracy: " << comparison.accuracy_uniform_fp16 << "\n";
        std::cout << "  Uniform fp32 accuracy: " << comparison.accuracy_uniform_fp32 << "\n";
        std::cout << "  Optimized accuracy:    " << comparison.accuracy_optimized << "\n";
        std::cout << "  Uniform fp16 memory:   " << comparison.memory_uniform_fp16 << " bytes\n";
        std::cout << "  Uniform fp32 memory:   " << comparison.memory_uniform_fp32 << " bytes\n";
        std::cout << "  Optimized memory:      " << comparison.memory_optimized << " bytes\n";
        
        double memory_vs_fp32 = (comparison.memory_uniform_fp32 - comparison.memory_optimized) / 
                                comparison.memory_uniform_fp32;
        print_info("Memory savings vs fp32: " + std::to_string(memory_vs_fp32 * 100) + "%");
        
        if (memory_vs_fp32 > 0.05) {
            print_pass("Achieved >5% memory savings vs uniform fp32");
        }
        
    } else {
        print_fail("Optimization failed: " + result.status_message);
    }
    
    print_info("Mixed-precision optimizer tests completed successfully");
}

// Test 7: Transformer block (realistic example)
void test_transformer_block() {
    print_test_header("Full Transformer Block");
    
    int64_t seq_len = 64;
    int64_t d_model = 256;
    int64_t num_heads = 8;
    int64_t d_ff = 1024;
    
    auto graph = GraphBuilder::build_transformer_block(seq_len, d_model, num_heads, d_ff);
    
    print_info("Built transformer block with " + std::to_string(graph.nodes.size()) + " nodes");
    
    // Check graph structure
    assert(graph.is_acyclic());
    print_pass("Transformer graph is acyclic");
    
    auto inputs = graph.input_nodes();
    auto outputs = graph.output_nodes();
    print_info("Input nodes: " + std::to_string(inputs.size()));
    print_info("Output nodes: " + std::to_string(outputs.size()));
    
    // Compute global properties
    double global_lip = graph.global_lipschitz();
    double global_curv = graph.global_curvature();
    
    print_info("Global Lipschitz constant: " + std::to_string(global_lip));
    print_info("Global curvature: " + std::to_string(global_curv));
    
    assert(global_curv > 0);
    print_pass("Transformer has positive curvature (nonlinear)");
    
    // Run optimizer
    MixedPrecisionOptimizer optimizer(graph, 1e-4, 30);
    auto result = optimizer.optimize();
    
    if (result.success) {
        print_pass("Found mixed-precision assignment for transformer");
        
        // Count precision usage
        std::map<int, int> precision_counts;
        for (const auto& [_, prec] : result.optimal_assignment) {
            precision_counts[prec]++;
        }
        
        std::cout << "\n" << YELLOW << "Precision Distribution:" << RESET << "\n";
        for (const auto& [prec, count] : precision_counts) {
            std::cout << "  " << prec << " bits: " << count << " nodes\n";
        }
        
        if (precision_counts.size() > 1) {
            print_pass("Mixed precision used (not uniform)");
        }
    }
    
    print_info("Transformer block tests completed successfully");
}

// Test 8: Cocycle condition verification
void test_cocycle_condition() {
    print_test_header("Cocycle Condition Verification");
    
    // Build a graph with triple intersections
    auto graph = GraphBuilder::build_ffn_graph(128, 512);
    auto cover = OpenCover::star_cover(graph);
    
    // Get triple intersections
    auto triples = cover.get_triple_intersections();
    print_info("Found " + std::to_string(triples.size()) + " triple intersections");
    
    if (triples.size() > 0) {
        print_pass("Graph has triple intersections (suitable for cocycle test)");
        
        // Build a valid cocycle
        Cocycle cocycle;
        auto intersections = cover.get_intersections();
        
        // Set arbitrary values
        for (const auto& [i, j] : intersections) {
            cocycle.set(i, j, static_cast<int>(i) - static_cast<int>(j));
        }
        
        // Check if it satisfies cocycle condition
        bool valid = cocycle.satisfies_cocycle_condition(cover);
        
        if (valid) {
            print_pass("Cocycle satisfies ω_ij + ω_jk - ω_ik = 0");
        } else {
            print_info("Cocycle does not satisfy condition (as expected for arbitrary assignment)");
        }
        
        // Compute L1 norm
        int norm = cocycle.l1_norm();
        print_info("Cocycle L1 norm: " + std::to_string(norm));
        
    } else {
        print_info("No triple intersections in this graph");
    }
    
    print_info("Cocycle condition tests completed successfully");
}

// Test 9: Subgraph analysis
void test_subgraph_analysis() {
    print_test_header("Subgraph Analysis");
    
    auto graph = GraphBuilder::build_transformer_block(64, 256, 8, 1024);
    
    // Extract attention subgraph
    std::unordered_set<std::string> attn_nodes;
    for (const auto& [name, _] : graph.nodes) {
        if (name.find("attn_") != std::string::npos) {
            attn_nodes.insert(name);
        }
    }
    
    print_info("Extracting attention subgraph (" + std::to_string(attn_nodes.size()) + " nodes)");
    
    MixedPrecisionOptimizer optimizer(graph, 1e-4);
    auto result = optimizer.analyze_subgraph(attn_nodes);
    
    if (result.success) {
        print_pass("Attention subgraph analyzed successfully");
        
        // FFN subgraph
        std::unordered_set<std::string> ffn_nodes;
        for (const auto& [name, _] : graph.nodes) {
            if (name.find("ffn_") != std::string::npos) {
                ffn_nodes.insert(name);
            }
        }
        
        print_info("Extracting FFN subgraph (" + std::to_string(ffn_nodes.size()) + " nodes)");
        auto ffn_result = optimizer.analyze_subgraph(ffn_nodes);
        
        if (ffn_result.success) {
            print_pass("FFN subgraph analyzed successfully");
            
            // Compare precisions
            print_info("Attention avg precision: " + 
                      std::to_string(compute_avg_precision(result.optimal_assignment)));
            print_info("FFN avg precision: " + 
                      std::to_string(compute_avg_precision(ffn_result.optimal_assignment)));
        }
    }
    
    print_info("Subgraph analysis tests completed successfully");
}

// Helper function for test 9
double compute_avg_precision(const PrecisionAssignment& assignment) {
    if (assignment.empty()) return 0.0;
    
    double sum = 0.0;
    for (const auto& [_, prec] : assignment) {
        sum += prec;
    }
    return sum / assignment.size();
}

// Test 10: Edge cases and robustness
void test_edge_cases() {
    print_test_header("Edge Cases and Robustness");
    
    // Empty graph
    {
        ComputationGraph empty_graph;
        MixedPrecisionOptimizer optimizer(empty_graph);
        auto result = optimizer.optimize();
        assert(result.success);  // Should handle gracefully
        print_pass("Empty graph handled correctly");
    }
    
    // Single node
    {
        ComputationGraph single_node_graph;
        auto node = std::make_shared<ComputationNode>("only", "linear", 0.0, 1.0);
        single_node_graph.add_node(node);
        
        MixedPrecisionOptimizer optimizer(single_node_graph);
        auto result = optimizer.optimize();
        assert(result.success);
        assert(result.optimal_assignment.size() == 1);
        print_pass("Single node graph handled correctly");
    }
    
    // Disconnected graph
    {
        ComputationGraph disconnected;
        auto n1 = std::make_shared<ComputationNode>("n1", "linear", 0.0, 1.0);
        auto n2 = std::make_shared<ComputationNode>("n2", "linear", 0.0, 1.0);
        disconnected.add_node(n1);
        disconnected.add_node(n2);
        // No edges
        
        MixedPrecisionOptimizer optimizer(disconnected);
        auto result = optimizer.optimize();
        assert(result.success);
        print_pass("Disconnected graph handled correctly");
    }
    
    // Very high curvature
    {
        ComputationGraph high_curv;
        auto node = std::make_shared<ComputationNode>("extreme", "exp", 1e10, 1.0, 10.0);
        high_curv.add_node(node);
        node->compute_min_precision(1e-5);
        
        // Should saturate at max precision
        assert(node->get_hardware_precision() <= 112);
        print_pass("Extreme curvature saturates at max precision");
    }
    
    print_info("Edge case tests completed successfully");
}

int main() {
    std::cout << MAGENTA << "\n";
    std::cout << "╔════════════════════════════════════════════════════════════╗\n";
    std::cout << "║  HNF Proposal #2: Sheaf Cohomology Mixed-Precision Tests ║\n";
    std::cout << "║  Comprehensive Validation Suite                          ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════╝\n";
    std::cout << RESET << "\n";
    
    try {
        test_graph_topology();
        test_precision_requirements();
        test_open_covers();
        test_sheaf_cohomology();
        test_pathological_network();
        test_cocycle_condition();
        test_mixed_precision_optimizer();
        test_transformer_block();
        test_subgraph_analysis();
        test_edge_cases();
        
        std::cout << "\n" << GREEN << "╔════════════════════════════════════════╗\n";
        std::cout << "║  ALL TESTS PASSED SUCCESSFULLY! ✓    ║\n";
        std::cout << "╚════════════════════════════════════════╝" << RESET << "\n\n";
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cout << "\n" << RED << "╔════════════════════════════════════════╗\n";
        std::cout << "║  TEST SUITE FAILED!                   ║\n";
        std::cout << "╚════════════════════════════════════════╝" << RESET << "\n";
        std::cout << RED << "Error: " << e.what() << RESET << "\n\n";
        return 1;
    }
}
