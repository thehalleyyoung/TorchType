#include "stability_linter.hpp"
#include <iostream>

using namespace hnf::stability_linter;

int main() {
    ComputationGraph graph;
    
    auto input = std::make_shared<Node>("input", OpType::PLACEHOLDER);
    auto log_node = std::make_shared<Node>("log", OpType::LOG);
    
    graph.add_node(input);
    graph.add_node(log_node);
    graph.add_edge("input", "log");
    
    std::cout << "Before propagation:" << std::endl;
    std::cout << "  Input range: [" << input->value_range.first << ", " << input->value_range.second << "]" << std::endl;
    std::cout << "  Log range: [" << log_node->value_range.first << ", " << log_node->value_range.second << "]" << std::endl;
    std::cout << "  Log curvature: " << log_node->curvature << std::endl;
    
    graph.propagate_ranges({1.0, 10.0});
    
    std::cout << "\nAfter propagation:" << std::endl;
    std::cout << "  Input range: [" << input->value_range.first << ", " << input->value_range.second << "]" << std::endl;
    std::cout << "  Log range: [" << log_node->value_range.first << ", " << log_node->value_range.second << "]" << std::endl;
    std::cout << "  Log curvature: " << log_node->curvature << std::endl;
    
    // Expected: κ = 1 / (1.0)^2 = 1.0
    std::cout << "\nExpected curvature: 1.0" << std::endl;
    
    return 0;
}
