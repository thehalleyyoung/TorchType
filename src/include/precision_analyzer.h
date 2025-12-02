#pragma once

#include "numerical_type.h"
#include "curvature_database.h"
#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>
#include <memory>

namespace hnf {

// Operation node in computation graph
struct OperationNode {
    std::string name;
    std::string op_type;
    NumericalType output;
    std::vector<std::shared_ptr<OperationNode>> inputs;
    
    double curvature;
    double lipschitz_constant;
    int precision_bits_required;
    HardwareModel recommended_hardware;
    
    OperationNode(
        const std::string& n,
        const std::string& type,
        const NumericalType& out
    ) : name(n), op_type(type), output(out) {
        curvature = out.curvature;
        lipschitz_constant = out.lipschitz_constant;
        precision_bits_required = out.precision_bits_required;
        recommended_hardware = out.recommend_hardware();
    }
};

// Computation Graph representing a traced model
class ComputationGraph {
public:
    std::vector<std::shared_ptr<OperationNode>> nodes;
    std::map<std::string, std::shared_ptr<OperationNode>> node_map;
    
    void add_node(std::shared_ptr<OperationNode> node) {
        nodes.push_back(node);
        node_map[node->name] = node;
    }
    
    // Compute global error bound using Theorem 3.8 (Stability Composition Theorem)
    // Φ_{f_n ∘ ... ∘ f_1}(ε) ≤ Σᵢ (Πⱼ₌ᵢ₊₁ⁿ Lⱼ) · Φᵢ(εᵢ)
    double compute_total_error(double input_error, HardwareModel H) const {
        if (nodes.empty()) return 0.0;
        
        double total_error = input_error;
        
        for (const auto& node : nodes) {
            // Accumulate error through this operation
            total_error = node->output.propagate_error(total_error);
        }
        
        return total_error;
    }
    
    // Find nodes that require precision beyond what's available
    std::vector<std::shared_ptr<OperationNode>> find_critical_nodes(HardwareModel H) const {
        std::vector<std::shared_ptr<OperationNode>> critical;
        int available_bits = mantissa_precision(H);
        
        for (const auto& node : nodes) {
            if (node->precision_bits_required > available_bits) {
                critical.push_back(node);
            }
        }
        
        return critical;
    }
    
    // Generate mixed-precision recommendations
    std::map<std::string, HardwareModel> generate_precision_config() const {
        std::map<std::string, HardwareModel> config;
        
        for (const auto& node : nodes) {
            config[node->name] = node->recommended_hardware;
        }
        
        return config;
    }
    
    // Compute maximum curvature in the graph
    double max_curvature() const {
        double max_curv = 0.0;
        for (const auto& node : nodes) {
            if (node->curvature > max_curv && !std::isinf(node->curvature)) {
                max_curv = node->curvature;
            }
        }
        return max_curv;
    }
    
    // Compute total Lipschitz constant (product through chain)
    double total_lipschitz() const {
        double L_total = 1.0;
        for (const auto& node : nodes) {
            L_total *= node->lipschitz_constant;
        }
        return L_total;
    }
    
    // Pretty print the graph with precision information
    std::string to_string() const {
        std::ostringstream oss;
        oss << "Computation Graph:\n";
        oss << "================================================================================\n";
        oss << std::left << std::setw(20) << "Operation"
            << std::setw(12) << "Type"
            << std::setw(15) << "Curvature"
            << std::setw(10) << "Bits Req."
            << std::setw(12) << "Recommend\n";
        oss << "--------------------------------------------------------------------------------\n";
        
        for (const auto& node : nodes) {
            oss << std::left << std::setw(20) << node->name
                << std::setw(12) << node->op_type
                << std::setw(15);
            
            if (std::isinf(node->curvature)) {
                oss << "∞";
            } else if (node->curvature > 1e6) {
                oss << std::scientific << std::setprecision(2) << node->curvature;
            } else {
                oss << std::fixed << std::setprecision(2) << node->curvature;
            }
            
            oss << std::setw(10) << node->precision_bits_required
                << std::setw(12);
            
            switch(node->recommended_hardware) {
                case HardwareModel::BFLOAT16: oss << "bfloat16"; break;
                case HardwareModel::FLOAT16: oss << "float16"; break;
                case HardwareModel::FLOAT32: oss << "float32"; break;
                case HardwareModel::FLOAT64: oss << "float64"; break;
                case HardwareModel::FLOAT128: oss << "float128"; break;
            }
            oss << "\n";
        }
        
        oss << "================================================================================\n";
        oss << "Global Statistics:\n";
        oss << "  Max Curvature: " << std::scientific << max_curvature() << "\n";
        oss << "  Total Lipschitz: " << std::scientific << total_lipschitz() << "\n";
        oss << "  Total Error (ε_in=1e-6): " << std::scientific 
            << compute_total_error(1e-6, HardwareModel::FLOAT32) << "\n";
        
        return oss.str();
    }
};

// Precision Analyzer - main interface for analyzing models
class PrecisionAnalyzer {
private:
    double target_accuracy;
    HardwareModel default_hardware;
    ComputationGraph graph;
    
public:
    PrecisionAnalyzer(
        double target_eps = 1e-6,
        HardwareModel H = HardwareModel::FLOAT32
    ) : target_accuracy(target_eps), default_hardware(H) {}
    
    // Add an operation to the computation graph
    void trace_operation(
        const std::string& name,
        const std::string& op_type,
        const NumericalType& output
    ) {
        auto node = std::make_shared<OperationNode>(name, op_type, output);
        graph.add_node(node);
    }
    
    // Analyze the traced computation graph
    ComputationGraph& get_graph() {
        return graph;
    }
    
    const ComputationGraph& get_graph() const {
        return graph;
    }
    
    // Generate report
    std::string generate_report() const {
        return graph.to_string();
    }
    
    // Check if model can run on given hardware
    bool can_run_on(HardwareModel H) const {
        auto critical = graph.find_critical_nodes(H);
        return critical.empty();
    }
    
    // Get minimum hardware needed
    HardwareModel minimum_hardware() const {
        int max_bits = 0;
        for (const auto& node : graph.nodes) {
            if (node->precision_bits_required > max_bits) {
                max_bits = node->precision_bits_required;
            }
        }
        
        if (max_bits <= 7) return HardwareModel::BFLOAT16;
        if (max_bits <= 10) return HardwareModel::FLOAT16;
        if (max_bits <= 23) return HardwareModel::FLOAT32;
        if (max_bits <= 52) return HardwareModel::FLOAT64;
        return HardwareModel::FLOAT128;
    }
};

} // namespace hnf
