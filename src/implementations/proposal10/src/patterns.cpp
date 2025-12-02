#include "stability_linter.hpp"
#include <algorithm>

namespace hnf {
namespace stability_linter {
namespace condition_helpers {

bool has_epsilon_protection(const ComputationGraph& graph, 
                           const std::vector<std::string>& nodes) {
    if (nodes.empty()) return false;
    
    // Check if there's an ADD operation before the division that adds a small constant
    std::string div_node_id = nodes.back();
    auto inputs = graph.get_inputs(div_node_id);
    
    for (const auto& input_id : inputs) {
        auto input_node = graph.get_node(input_id);
        if (!input_node) continue;
        
        if (input_node->op == OpType::ADD) {
            // Check if one of the inputs to ADD is a small constant
            // This is a simplified check - in practice, would need to trace constants
            return true;
        }
    }
    
    return false;
}

bool has_clamp_protection(const ComputationGraph& graph,
                        const std::vector<std::string>& nodes) {
    if (nodes.empty()) return false;
    
    // Check if input to the operation comes from a CLAMP
    std::string op_node_id = nodes[0];
    auto inputs = graph.get_inputs(op_node_id);
    
    for (const auto& input_id : inputs) {
        auto input_node = graph.get_node(input_id);
        if (input_node && input_node->op == OpType::CLAMP) {
            return true;
        }
    }
    
    return false;
}

bool has_max_subtraction(const ComputationGraph& graph,
                       const std::vector<std::string>& nodes) {
    if (nodes.empty()) return false;
    
    // For softmax stability, check if there's a SUB and MAX before the EXP
    std::string exp_node_id = nodes[0];
    auto inputs = graph.get_inputs(exp_node_id);
    
    for (const auto& input_id : inputs) {
        auto input_node = graph.get_node(input_id);
        if (!input_node) continue;
        
        if (input_node->op == OpType::SUB) {
            // Check if one of the inputs to SUB is MAX
            auto sub_inputs = graph.get_inputs(input_id);
            for (const auto& sub_input_id : sub_inputs) {
                auto sub_input_node = graph.get_node(sub_input_id);
                if (sub_input_node && sub_input_node->op == OpType::MAX) {
                    return true;
                }
            }
        }
    }
    
    return false;
}

double estimate_max_input(const ComputationGraph& graph,
                        const std::string& node_id) {
    auto node = graph.get_node(node_id);
    if (!node) return 0.0;
    
    auto inputs = graph.get_inputs(node_id);
    double max_val = -std::numeric_limits<double>::infinity();
    
    for (const auto& input_id : inputs) {
        auto input_node = graph.get_node(input_id);
        if (input_node) {
            max_val = std::max(max_val, input_node->value_range.second);
        }
    }
    
    return max_val;
}

bool is_adding_one(const ComputationGraph& graph,
                  const std::vector<std::string>& nodes) {
    if (nodes.size() < 2) return false;
    
    auto add_node = graph.get_node(nodes[0]);
    if (!add_node || add_node->op != OpType::ADD) return false;
    
    // Simplified check - would need constant propagation to verify the constant is 1
    return true;
}

bool is_subtracting_one(const ComputationGraph& graph,
                       const std::vector<std::string>& nodes) {
    if (nodes.size() < 2) return false;
    
    auto sub_node = graph.get_node(nodes[0]);
    if (!sub_node || sub_node->op != OpType::SUB) return false;
    
    // Simplified check
    return true;
}

} // namespace condition_helpers

namespace patterns {

std::vector<LintPattern> get_builtin_patterns() {
    return {
        naive_softmax(),
        naive_logsoftmax(),
        unprotected_division(),
        unprotected_log(),
        unprotected_sqrt(),
        double_exp(),
        exp_overflow(),
        catastrophic_cancellation(),
        layernorm_without_eps(),
        attention_without_scaling(),
        temperature_sharpening(),
        naive_log1p(),
        naive_expm1(),
        variance_cancellation()
    };
}

LintPattern naive_softmax() {
    return LintPattern(
        "naive-softmax",
        "Softmax implemented without numerical stabilization (max subtraction)",
        Severity::WARNING,
        {OpType::EXP, OpType::DIV},
        "Use torch.nn.functional.softmax() or subtract max before exp: exp(x - x.max())"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        return !condition_helpers::has_max_subtraction(graph, nodes);
    });
}

LintPattern naive_logsoftmax() {
    return LintPattern(
        "naive-logsoftmax",
        "log(softmax(x)) chain is numerically unstable",
        Severity::ERROR,
        {OpType::SOFTMAX, OpType::LOG},
        "Use torch.nn.functional.log_softmax() which fuses the operations"
    );
}

LintPattern unprotected_division() {
    return LintPattern(
        "unprotected-division",
        "Division without epsilon protection in denominator",
        Severity::WARNING,
        {OpType::DIV},
        "Add small epsilon to denominator: x / (y + 1e-8)"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        return !condition_helpers::has_epsilon_protection(graph, nodes);
    });
}

LintPattern unprotected_log() {
    return LintPattern(
        "unprotected-log",
        "Logarithm of potentially non-positive value",
        Severity::WARNING,
        {OpType::LOG},
        "Clamp input: torch.log(x.clamp(min=1e-8)) or use torch.log1p for x near 0"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        if (nodes.empty()) return false;
        auto node = graph.get_node(nodes[0]);
        if (!node) return false;
        // Check if the input range could include non-positive values
        auto inputs = graph.get_inputs(nodes[0]);
        for (const auto& input_id : inputs) {
            auto input_node = graph.get_node(input_id);
            if (input_node && input_node->value_range.first <= 0) {
                return !condition_helpers::has_clamp_protection(graph, nodes);
            }
        }
        return false;
    });
}

LintPattern unprotected_sqrt() {
    return LintPattern(
        "unprotected-sqrt",
        "Square root of potentially negative value",
        Severity::WARNING,
        {OpType::SQRT},
        "Clamp input: torch.sqrt(x.clamp(min=0))"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        if (nodes.empty()) return false;
        auto inputs = graph.get_inputs(nodes[0]);
        for (const auto& input_id : inputs) {
            auto input_node = graph.get_node(input_id);
            if (input_node && input_node->value_range.first < 0) {
                return !condition_helpers::has_clamp_protection(graph, nodes);
            }
        }
        return false;
    });
}

LintPattern double_exp() {
    return LintPattern(
        "double-exponential",
        "exp(exp(x)) has extremely high curvature and overflows for x > ~4",
        Severity::ERROR,
        {OpType::EXP, OpType::EXP},
        "Reconsider computation structure; this is fundamentally unstable"
    );
}

LintPattern exp_overflow() {
    return LintPattern(
        "exp-overflow",
        "Exponential of potentially large value (overflow risk)",
        Severity::WARNING,
        {OpType::EXP},
        "Clamp input: torch.exp(x.clamp(max=80)) for FP32, max=700 for FP64"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        if (nodes.empty()) return false;
        double max_input = condition_helpers::estimate_max_input(graph, nodes[0]);
        return max_input > 80.0;  // FP32 overflow threshold
    });
}

LintPattern catastrophic_cancellation() {
    return LintPattern(
        "catastrophic-cancellation",
        "Subtraction of values with similar magnitude loses precision",
        Severity::INFO,
        {OpType::SUB},
        "Consider reformulating to avoid subtraction or use Kahan summation"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        if (nodes.empty()) return false;
        auto inputs = graph.get_inputs(nodes[0]);
        if (inputs.size() < 2) return false;
        
        auto input1 = graph.get_node(inputs[0]);
        auto input2 = graph.get_node(inputs[1]);
        
        if (!input1 || !input2) return false;
        
        // Check if ranges overlap significantly
        double range1_mag = std::max(std::abs(input1->value_range.first), 
                                     std::abs(input1->value_range.second));
        double range2_mag = std::max(std::abs(input2->value_range.first), 
                                     std::abs(input2->value_range.second));
        
        return std::abs(range1_mag - range2_mag) < 0.1 * std::max(range1_mag, range2_mag);
    });
}

LintPattern layernorm_without_eps() {
    return LintPattern(
        "layernorm-without-eps",
        "LayerNorm-like pattern without epsilon (division by std without protection)",
        Severity::WARNING,
        {OpType::STD, OpType::DIV},
        "Use torch.nn.LayerNorm or add epsilon: x / (std + 1e-5)"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        return !condition_helpers::has_epsilon_protection(graph, nodes);
    });
}

LintPattern attention_without_scaling() {
    return LintPattern(
        "attention-without-scaling",
        "Attention scores (Q @ K^T) without scaling by sqrt(d_k)",
        Severity::WARNING,
        {OpType::MATMUL, OpType::SOFTMAX},
        "Scale attention scores: scores / sqrt(d_k) before softmax"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        if (nodes.size() < 2) return false;
        
        // Check if there's a division between matmul and softmax
        auto matmul_outputs = graph.get_outputs(nodes[0]);
        for (const auto& output_id : matmul_outputs) {
            auto output_node = graph.get_node(output_id);
            if (output_node && output_node->op == OpType::DIV) {
                return false;  // Has scaling
            }
        }
        return true;  // No scaling found
    });
}

LintPattern temperature_sharpening() {
    return LintPattern(
        "temperature-sharpening",
        "Temperature < 1 increases curvature and requires higher precision",
        Severity::INFO,
        {OpType::DIV, OpType::SOFTMAX},
        "Document precision requirements when using temperature < 1; curvature scales as 1/T²"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        // This would require constant analysis to detect temperature < 1
        // Simplified: always flag DIV before SOFTMAX as potential temperature
        return true;
    });
}

LintPattern naive_log1p() {
    return LintPattern(
        "naive-log1p",
        "log(1 + x) loses precision for small x",
        Severity::INFO,
        {OpType::ADD, OpType::LOG},
        "Use torch.log1p(x) for numerically stable log(1 + x)"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        return condition_helpers::is_adding_one(graph, nodes);
    });
}

LintPattern naive_expm1() {
    return LintPattern(
        "naive-expm1",
        "exp(x) - 1 loses precision for small x",
        Severity::INFO,
        {OpType::EXP, OpType::SUB},
        "Use torch.expm1(x) for numerically stable exp(x) - 1"
    ).with_condition([](const ComputationGraph& graph, 
                        const std::vector<std::string>& nodes) {
        return condition_helpers::is_subtracting_one(graph, nodes);
    });
}

LintPattern variance_cancellation() {
    return LintPattern(
        "variance-cancellation",
        "Variance computed as E[X²] - E[X]² loses precision",
        Severity::WARNING,
        {OpType::MEAN, OpType::POW, OpType::SUB},
        "Use torch.var() or Welford's algorithm for numerical stability"
    );
}

} // namespace patterns

// PrecisionAnalyzer implementation
int PrecisionAnalyzer::compute_min_bits(double curvature, double diameter, double target_eps) {
    // From HNF Theorem: p >= log2(c * κ * D² / ε)
    // Using c ≈ 1/8 from the proof
    const double c = 0.125;
    
    if (curvature <= 0 || diameter <= 0 || target_eps <= 0) {
        return 0;
    }
    
    double required_precision = (c * curvature * diameter * diameter) / target_eps;
    return static_cast<int>(std::ceil(std::log2(required_precision)));
}

std::vector<PrecisionAnalyzer::PrecisionRequirement> 
PrecisionAnalyzer::analyze_precision_requirements(
    const ComputationGraph& graph,
    double target_accuracy,
    const std::pair<double, double>& domain_range) const {
    
    std::vector<PrecisionRequirement> requirements;
    double diameter = domain_range.second - domain_range.first;
    
    for (const auto& [node_id, node] : graph.nodes) {
        if (node->curvature > 1.0) {  // Only analyze high-curvature nodes
            double node_diameter = node->value_range.second - node->value_range.first;
            int min_bits = compute_min_bits(node->curvature, node_diameter, target_accuracy);
            
            std::stringstream reasoning;
            reasoning << "HNF curvature bound: κ=" << std::scientific << node->curvature
                     << ", D=" << node_diameter 
                     << ", target ε=" << target_accuracy
                     << " => p >= " << min_bits << " bits";
            
            requirements.push_back({
                node_id,
                min_bits,
                node->curvature,
                node_diameter,
                target_accuracy,
                reasoning.str()
            });
        }
    }
    
    return requirements;
}

double PrecisionAnalyzer::compute_curvature_bound(
    const ComputationGraph& graph, const std::string& node_id) const {
    
    auto node = graph.get_node(node_id);
    if (!node) return 0.0;
    
    // Use the curvature already computed during range propagation
    return node->curvature;
}

} // namespace stability_linter
} // namespace hnf
