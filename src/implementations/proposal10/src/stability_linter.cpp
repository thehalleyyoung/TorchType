#include "stability_linter.hpp"
#include <algorithm>
#include <sstream>
#include <queue>
#include <cmath>
#include <limits>
#include <iomanip>

namespace hnf {
namespace stability_linter {

// Utility functions
std::string severity_to_string(Severity s) {
    switch (s) {
        case Severity::ERROR: return "ERROR";
        case Severity::WARNING: return "WARNING";
        case Severity::INFO: return "INFO";
        default: return "UNKNOWN";
    }
}

std::string optype_to_string(OpType op) {
    switch (op) {
        case OpType::PLACEHOLDER: return "placeholder";
        case OpType::EXP: return "exp";
        case OpType::LOG: return "log";
        case OpType::LOG1P: return "log1p";
        case OpType::EXPM1: return "expm1";
        case OpType::SQRT: return "sqrt";
        case OpType::POW: return "pow";
        case OpType::DIV: return "div";
        case OpType::ADD: return "add";
        case OpType::SUB: return "sub";
        case OpType::MUL: return "mul";
        case OpType::MATMUL: return "matmul";
        case OpType::SOFTMAX: return "softmax";
        case OpType::LOG_SOFTMAX: return "log_softmax";
        case OpType::RELU: return "relu";
        case OpType::SIGMOID: return "sigmoid";
        case OpType::TANH: return "tanh";
        case OpType::LAYERNORM: return "layernorm";
        case OpType::ATTENTION: return "attention";
        case OpType::SUM: return "sum";
        case OpType::MEAN: return "mean";
        case OpType::VAR: return "var";
        case OpType::STD: return "std";
        case OpType::CLAMP: return "clamp";
        case OpType::MAX: return "max";
        case OpType::MIN: return "min";
        case OpType::TRANSPOSE: return "transpose";
        case OpType::RESHAPE: return "reshape";
        default: return "unknown";
    }
}

OpType string_to_optype(const std::string& s) {
    if (s == "exp" || s == "aten::exp") return OpType::EXP;
    if (s == "log" || s == "aten::log") return OpType::LOG;
    if (s == "log1p" || s == "aten::log1p") return OpType::LOG1P;
    if (s == "expm1" || s == "aten::expm1") return OpType::EXPM1;
    if (s == "sqrt" || s == "aten::sqrt") return OpType::SQRT;
    if (s == "pow" || s == "aten::pow") return OpType::POW;
    if (s == "div" || s == "aten::div" || s == "aten::true_divide") return OpType::DIV;
    if (s == "add" || s == "aten::add") return OpType::ADD;
    if (s == "sub" || s == "aten::sub") return OpType::SUB;
    if (s == "mul" || s == "aten::mul" || s == "aten::multiply") return OpType::MUL;
    if (s == "matmul" || s == "aten::matmul" || s == "aten::mm") return OpType::MATMUL;
    if (s == "softmax" || s == "aten::softmax") return OpType::SOFTMAX;
    if (s == "log_softmax" || s == "aten::log_softmax") return OpType::LOG_SOFTMAX;
    if (s == "relu" || s == "aten::relu") return OpType::RELU;
    if (s == "sigmoid" || s == "aten::sigmoid") return OpType::SIGMOID;
    if (s == "tanh" || s == "aten::tanh") return OpType::TANH;
    if (s == "layer_norm" || s == "aten::layer_norm") return OpType::LAYERNORM;
    if (s == "sum" || s == "aten::sum") return OpType::SUM;
    if (s == "mean" || s == "aten::mean") return OpType::MEAN;
    if (s == "var" || s == "aten::var") return OpType::VAR;
    if (s == "std" || s == "aten::std") return OpType::STD;
    if (s == "clamp" || s == "aten::clamp") return OpType::CLAMP;
    if (s == "max" || s == "aten::max") return OpType::MAX;
    if (s == "min" || s == "aten::min") return OpType::MIN;
    if (s == "transpose" || s == "aten::transpose") return OpType::TRANSPOSE;
    if (s == "reshape" || s == "aten::reshape" || s == "aten::view") return OpType::RESHAPE;
    return OpType::UNKNOWN;
}

// ComputationGraph implementation
void ComputationGraph::add_node(std::shared_ptr<Node> node) {
    nodes[node->id] = node;
}

void ComputationGraph::add_edge(const std::string& from, const std::string& to) {
    edges.push_back({from, to});
}

std::shared_ptr<Node> ComputationGraph::get_node(const std::string& id) const {
    auto it = nodes.find(id);
    return (it != nodes.end()) ? it->second : nullptr;
}

std::vector<std::string> ComputationGraph::get_outputs(const std::string& node_id) const {
    std::vector<std::string> outputs;
    for (const auto& edge : edges) {
        if (edge.first == node_id) {
            outputs.push_back(edge.second);
        }
    }
    return outputs;
}

std::vector<std::string> ComputationGraph::get_inputs(const std::string& node_id) const {
    std::vector<std::string> inputs;
    for (const auto& edge : edges) {
        if (edge.second == node_id) {
            inputs.push_back(edge.first);
        }
    }
    return inputs;
}

std::vector<std::string> ComputationGraph::topological_sort() const {
    std::map<std::string, int> in_degree;
    for (const auto& [id, node] : nodes) {
        in_degree[id] = 0;
    }
    for (const auto& edge : edges) {
        in_degree[edge.second]++;
    }
    
    std::queue<std::string> q;
    for (const auto& [id, deg] : in_degree) {
        if (deg == 0) {
            q.push(id);
        }
    }
    
    std::vector<std::string> result;
    while (!q.empty()) {
        std::string curr = q.front();
        q.pop();
        result.push_back(curr);
        
        for (const auto& next : get_outputs(curr)) {
            in_degree[next]--;
            if (in_degree[next] == 0) {
                q.push(next);
            }
        }
    }
    
    return result;
}

void ComputationGraph::propagate_ranges(const std::pair<double, double>& input_range) {
    auto order = topological_sort();
    
    for (const auto& node_id : order) {
        auto node = get_node(node_id);
        if (!node) continue;
        
        if (node->op == OpType::PLACEHOLDER) {
            node->value_range = input_range;
        } else {
            std::vector<std::pair<double, double>> input_ranges;
            for (const auto& input_id : get_inputs(node_id)) {
                auto input_node = get_node(input_id);
                if (input_node) {
                    input_ranges.push_back(input_node->value_range);
                }
            }
            
            if (!input_ranges.empty()) {
                double lo = input_ranges[0].first;
                double hi = input_ranges[0].second;
                
                switch (node->op) {
                    case OpType::EXP:
                        node->value_range = {std::exp(lo), std::exp(hi)};
                        break;
                    case OpType::LOG:
                        node->value_range = {std::log(std::max(lo, 1e-10)), 
                                            std::log(std::max(hi, 1e-10))};
                        break;
                    case OpType::SQRT:
                        node->value_range = {std::sqrt(std::max(lo, 0.0)), 
                                            std::sqrt(std::max(hi, 0.0))};
                        break;
                    case OpType::RELU:
                        node->value_range = {std::max(0.0, lo), std::max(0.0, hi)};
                        break;
                    case OpType::SIGMOID:
                    case OpType::SOFTMAX:
                        node->value_range = {0.0, 1.0};
                        break;
                    case OpType::TANH:
                        node->value_range = {-1.0, 1.0};
                        break;
                    case OpType::ADD:
                        if (input_ranges.size() >= 2) {
                            node->value_range = {
                                input_ranges[0].first + input_ranges[1].first,
                                input_ranges[0].second + input_ranges[1].second
                            };
                        }
                        break;
                    case OpType::MUL:
                        if (input_ranges.size() >= 2) {
                            double vals[4] = {
                                input_ranges[0].first * input_ranges[1].first,
                                input_ranges[0].first * input_ranges[1].second,
                                input_ranges[0].second * input_ranges[1].first,
                                input_ranges[0].second * input_ranges[1].second
                            };
                            node->value_range = {
                                *std::min_element(vals, vals + 4),
                                *std::max_element(vals, vals + 4)
                            };
                        }
                        break;
                    default:
                        node->value_range = {lo, hi};
                }
            }
        }
        
        // Compute curvature estimates based on HNF theory
        // For operations like log and div, curvature depends on INPUT range
        // For operations like exp and softmax, curvature depends on OUTPUT range
        double input_lo = -1e10, input_hi = 1e10;
        if (!get_inputs(node_id).empty()) {
            auto first_input = get_node(get_inputs(node_id)[0]);
            if (first_input) {
                input_lo = first_input->value_range.first;
                input_hi = first_input->value_range.second;
            }
        }
        
        double lo = node->value_range.first;
        double hi = node->value_range.second;
        
        switch (node->op) {
            case OpType::EXP:
                // κ_exp = e^(2x) at maximum - depends on INPUT
                // Clamp to avoid overflow in curvature calculation
                if (input_hi > 50.0) {
                    node->curvature = std::numeric_limits<double>::infinity();
                    node->lipschitz_constant = std::numeric_limits<double>::infinity();
                } else {
                    node->curvature = std::exp(2.0 * input_hi);
                    node->lipschitz_constant = std::exp(input_hi);
                }
                break;
            case OpType::LOG:
                // κ_log = 1/x^2 at minimum - depends on INPUT
                node->curvature = 1.0 / std::pow(std::max(std::abs(input_lo), 1e-10), 2);
                node->lipschitz_constant = 1.0 / std::max(std::abs(input_lo), 1e-10);
                break;
            case OpType::DIV:
                // High curvature for division - depends on DENOMINATOR (second input)
                // Simplified: use INPUT range
                node->curvature = 1.0 / std::pow(std::max(std::abs(input_lo), 1e-10), 3);
                node->lipschitz_constant = 1.0 / std::max(std::abs(input_lo), 1e-10);
                break;
            case OpType::SOFTMAX:
                // κ_softmax = e^(2·range(x)) - depends on INPUT range
                node->curvature = std::exp(2.0 * (input_hi - input_lo));
                node->lipschitz_constant = 1.0;
                break;
            case OpType::SQRT:
                // κ_sqrt = 1/(4x^1.5) - depends on INPUT
                node->curvature = 1.0 / (4.0 * std::pow(std::max(input_lo, 1e-10), 1.5));
                node->lipschitz_constant = 0.5 / std::sqrt(std::max(input_lo, 1e-10));
                break;
            case OpType::SIGMOID:
                node->curvature = 0.25;
                node->lipschitz_constant = 0.25;
                break;
            case OpType::TANH:
                node->curvature = 1.0;
                node->lipschitz_constant = 1.0;
                break;
            case OpType::RELU:
            case OpType::ADD:
            case OpType::MUL:
                node->curvature = 0.0;
                node->lipschitz_constant = 1.0;
                break;
            default:
                node->curvature = 1.0;
                node->lipschitz_constant = 1.0;
        }
    }
}

ComputationGraph ComputationGraph::from_traced_model(torch::jit::script::Module& model) {
    ComputationGraph graph;
    
    // Get the forward method
    auto method = model.get_method("forward");
    auto schema = method.function().getSchema();
    
    // Parse the graph from the traced model
    auto g = method.graph();
    
    int node_counter = 0;
    std::map<torch::jit::Value*, std::string> value_to_id;
    
    for (auto input : g->inputs()) {
        std::string id = "input_" + std::to_string(node_counter++);
        value_to_id[input] = id;
        auto node = std::make_shared<Node>(id, OpType::PLACEHOLDER);
        graph.add_node(node);
    }
    
    for (auto node_ptr : g->nodes()) {
        std::string kind_str = node_ptr->kind().toQualString();
        OpType op = string_to_optype(kind_str);
        
        std::string node_id = "node_" + std::to_string(node_counter++);
        auto node = std::make_shared<Node>(node_id, op, kind_str);
        
        // Track inputs
        for (auto input : node_ptr->inputs()) {
            if (value_to_id.find(input) != value_to_id.end()) {
                node->input_ids.push_back(value_to_id[input]);
                graph.add_edge(value_to_id[input], node_id);
            }
        }
        
        graph.add_node(node);
        
        // Track outputs
        for (auto output : node_ptr->outputs()) {
            value_to_id[output] = node_id;
        }
    }
    
    return graph;
}

// LintPattern implementation
std::optional<std::vector<std::string>> LintPattern::matches(
    const ComputationGraph& graph, const std::string& start_node) const {
    
    std::vector<std::string> matched_nodes;
    std::string current = start_node;
    
    for (size_t i = 0; i < ops.size(); ++i) {
        auto node = graph.get_node(current);
        if (!node) return std::nullopt;
        
        if (node->op != ops[i]) return std::nullopt;
        
        matched_nodes.push_back(current);
        
        if (i < ops.size() - 1) {
            auto outputs = graph.get_outputs(current);
            if (outputs.empty()) return std::nullopt;
            current = outputs[0];
        }
    }
    
    if (condition && !condition(graph, matched_nodes)) {
        return std::nullopt;
    }
    
    return matched_nodes;
}

// LintResult implementation
std::string LintResult::to_string() const {
    std::stringstream ss;
    std::string icon;
    switch (severity) {
        case Severity::ERROR: icon = "❌"; break;
        case Severity::WARNING: icon = "⚠️ "; break;
        case Severity::INFO: icon = "ℹ️ "; break;
    }
    
    ss << icon << " [" << severity_to_string(severity) << "] ";
    if (!pattern_name.empty()) {
        ss << "(" << pattern_name << ") ";
    }
    ss << "at nodes: ";
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (i > 0) ss << " -> ";
        ss << nodes[i];
    }
    ss << "\n   " << message;
    if (!suggestion.empty()) {
        ss << "\n   Suggestion: " << suggestion;
    }
    if (curvature_estimate > 0) {
        ss << "\n   Curvature: " << std::scientific << std::setprecision(2) 
           << curvature_estimate;
    }
    return ss.str();
}

// LintReport implementation
int LintReport::n_errors() const {
    return std::count_if(results.begin(), results.end(),
                        [](const LintResult& r) { return r.severity == Severity::ERROR; });
}

int LintReport::n_warnings() const {
    return std::count_if(results.begin(), results.end(),
                        [](const LintResult& r) { return r.severity == Severity::WARNING; });
}

int LintReport::n_infos() const {
    return std::count_if(results.begin(), results.end(),
                        [](const LintResult& r) { return r.severity == Severity::INFO; });
}

std::string LintReport::to_string() const {
    std::stringstream ss;
    ss << "=== Numerical Stability Lint Report ===\n\n";
    
    for (const auto& result : results) {
        ss << result.to_string() << "\n\n";
    }
    
    ss << "Summary: " << n_errors() << " errors, " 
       << n_warnings() << " warnings, " 
       << n_infos() << " info messages\n";
    
    return ss.str();
}

std::string LintReport::to_json() const {
    std::stringstream ss;
    ss << "{\n  \"results\": [\n";
    
    for (size_t i = 0; i < results.size(); ++i) {
        const auto& r = results[i];
        ss << "    {\n";
        ss << "      \"severity\": \"" << severity_to_string(r.severity) << "\",\n";
        ss << "      \"pattern\": \"" << r.pattern_name << "\",\n";
        ss << "      \"message\": \"" << r.message << "\",\n";
        ss << "      \"suggestion\": \"" << r.suggestion << "\",\n";
        ss << "      \"nodes\": [";
        for (size_t j = 0; j < r.nodes.size(); ++j) {
            if (j > 0) ss << ", ";
            ss << "\"" << r.nodes[j] << "\"";
        }
        ss << "],\n";
        ss << "      \"curvature\": " << r.curvature_estimate << "\n";
        ss << "    }";
        if (i < results.size() - 1) ss << ",";
        ss << "\n";
    }
    
    ss << "  ],\n";
    ss << "  \"summary\": {\n";
    ss << "    \"errors\": " << n_errors() << ",\n";
    ss << "    \"warnings\": " << n_warnings() << ",\n";
    ss << "    \"infos\": " << n_infos() << "\n";
    ss << "  }\n";
    ss << "}\n";
    
    return ss.str();
}

void LintReport::add_result(const LintResult& result) {
    results.push_back(result);
}

// CurvatureLinter implementation
double CurvatureLinter::estimate_curvature(
    const Node& node, const std::pair<double, double>& range) const {
    
    return node.curvature;  // Already computed during range propagation
}

std::string CurvatureLinter::suggest_fix(const Node& node, double curvature) const {
    std::stringstream ss;
    
    if (node.op == OpType::EXP) {
        ss << "Consider clamping input to exp: x.clamp(max=80) to avoid overflow";
    } else if (node.op == OpType::LOG) {
        ss << "Add protection: log(x.clamp(min=1e-8)) or use log1p for values near 1";
    } else if (node.op == OpType::DIV) {
        ss << "Add epsilon to denominator: x / (y + 1e-8) to avoid division by zero";
    } else if (node.op == OpType::SOFTMAX) {
        ss << "Use built-in torch.softmax which implements numerical stabilization";
    } else {
        ss << "High curvature detected; consider using higher precision (FP32 or FP64)";
    }
    
    return ss.str();
}

std::vector<LintResult> CurvatureLinter::analyze(
    const ComputationGraph& graph,
    const std::pair<double, double>& input_range) const {
    
    std::vector<LintResult> results;
    
    for (const auto& [node_id, node] : graph.nodes) {
        if (node->curvature > threshold_) {
            std::stringstream msg;
            msg << "High curvature (" << std::scientific << std::setprecision(2) 
                << node->curvature << ") at " << optype_to_string(node->op) 
                << " may cause precision issues";
            
            results.push_back(LintResult(
                Severity::WARNING,
                {node_id},
                "high-curvature",
                msg.str(),
                suggest_fix(*node, node->curvature),
                node->curvature
            ));
        }
    }
    
    return results;
}

// NumericalLinter implementation
NumericalLinter::NumericalLinter() 
    : curvature_linter_(std::make_unique<CurvatureLinter>()) {
    initialize_pattern_library();
}

NumericalLinter::NumericalLinter(double curvature_threshold)
    : curvature_linter_(std::make_unique<CurvatureLinter>(curvature_threshold)) {
    initialize_pattern_library();
}

void NumericalLinter::initialize_pattern_library() {
    pattern_library_ = patterns::get_builtin_patterns();
}

LintReport NumericalLinter::lint(
    torch::jit::script::Module& model,
    const std::pair<double, double>& input_range) {
    
    LintReport report;
    
    // Parse model to graph
    auto graph = ComputationGraph::from_traced_model(model);
    report.graph = std::make_shared<ComputationGraph>(graph);
    
    // Propagate ranges and compute curvatures
    graph.propagate_ranges(input_range);
    
    // Pattern matching
    for (const auto& pattern : pattern_library_) {
        for (const auto& [node_id, node] : graph.nodes) {
            auto match = pattern.matches(graph, node_id);
            if (match) {
                report.add_result(LintResult(
                    pattern.severity,
                    *match,
                    pattern.name,
                    pattern.description,
                    pattern.suggestion
                ));
            }
        }
    }
    
    // Curvature analysis
    auto curv_results = curvature_linter_->analyze(graph, input_range);
    for (const auto& result : curv_results) {
        report.add_result(result);
    }
    
    return report;
}

void NumericalLinter::add_pattern(const LintPattern& pattern) {
    pattern_library_.push_back(pattern);
}

void NumericalLinter::set_curvature_threshold(double threshold) {
    curvature_linter_ = std::make_unique<CurvatureLinter>(threshold);
}

} // namespace stability_linter
} // namespace hnf
