#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <functional>
#include <set>
#include <optional>
#include <cmath>

namespace hnf {
namespace stability_linter {

// Forward declarations
struct Node;
struct ComputationGraph;
struct LintPattern;
struct LintResult;
struct LintReport;

// Severity levels for lint results
enum class Severity {
    ERROR,
    WARNING,
    INFO
};

std::string severity_to_string(Severity s);

// Operation types that can be detected in computation graphs
enum class OpType {
    PLACEHOLDER,
    EXP,
    LOG,
    LOG1P,
    EXPM1,
    SQRT,
    POW,
    DIV,
    ADD,
    SUB,
    MUL,
    MATMUL,
    SOFTMAX,
    LOG_SOFTMAX,
    RELU,
    SIGMOID,
    TANH,
    LAYERNORM,
    ATTENTION,
    SUM,
    MEAN,
    VAR,
    STD,
    CLAMP,
    MAX,
    MIN,
    TRANSPOSE,
    RESHAPE,
    UNKNOWN
};

std::string optype_to_string(OpType op);
OpType string_to_optype(const std::string& s);

// Represents a single node in the computation graph
struct Node {
    std::string id;
    OpType op;
    std::string target_name;
    std::vector<std::string> input_ids;
    std::map<std::string, std::string> kwargs;
    
    // Numerical properties
    std::pair<double, double> value_range;  // (min, max)
    double lipschitz_constant;
    double curvature;
    
    Node(const std::string& _id, OpType _op, const std::string& target = "")
        : id(_id), op(_op), target_name(target), 
          value_range({-1e10, 1e10}), lipschitz_constant(1.0), curvature(0.0) {}
};

// Represents the full computation graph extracted from a model
struct ComputationGraph {
    std::map<std::string, std::shared_ptr<Node>> nodes;
    std::vector<std::pair<std::string, std::string>> edges;  // (from, to)
    
    void add_node(std::shared_ptr<Node> node);
    void add_edge(const std::string& from, const std::string& to);
    
    std::shared_ptr<Node> get_node(const std::string& id) const;
    std::vector<std::string> get_outputs(const std::string& node_id) const;
    std::vector<std::string> get_inputs(const std::string& node_id) const;
    std::vector<std::string> topological_sort() const;
    
    // Range propagation for curvature analysis
    void propagate_ranges(const std::pair<double, double>& input_range);
    
    // Build from traced torch model
    static ComputationGraph from_traced_model(torch::jit::script::Module& model);
};

// Pattern matching for numerical anti-patterns
struct LintPattern {
    std::string name;
    std::string description;
    Severity severity;
    std::vector<OpType> ops;  // Sequence of operations to match
    std::function<bool(const ComputationGraph&, const std::vector<std::string>&)> condition;
    std::string suggestion;
    
    LintPattern(const std::string& n, const std::string& desc, Severity sev,
                const std::vector<OpType>& op_seq, const std::string& sug)
        : name(n), description(desc), severity(sev), ops(op_seq), suggestion(sug),
          condition(nullptr) {}
    
    LintPattern& with_condition(std::function<bool(const ComputationGraph&, 
                                                    const std::vector<std::string>&)> cond) {
        condition = cond;
        return *this;
    }
    
    std::optional<std::vector<std::string>> matches(const ComputationGraph& graph, 
                                                      const std::string& start_node) const;
};

// Result from a single lint check
struct LintResult {
    Severity severity;
    std::vector<std::string> nodes;
    std::string pattern_name;
    std::string message;
    std::string suggestion;
    double curvature_estimate;  // For curvature-based warnings
    
    LintResult(Severity sev, const std::vector<std::string>& n, 
               const std::string& pat, const std::string& msg, const std::string& sug,
               double curv = 0.0)
        : severity(sev), nodes(n), pattern_name(pat), message(msg), 
          suggestion(sug), curvature_estimate(curv) {}
    
    std::string to_string() const;
};

// Full report from linting a model
struct LintReport {
    std::vector<LintResult> results;
    std::shared_ptr<ComputationGraph> graph;
    
    int n_errors() const;
    int n_warnings() const;
    int n_infos() const;
    
    std::string to_string() const;
    std::string to_json() const;
    
    void add_result(const LintResult& result);
};

// Curvature-based linting
class CurvatureLinter {
private:
    double threshold_;
    
    double estimate_curvature(const Node& node, 
                              const std::pair<double, double>& range) const;
    
    std::pair<double, double> apply_op_range(const Node& node,
                                             const std::vector<std::pair<double, double>>& input_ranges) const;
    
public:
    explicit CurvatureLinter(double threshold = 1e6) : threshold_(threshold) {}
    
    std::vector<LintResult> analyze(const ComputationGraph& graph,
                                    const std::pair<double, double>& input_range) const;
    
    std::string suggest_fix(const Node& node, double curvature) const;
};

// Main linting engine
class NumericalLinter {
private:
    std::vector<LintPattern> pattern_library_;
    std::unique_ptr<CurvatureLinter> curvature_linter_;
    
    void initialize_pattern_library();
    
public:
    NumericalLinter();
    explicit NumericalLinter(double curvature_threshold);
    
    LintReport lint(torch::jit::script::Module& model,
                   const std::pair<double, double>& input_range = {-10.0, 10.0});
    
    void add_pattern(const LintPattern& pattern);
    void set_curvature_threshold(double threshold);
};

// Built-in pattern library
namespace patterns {
    // Returns all built-in patterns
    std::vector<LintPattern> get_builtin_patterns();
    
    // Individual pattern getters
    LintPattern naive_softmax();
    LintPattern naive_logsoftmax();
    LintPattern unprotected_division();
    LintPattern unprotected_log();
    LintPattern unprotected_sqrt();
    LintPattern double_exp();
    LintPattern exp_overflow();
    LintPattern catastrophic_cancellation();
    LintPattern layernorm_without_eps();
    LintPattern attention_without_scaling();
    LintPattern temperature_sharpening();
    LintPattern naive_log1p();
    LintPattern naive_expm1();
    LintPattern variance_cancellation();
}

// Utility functions for pattern conditions
namespace condition_helpers {
    bool has_epsilon_protection(const ComputationGraph& graph, 
                               const std::vector<std::string>& nodes);
    
    bool has_clamp_protection(const ComputationGraph& graph,
                            const std::vector<std::string>& nodes);
    
    bool has_max_subtraction(const ComputationGraph& graph,
                           const std::vector<std::string>& nodes);
    
    double estimate_max_input(const ComputationGraph& graph,
                            const std::string& node_id);
    
    bool is_adding_one(const ComputationGraph& graph,
                      const std::vector<std::string>& nodes);
    
    bool is_subtracting_one(const ComputationGraph& graph,
                           const std::vector<std::string>& nodes);
}

// Precision requirement analysis based on HNF curvature bounds
class PrecisionAnalyzer {
private:
    double compute_curvature_bound(const ComputationGraph& graph,
                                   const std::string& node_id) const;
    
public:
    struct PrecisionRequirement {
        std::string node_id;
        int min_mantissa_bits;  // Minimum required precision
        double curvature;
        double diameter;
        double target_accuracy;
        std::string reasoning;
    };
    
    std::vector<PrecisionRequirement> analyze_precision_requirements(
        const ComputationGraph& graph,
        double target_accuracy,
        const std::pair<double, double>& domain_range) const;
    
    // Compute minimum bits from HNF obstruction theorem
    // p >= log2(c * κ * D^2 / ε)
    static int compute_min_bits(double curvature, double diameter, double target_eps);
};

} // namespace stability_linter
} // namespace hnf
