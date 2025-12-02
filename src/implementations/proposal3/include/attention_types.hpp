#pragma once

#include <torch/torch.h>
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cmath>

namespace hnf {
namespace attention {

// Hardware model for precision analysis
struct HardwareModel {
    enum class Type {
        FP16,
        FP32,
        FP64,
        BF16
    };
    
    int mantissa_bits;  // Precision bits
    int exponent_min;   // Minimum exponent
    int exponent_max;   // Maximum exponent
    std::string name;   // e.g., "fp16", "fp32", "fp64"
    
    HardwareModel(Type type) {
        switch (type) {
            case Type::FP16:
                mantissa_bits = 10;
                exponent_min = -14;
                exponent_max = 15;
                name = "fp16";
                break;
            case Type::FP32:
                mantissa_bits = 23;
                exponent_min = -126;
                exponent_max = 127;
                name = "fp32";
                break;
            case Type::FP64:
                mantissa_bits = 52;
                exponent_min = -1022;
                exponent_max = 1023;
                name = "fp64";
                break;
            case Type::BF16:
                mantissa_bits = 7;
                exponent_min = -126;
                exponent_max = 127;
                name = "bf16";
                break;
        }
    }
    
    int precision_bits() const {
        return mantissa_bits;
    }
    
    double machine_epsilon() const {
        return std::pow(2.0, -mantissa_bits);
    }
    
    double max_value() const {
        return std::pow(2.0, exponent_max);
    }
    
    double min_normal() const {
        return std::pow(2.0, exponent_min);
    }
    
    static HardwareModel fp16() {
        return HardwareModel(Type::FP16);
    }
    
    static HardwareModel fp32() {
        return HardwareModel(Type::FP32);
    }
    
    static HardwareModel fp64() {
        return HardwareModel(Type::FP64);
    }
    
    static HardwareModel bf16() {
        return HardwareModel(Type::BF16);
    }
};

// Stability issue types
enum class IssueType {
    ENTROPY_COLLAPSE,      // Attention too focused
    OVERFLOW_RISK,         // Softmax input too large
    UNDERFLOW_RISK,        // Softmax input too small
    ATTENTION_SPIKE,       // Near one-hot attention
    GRADIENT_VANISHING,    // Derivatives near zero
    HIGH_CURVATURE,        // Nonlinearity amplification
    PRECISION_INSUFFICIENT // Hardware precision inadequate
};

// Severity levels
enum class Severity {
    INFO,
    WARNING,
    ERROR,
    CRITICAL
};

// Stability issue report
struct StabilityIssue {
    std::string layer_name;
    int head_index;
    IssueType type;
    Severity severity;
    double value;
    std::string message;
    std::string suggestion;
    
    StabilityIssue(const std::string& layer, int head, IssueType t, 
                   Severity sev, double val, const std::string& msg, 
                   const std::string& sugg)
        : layer_name(layer), head_index(head), type(t), severity(sev),
          value(val), message(msg), suggestion(sugg) {}
};

// Statistics for a single attention pattern
struct AttentionStats {
    // Basic statistics
    torch::Tensor entropy_per_head;        // [num_heads]
    torch::Tensor max_attention_per_head;  // [num_heads]
    torch::Tensor min_attention_per_head;  // [num_heads]
    
    // Logit statistics
    torch::Tensor logit_max;               // [num_heads]
    torch::Tensor logit_min;               // [num_heads]
    torch::Tensor logit_range;             // [num_heads]
    torch::Tensor logit_std;               // [num_heads]
    
    // Curvature estimates (HNF-specific)
    torch::Tensor curvature_estimate;      // [num_heads]
    torch::Tensor lipschitz_constant;      // [num_heads]
    
    // Precision requirements
    torch::Tensor precision_bits_required; // [num_heads]
    
    // Gradient statistics
    torch::Tensor gradient_norm;           // [num_heads]
    
    int batch_size;
    int num_heads;
    int seq_length;
    int hidden_dim;
};

// Configuration for attention analysis
struct AttentionConfig {
    int num_heads;
    int head_dim;
    int hidden_size;
    int max_seq_length;
    double temperature = 1.0;
    bool use_alibi = false;
    bool use_rope = false;
    
    // Thresholds
    double entropy_collapse_threshold = 0.5;     // nats
    double spike_threshold = 0.95;               // max attention weight
    double curvature_threshold = 1e6;            // HNF curvature bound
    double logit_overflow_threshold = 88.0;      // softmax overflow at ~exp(88)
    double gradient_vanishing_threshold = 1e-6;
    
    // HNF parameters
    double target_accuracy = 1e-6;
    HardwareModel hardware = HardwareModel::fp32();
};

// Diagnosis result
struct AttentionDiagnosis {
    std::vector<StabilityIssue> issues;
    std::map<std::string, AttentionStats> layer_stats;
    AttentionConfig config;
    
    bool has_critical_issues() const {
        for (const auto& issue : issues) {
            if (issue.severity == Severity::CRITICAL || 
                issue.severity == Severity::ERROR) {
                return true;
            }
        }
        return false;
    }
    
    std::vector<StabilityIssue> get_issues_by_severity(Severity sev) const {
        std::vector<StabilityIssue> result;
        for (const auto& issue : issues) {
            if (issue.severity == sev) {
                result.push_back(issue);
            }
        }
        return result;
    }
};

// Intervention suggestion
struct InterventionSuggestion {
    std::string action;  // "reduce_lr", "increase_dropout", "change_precision", etc.
    std::map<std::string, double> parameters;
    std::string reason;
    double expected_improvement;
};

} // namespace attention
} // namespace hnf
