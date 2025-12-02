#include "../include/attention_analyzer.hpp"
#include <sstream>
#include <iomanip>

namespace hnf {
namespace attention {

AttentionAnalyzer::AttentionAnalyzer(const AttentionConfig& config)
    : config_(config) {}

AttentionStats AttentionAnalyzer::analyze_pattern(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const std::string& layer_name
) {
    AttentionStats stats;
    
    stats.batch_size = Q.size(0);
    stats.num_heads = Q.size(1);
    stats.seq_length = Q.size(2);
    stats.hidden_dim = Q.size(3);
    
    // Compute attention weights for analysis
    auto QK = torch::matmul(Q, K.transpose(-2, -1));
    QK = QK / (std::sqrt(static_cast<double>(stats.hidden_dim)) * config_.temperature);
    auto attention_weights = torch::softmax(QK, /*dim=*/-1);
    
    // 1. Entropy per head
    auto log_attn = torch::log(attention_weights + 1e-10);
    stats.entropy_per_head = -(attention_weights * log_attn).sum(-1).mean(-1).mean(0);  // [num_heads]
    
    // 2. Max/min attention per head
    stats.max_attention_per_head = attention_weights.amax({0, -1, -2});  // [num_heads]
    stats.min_attention_per_head = attention_weights.amin({0, -1, -2});  // [num_heads]
    
    // 3. Logit statistics
    stats.logit_max = QK.amax({0, -1, -2});  // [num_heads]
    stats.logit_min = QK.amin({0, -1, -2});  // [num_heads]
    stats.logit_range = stats.logit_max - stats.logit_min;
    stats.logit_std = QK.std({0, -1, -2});  // [num_heads]
    
    // 4. HNF curvature analysis
    auto curvature_per_batch = AttentionCurvature::compute_curvature(Q, K, config_.temperature);
    stats.curvature_estimate = curvature_per_batch.mean(0);  // [num_heads]
    
    // 5. Lipschitz constant
    auto lipschitz_per_batch = AttentionCurvature::compute_lipschitz_constant(Q, K, config_.temperature);
    stats.lipschitz_constant = lipschitz_per_batch.mean(0);  // [num_heads]
    
    // 6. Precision requirements (HNF Theorem)
    double diameter = AttentionCurvature::estimate_domain_diameter(Q, K);
    stats.precision_bits_required = AttentionCurvature::estimate_precision_requirement(
        stats.curvature_estimate,
        diameter,
        config_.target_accuracy
    );
    
    // 7. Gradient analysis
    stats.gradient_norm = AttentionCurvature::analyze_gradient_flow(attention_weights, V).mean(0);
    
    return stats;
}

std::pair<torch::Tensor, AttentionStats> AttentionAnalyzer::compute_attention_with_stats(
    const torch::Tensor& Q,
    const torch::Tensor& K,
    const torch::Tensor& V,
    const std::string& layer_name
) {
    // Compute attention
    auto QK = torch::matmul(Q, K.transpose(-2, -1));
    QK = QK / (std::sqrt(static_cast<double>(Q.size(-1))) * config_.temperature);
    auto attention_weights = torch::softmax(QK, /*dim=*/-1);
    auto output = torch::matmul(attention_weights, V);
    
    // Analyze
    auto stats = analyze_pattern(Q, K, V, layer_name);
    
    return {output, stats};
}

AttentionDiagnosis AttentionAnalyzer::check_pretraining_stability(
    int num_layers,
    const std::vector<std::string>& layer_names
) {
    AttentionDiagnosis diagnosis;
    diagnosis.config = config_;
    
    // Theoretical analysis based on architecture
    // Expected logit max for random initialization:
    // QK^T ~ N(0, head_dim) element-wise
    // max ~ sqrt(2 * log(seq_len)) * sqrt(head_dim)
    
    const double expected_logit_max = std::sqrt(2.0 * std::log(config_.max_seq_length)) 
                                    * std::sqrt(config_.head_dim) 
                                    / std::sqrt(config_.head_dim);  // Normalized
    
    // Curvature bound
    const double expected_curvature = std::exp(2.0 * expected_logit_max);
    
    // Precision requirement
    const double diameter = std::sqrt(config_.max_seq_length * config_.head_dim);
    const double precision_bits = std::log2(expected_curvature * diameter * diameter / config_.target_accuracy);
    
    // Check against hardware
    if (precision_bits > config_.hardware.mantissa_bits) {
        diagnosis.issues.push_back(StabilityIssue(
            "all_layers",
            -1,
            IssueType::PRECISION_INSUFFICIENT,
            Severity::ERROR,
            precision_bits,
            "Architecture requires " + std::to_string(static_cast<int>(precision_bits)) + 
            " bits, but hardware provides " + std::to_string(config_.hardware.mantissa_bits),
            "Use higher precision (fp32 or fp64) for attention softmax, or reduce sequence length"
        ));
    }
    
    // Check for overflow risk
    if (expected_logit_max > config_.logit_overflow_threshold) {
        diagnosis.issues.push_back(StabilityIssue(
            "all_layers",
            -1,
            IssueType::OVERFLOW_RISK,
            Severity::WARNING,
            expected_logit_max,
            "Expected logit maximum " + std::to_string(expected_logit_max) + 
            " may cause softmax overflow",
            "Consider using temperature scaling, ALiBi, or RoPE positional embeddings"
        ));
    }
    
    // Check if sequence length is too large
    if (config_.max_seq_length > 4096 && config_.hardware.mantissa_bits <= 16) {
        diagnosis.issues.push_back(StabilityIssue(
            "all_layers",
            -1,
            IssueType::HIGH_CURVATURE,
            Severity::WARNING,
            config_.max_seq_length,
            "Long sequences with low precision may be unstable",
            "Use fp32 for long-context attention or implement sparse attention"
        ));
    }
    
    return diagnosis;
}

AttentionDiagnosis AttentionAnalyzer::diagnose(
    const std::map<std::string, std::vector<AttentionStats>>& history
) {
    AttentionDiagnosis diagnosis;
    diagnosis.config = config_;
    
    for (const auto& [layer_name, stats_list] : history) {
        if (stats_list.empty()) continue;
        
        // Average over recent history (last 100 samples or all if fewer)
        size_t window = std::min(stats_list.size(), size_t(100));
        size_t start_idx = stats_list.size() - window;
        
        // Average statistics
        auto avg_entropy = torch::zeros({stats_list[0].num_heads});
        auto avg_curvature = torch::zeros({stats_list[0].num_heads});
        auto avg_max_attn = torch::zeros({stats_list[0].num_heads});
        auto avg_precision_req = torch::zeros({stats_list[0].num_heads});
        auto avg_logit_max = torch::zeros({stats_list[0].num_heads});
        
        for (size_t i = start_idx; i < stats_list.size(); ++i) {
            avg_entropy += stats_list[i].entropy_per_head;
            avg_curvature += stats_list[i].curvature_estimate;
            avg_max_attn += stats_list[i].max_attention_per_head;
            avg_precision_req += stats_list[i].precision_bits_required;
            avg_logit_max += stats_list[i].logit_max;
        }
        
        avg_entropy /= static_cast<double>(window);
        avg_curvature /= static_cast<double>(window);
        avg_max_attn /= static_cast<double>(window);
        avg_precision_req /= static_cast<double>(window);
        avg_logit_max /= static_cast<double>(window);
        
        // Check each head
        for (int h = 0; h < stats_list[0].num_heads; ++h) {
            // Entropy collapse
            double entropy = avg_entropy[h].item<double>();
            if (entropy < config_.entropy_collapse_threshold) {
                diagnosis.issues.push_back(StabilityIssue(
                    layer_name,
                    h,
                    IssueType::ENTROPY_COLLAPSE,
                    determine_severity(IssueType::ENTROPY_COLLAPSE, entropy, config_.entropy_collapse_threshold),
                    entropy,
                    generate_message(IssueType::ENTROPY_COLLAPSE, layer_name, h, entropy),
                    generate_suggestion(IssueType::ENTROPY_COLLAPSE, entropy)
                ));
            }
            
            // High curvature
            double curvature = avg_curvature[h].item<double>();
            if (curvature > config_.curvature_threshold) {
                diagnosis.issues.push_back(StabilityIssue(
                    layer_name,
                    h,
                    IssueType::HIGH_CURVATURE,
                    determine_severity(IssueType::HIGH_CURVATURE, curvature, config_.curvature_threshold),
                    curvature,
                    generate_message(IssueType::HIGH_CURVATURE, layer_name, h, curvature),
                    generate_suggestion(IssueType::HIGH_CURVATURE, curvature)
                ));
            }
            
            // Attention spike
            double max_attn = avg_max_attn[h].item<double>();
            if (max_attn > config_.spike_threshold) {
                diagnosis.issues.push_back(StabilityIssue(
                    layer_name,
                    h,
                    IssueType::ATTENTION_SPIKE,
                    determine_severity(IssueType::ATTENTION_SPIKE, max_attn, config_.spike_threshold),
                    max_attn,
                    generate_message(IssueType::ATTENTION_SPIKE, layer_name, h, max_attn),
                    generate_suggestion(IssueType::ATTENTION_SPIKE, max_attn)
                ));
            }
            
            // Precision insufficient
            double prec_req = avg_precision_req[h].item<double>();
            if (prec_req > config_.hardware.mantissa_bits) {
                diagnosis.issues.push_back(StabilityIssue(
                    layer_name,
                    h,
                    IssueType::PRECISION_INSUFFICIENT,
                    Severity::ERROR,
                    prec_req,
                    generate_message(IssueType::PRECISION_INSUFFICIENT, layer_name, h, prec_req),
                    generate_suggestion(IssueType::PRECISION_INSUFFICIENT, prec_req)
                ));
            }
            
            // Overflow risk
            double logit_max = avg_logit_max[h].item<double>();
            if (logit_max > config_.logit_overflow_threshold) {
                diagnosis.issues.push_back(StabilityIssue(
                    layer_name,
                    h,
                    IssueType::OVERFLOW_RISK,
                    Severity::ERROR,
                    logit_max,
                    generate_message(IssueType::OVERFLOW_RISK, layer_name, h, logit_max),
                    generate_suggestion(IssueType::OVERFLOW_RISK, logit_max)
                ));
            }
        }
        
        // Store averaged stats
        AttentionStats avg_stats;
        avg_stats.entropy_per_head = avg_entropy;
        avg_stats.curvature_estimate = avg_curvature;
        avg_stats.max_attention_per_head = avg_max_attn;
        avg_stats.precision_bits_required = avg_precision_req;
        avg_stats.logit_max = avg_logit_max;
        avg_stats.num_heads = stats_list[0].num_heads;
        
        diagnosis.layer_stats[layer_name] = avg_stats;
    }
    
    return diagnosis;
}

std::vector<InterventionSuggestion> AttentionAnalyzer::suggest_interventions(
    const AttentionDiagnosis& diagnosis
) {
    std::vector<InterventionSuggestion> suggestions;
    
    // Analyze issues and suggest interventions
    std::map<std::string, int> issue_counts;
    for (const auto& issue : diagnosis.issues) {
        std::string type_str;
        switch (issue.type) {
            case IssueType::ENTROPY_COLLAPSE: type_str = "entropy_collapse"; break;
            case IssueType::OVERFLOW_RISK: type_str = "overflow"; break;
            case IssueType::HIGH_CURVATURE: type_str = "curvature"; break;
            case IssueType::ATTENTION_SPIKE: type_str = "spike"; break;
            case IssueType::PRECISION_INSUFFICIENT: type_str = "precision"; break;
            default: type_str = "other"; break;
        }
        issue_counts[type_str]++;
    }
    
    // Suggest based on dominant issues
    if (issue_counts["entropy_collapse"] > 0) {
        InterventionSuggestion sugg;
        sugg.action = "add_entropy_regularization";
        sugg.parameters["lambda"] = 0.01;
        sugg.reason = "Multiple heads showing entropy collapse";
        sugg.expected_improvement = 0.3;
        suggestions.push_back(sugg);
    }
    
    if (issue_counts["overflow"] > 0) {
        InterventionSuggestion sugg;
        sugg.action = "clamp_logits";
        sugg.parameters["max_value"] = 20.0;
        sugg.reason = "Attention logits approaching overflow threshold";
        sugg.expected_improvement = 0.8;
        suggestions.push_back(sugg);
        
        InterventionSuggestion sugg2;
        sugg2.action = "increase_temperature";
        sugg2.parameters["temperature"] = config_.temperature * 1.5;
        sugg2.reason = "Temperature scaling can prevent overflow";
        sugg2.expected_improvement = 0.6;
        suggestions.push_back(sugg2);
    }
    
    if (issue_counts["precision"] > 0) {
        InterventionSuggestion sugg;
        sugg.action = "use_higher_precision";
        sugg.parameters["target_bits"] = 32.0;
        sugg.reason = "Current hardware precision insufficient";
        sugg.expected_improvement = 1.0;
        suggestions.push_back(sugg);
    }
    
    if (issue_counts["curvature"] > 0) {
        InterventionSuggestion sugg;
        sugg.action = "reduce_learning_rate";
        sugg.parameters["factor"] = 0.5;
        sugg.reason = "High curvature indicates optimization instability";
        sugg.expected_improvement = 0.4;
        suggestions.push_back(sugg);
    }
    
    return suggestions;
}

AttentionAnalyzer::StabilityPrediction AttentionAnalyzer::predict_stability(
    int seq_length,
    int num_heads,
    int head_dim,
    double temperature,
    const HardwareModel& hardware
) {
    StabilityPrediction pred;
    pred.is_stable = true;
    
    // Estimate expected logit range for random initialization
    double expected_logit_max = std::sqrt(2.0 * std::log(seq_length)) * std::sqrt(head_dim) / std::sqrt(head_dim);
    
    // Expected curvature
    pred.expected_curvature = std::exp(2.0 * expected_logit_max / temperature);
    
    // Required precision
    double diameter = std::sqrt(seq_length * head_dim);
    pred.required_precision_bits = std::log2(pred.expected_curvature * diameter * diameter / config_.target_accuracy);
    
    // Check stability
    if (pred.required_precision_bits > hardware.mantissa_bits) {
        pred.is_stable = false;
        std::stringstream ss;
        ss << "Requires " << std::fixed << std::setprecision(1) 
           << pred.required_precision_bits << " bits, hardware provides " 
           << hardware.mantissa_bits;
        pred.warnings.push_back(ss.str());
        pred.recommendations.push_back("Use " + std::string(hardware.mantissa_bits >= 32 ? "fp64" : "fp32") + " for softmax");
    }
    
    if (expected_logit_max > config_.logit_overflow_threshold / 2.0) {
        pred.warnings.push_back("Logits may approach overflow threshold");
        pred.recommendations.push_back("Consider temperature scaling or logit clamping");
    }
    
    if (seq_length > 2048 && num_heads > 32) {
        pred.warnings.push_back("Large sequence length and many heads may cause memory/stability issues");
        pred.recommendations.push_back("Consider sparse attention or reducing num_heads");
    }
    
    return pred;
}

torch::Tensor AttentionAnalyzer::compute_entropy(const torch::Tensor& attention_weights) {
    auto log_attn = torch::log(attention_weights + 1e-10);
    return -(attention_weights * log_attn).sum(-1);
}

AttentionAnalyzer::OverflowRisk AttentionAnalyzer::detect_overflow_risk(
    const torch::Tensor& logits
) {
    OverflowRisk risk;
    
    risk.max_logits = logits.amax({-1, -2});  // [batch, heads]
    risk.min_logits = logits.amin({-1, -2});  // [batch, heads]
    
    auto max_val = risk.max_logits.max().item<double>();
    auto min_val = risk.min_logits.min().item<double>();
    
    risk.overflow_likely = (max_val > config_.logit_overflow_threshold);
    risk.underflow_likely = (min_val < -config_.logit_overflow_threshold);
    
    if (risk.overflow_likely) {
        risk.recommendation = "Clamp logits to [-" + std::to_string(config_.logit_overflow_threshold) + 
                            ", " + std::to_string(config_.logit_overflow_threshold) + "] or use temperature scaling";
    } else if (risk.underflow_likely) {
        risk.recommendation = "Very negative logits may cause underflow; check input scaling";
    } else {
        risk.recommendation = "No overflow/underflow risk detected";
    }
    
    return risk;
}

// Helper methods
Severity AttentionAnalyzer::determine_severity(IssueType type, double value, double threshold) {
    double ratio = std::abs(value / threshold);
    
    switch (type) {
        case IssueType::ENTROPY_COLLAPSE:
            if (ratio < 0.5) return Severity::CRITICAL;
            if (ratio < 0.75) return Severity::ERROR;
            return Severity::WARNING;
            
        case IssueType::OVERFLOW_RISK:
        case IssueType::PRECISION_INSUFFICIENT:
            if (ratio > 2.0) return Severity::CRITICAL;
            if (ratio > 1.2) return Severity::ERROR;
            return Severity::WARNING;
            
        case IssueType::HIGH_CURVATURE:
            if (ratio > 10.0) return Severity::CRITICAL;
            if (ratio > 2.0) return Severity::ERROR;
            return Severity::WARNING;
            
        default:
            return Severity::INFO;
    }
}

std::string AttentionAnalyzer::generate_message(
    IssueType type, 
    const std::string& layer, 
    int head, 
    double value
) {
    std::stringstream ss;
    ss << std::fixed << std::setprecision(4);
    
    switch (type) {
        case IssueType::ENTROPY_COLLAPSE:
            ss << "Layer " << layer << " head " << head << ": entropy = " << value 
               << " (threshold: " << config_.entropy_collapse_threshold << ")";
            break;
        case IssueType::OVERFLOW_RISK:
            ss << "Layer " << layer << " head " << head << ": max logit = " << value
               << " approaching overflow";
            break;
        case IssueType::HIGH_CURVATURE:
            ss << "Layer " << layer << " head " << head << ": curvature = " << std::scientific << value;
            break;
        case IssueType::ATTENTION_SPIKE:
            ss << "Layer " << layer << " head " << head << ": max attention = " << value
               << " (near one-hot)";
            break;
        case IssueType::PRECISION_INSUFFICIENT:
            ss << "Layer " << layer << " head " << head << ": requires " << value
               << " bits (have " << config_.hardware.mantissa_bits << ")";
            break;
        default:
            ss << "Layer " << layer << " head " << head << ": issue detected";
    }
    
    return ss.str();
}

std::string AttentionAnalyzer::generate_suggestion(IssueType type, double /* value */) {
    switch (type) {
        case IssueType::ENTROPY_COLLAPSE:
            return "Add entropy regularization or increase attention dropout";
        case IssueType::OVERFLOW_RISK:
            return "Clamp attention logits or increase temperature";
        case IssueType::HIGH_CURVATURE:
            return "Use higher precision (fp32/fp64) or reduce learning rate";
        case IssueType::ATTENTION_SPIKE:
            return "Gradients may vanish; consider label smoothing or attention smoothing";
        case IssueType::PRECISION_INSUFFICIENT:
            return "Use fp32 or fp64 for attention computation";
        default:
            return "Monitor this issue";
    }
}

// AttentionMonitor implementation
AttentionMonitor::AttentionMonitor(const AttentionConfig& config, int log_frequency)
    : config_(config), log_frequency_(log_frequency), current_step_(0), analyzer_(config) {}

void AttentionMonitor::record(const std::string& layer_name, const AttentionStats& stats) {
    history_[layer_name].push_back(stats);
    current_step_++;
    
    // Trigger hooks
    for (const auto& hook : hooks_) {
        hook(layer_name, stats);
    }
}

bool AttentionMonitor::should_monitor(int step) {
    return (step % log_frequency_ == 0);
}

AttentionDiagnosis AttentionMonitor::get_diagnosis() {
    return analyzer_.diagnose(history_);
}

void AttentionMonitor::register_hook(HookFunction hook) {
    hooks_.push_back(hook);
}

void AttentionMonitor::clear() {
    history_.clear();
    current_step_ = 0;
}

} // namespace attention
} // namespace hnf
