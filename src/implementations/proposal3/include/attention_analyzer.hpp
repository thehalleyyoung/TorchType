#pragma once

#include "attention_types.hpp"
#include "attention_curvature.hpp"
#include <torch/torch.h>
#include <memory>
#include <functional>

namespace hnf {
namespace attention {

/**
 * Main analyzer for attention stability.
 * 
 * This implements the HNF-based stability analysis from the paper,
 * detecting numerical issues in attention mechanisms and suggesting fixes.
 */
class AttentionAnalyzer {
public:
    explicit AttentionAnalyzer(const AttentionConfig& config = AttentionConfig());
    
    /**
     * Analyze attention pattern for stability.
     * 
     * This is the main entry point. Given Q, K, V tensors, it computes:
     * 1. Curvature bounds (HNF Theorem 4.1)
     * 2. Precision requirements (HNF Theorem 4.2)
     * 3. Error functionals (HNF Theorem 3.1)
     * 4. Stability diagnostics
     */
    AttentionStats analyze_pattern(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& V,
        const std::string& layer_name = "unnamed"
    );
    
    /**
     * Compute attention weights with stability tracking.
     */
    std::pair<torch::Tensor, AttentionStats> compute_attention_with_stats(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& V,
        const std::string& layer_name = "unnamed"
    );
    
    /**
     * Pre-training stability check.
     * 
     * Before training, analyze whether the model architecture
     * will be numerically stable with given configuration.
     */
    AttentionDiagnosis check_pretraining_stability(
        int num_layers,
        const std::vector<std::string>& layer_names = {}
    );
    
    /**
     * Diagnose issues from collected statistics.
     */
    AttentionDiagnosis diagnose(
        const std::map<std::string, std::vector<AttentionStats>>& history
    );
    
    /**
     * Suggest intervention based on diagnosis.
     */
    std::vector<InterventionSuggestion> suggest_interventions(
        const AttentionDiagnosis& diagnosis
    );
    
    /**
     * Predict stability for a given configuration.
     * 
     * Used for architecture design: given sequence length, num heads, etc.,
     * predict whether attention will be stable.
     */
    struct StabilityPrediction {
        bool is_stable;
        double expected_curvature;
        double required_precision_bits;
        std::vector<std::string> warnings;
        std::vector<std::string> recommendations;
    };
    
    StabilityPrediction predict_stability(
        int seq_length,
        int num_heads,
        int head_dim,
        double temperature = 1.0,
        const HardwareModel& hardware = HardwareModel::fp32()
    );
    
    /**
     * Analyze entropy dynamics.
     * 
     * Computes attention entropy H = -sum(A * log(A)) per head.
     * Low entropy indicates attention collapse.
     */
    torch::Tensor compute_entropy(const torch::Tensor& attention_weights);
    
    /**
     * Detect overflow/underflow risks.
     */
    struct OverflowRisk {
        bool overflow_likely;
        bool underflow_likely;
        torch::Tensor max_logits;
        torch::Tensor min_logits;
        std::string recommendation;
    };
    
    OverflowRisk detect_overflow_risk(const torch::Tensor& logits);
    
    /**
     * Analyze gradient vanishing.
     * 
     * Estimates ||∂L/∂Q|| and ||∂L/∂K|| to detect gradient issues.
     */
    struct GradientAnalysis {
        torch::Tensor gradient_norm_Q;
        torch::Tensor gradient_norm_K;
        torch::Tensor gradient_norm_V;
        bool vanishing_detected;
        bool explosion_detected;
    };
    
    GradientAnalysis analyze_gradients(
        const torch::Tensor& Q,
        const torch::Tensor& K,
        const torch::Tensor& V,
        const torch::Tensor& grad_output
    );
    
    /**
     * Compare two attention mechanisms for stability.
     */
    struct ComparisonResult {
        std::string more_stable;
        double curvature_ratio;
        double precision_ratio;
        std::string explanation;
    };
    
    ComparisonResult compare_mechanisms(
        const AttentionStats& stats1,
        const AttentionStats& stats2,
        const std::string& name1 = "mechanism1",
        const std::string& name2 = "mechanism2"
    );
    
    // Configuration accessors
    const AttentionConfig& config() const { return config_; }
    void set_config(const AttentionConfig& config) { config_ = config; }
    
private:
    AttentionConfig config_;
    
    // Internal analysis helpers
    StabilityIssue create_issue(
        const std::string& layer_name,
        int head_index,
        IssueType type,
        double value,
        const torch::Tensor& threshold
    );
    
    Severity determine_severity(IssueType type, double value, double threshold);
    
    std::string generate_message(IssueType type, const std::string& layer, int head, double value);
    
    std::string generate_suggestion(IssueType type, double value);
    
    // Statistical analysis
    torch::Tensor compute_logit_statistics(const torch::Tensor& Q, const torch::Tensor& K);
    
    torch::Tensor estimate_max_logit_random_init(int seq_len, int head_dim);
};

/**
 * Hook-based monitoring for training-time analysis.
 * 
 * This class can be used to monitor attention during training,
 * collecting statistics and detecting issues on-the-fly.
 */
class AttentionMonitor {
public:
    using HookFunction = std::function<void(const std::string&, const AttentionStats&)>;
    
    explicit AttentionMonitor(
        const AttentionConfig& config = AttentionConfig(),
        int log_frequency = 100
    );
    
    /**
     * Record attention statistics.
     */
    void record(const std::string& layer_name, const AttentionStats& stats);
    
    /**
     * Check if monitoring should trigger (based on frequency).
     */
    bool should_monitor(int step);
    
    /**
     * Get diagnosis from accumulated statistics.
     */
    AttentionDiagnosis get_diagnosis();
    
    /**
     * Register callback for real-time alerts.
     */
    void register_hook(HookFunction hook);
    
    /**
     * Clear accumulated statistics.
     */
    void clear();
    
    /**
     * Get statistics history.
     */
    const std::map<std::string, std::vector<AttentionStats>>& get_history() const {
        return history_;
    }
    
private:
    AttentionConfig config_;
    int log_frequency_;
    int current_step_;
    std::map<std::string, std::vector<AttentionStats>> history_;
    std::vector<HookFunction> hooks_;
    AttentionAnalyzer analyzer_;
};

} // namespace attention
} // namespace hnf
