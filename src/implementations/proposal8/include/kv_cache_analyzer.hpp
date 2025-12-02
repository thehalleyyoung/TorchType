#pragma once

#include "kv_cache_types.hpp"
#include "curvature_analyzer.hpp"
#include "precision_mapper.hpp"
#include "mixed_precision_buffer.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace hnf {
namespace kv_cache {

/**
 * KVCacheAnalyzer: Main interface for KV cache precision analysis
 * 
 * Orchestrates the complete analysis pipeline:
 * 1. Collect attention patterns from calibration runs
 * 2. Compute position-wise curvature scores
 * 3. Map curvatures to precision requirements
 * 4. Optimize for memory/quality trade-offs
 * 5. Generate adaptive precision KV cache
 */
class KVCacheAnalyzer {
public:
    explicit KVCacheAnalyzer(const KVCacheConfig& config);
    
    /**
     * Main analysis entry point
     * 
     * @param calibration_data: List of input sequences for calibration
     * @param forward_fn: Function that runs forward pass and returns attention
     *                    Should have signature: 
     *                    (input) -> (output, attention_weights_per_layer)
     * @return: Complete precision analysis result
     */
    PrecisionAnalysisResult analyze(
        const std::vector<torch::Tensor>& calibration_data,
        std::function<std::pair<torch::Tensor, std::vector<torch::Tensor>>(const torch::Tensor&)> forward_fn
    );
    
    /**
     * Analyze with full model hooks (more detailed)
     * Hooks into attention layers to collect patterns
     */
    PrecisionAnalysisResult analyze_with_hooks(
        torch::nn::Module& model,
        const std::vector<torch::Tensor>& calibration_data
    );
    
    /**
     * Create adaptive KV cache from analysis result
     */
    std::shared_ptr<AdaptivePrecisionKVCache> create_adaptive_cache(
        const PrecisionAnalysisResult& analysis
    );
    
    /**
     * Analyze a single layer
     */
    LayerPrecisionMap analyze_layer(
        int64_t layer_idx,
        const std::vector<torch::Tensor>& attention_weights_samples,
        const std::vector<torch::Tensor>& keys_samples,
        const std::vector<torch::Tensor>& values_samples,
        const std::vector<torch::Tensor>& queries_samples
    );
    
    /**
     * Generate recommendations based on analysis
     */
    std::vector<std::string> generate_recommendations(
        const PrecisionAnalysisResult& analysis
    ) const;
    
    /**
     * Print detailed analysis report
     */
    void print_analysis_report(
        const PrecisionAnalysisResult& analysis,
        std::ostream& os = std::cout
    ) const;
    
    /**
     * Save analysis to file
     */
    void save_analysis(
        const PrecisionAnalysisResult& analysis,
        const std::string& filepath
    ) const;
    
    /**
     * Load analysis from file
     */
    static PrecisionAnalysisResult load_analysis(
        const std::string& filepath
    );
    
private:
    KVCacheConfig config_;
    std::unique_ptr<CurvatureAnalyzer> curvature_analyzer_;
    std::unique_ptr<PrecisionMapper> precision_mapper_;
    
    // Collected data during analysis
    std::map<int64_t, std::vector<AttentionPattern>> layer_attention_patterns_;
    std::map<int64_t, std::vector<std::vector<PositionCurvature>>> layer_position_curvatures_;
    
    // Aggregate curvatures across multiple calibration samples
    std::vector<PositionCurvature> aggregate_curvatures(
        const std::vector<std::vector<PositionCurvature>>& samples
    );
    
    // Compute global statistics
    void compute_global_statistics(PrecisionAnalysisResult& result);
    
    // Validate analysis result
    bool validate_analysis(const PrecisionAnalysisResult& result) const;
};

/**
 * DynamicPrecisionAdjuster: Adjusts precision on-the-fly during inference
 * 
 * Monitors attention patterns and upgrades/downgrades precision as needed
 */
class DynamicPrecisionAdjuster {
public:
    explicit DynamicPrecisionAdjuster(
        const KVCacheConfig& config,
        std::shared_ptr<AdaptivePrecisionKVCache> cache
    );
    
    /**
     * Update importance scores based on current attention
     */
    void update_importance(
        int64_t layer_idx,
        const torch::Tensor& attention_weights
    );
    
    /**
     * Get current precision for a position
     */
    PrecisionLevel get_precision(int64_t layer_idx, int64_t position);
    
    /**
     * Trigger precision adjustment if needed
     * Returns true if any precision was changed
     */
    bool adjust_precisions(int64_t layer_idx);
    
    /**
     * Set upgrade threshold (attention weight above which to upgrade)
     */
    void set_upgrade_threshold(double threshold) {
        upgrade_threshold_ = threshold;
    }
    
    /**
     * Set downgrade threshold (attention weight below which to downgrade)
     */
    void set_downgrade_threshold(double threshold) {
        downgrade_threshold_ = threshold;
    }
    
private:
    KVCacheConfig config_;
    std::shared_ptr<AdaptivePrecisionKVCache> cache_;
    
    // Exponential moving average of position importance
    std::map<int64_t, torch::Tensor> position_importance_;
    
    // Thresholds for precision changes
    double upgrade_threshold_ = 0.1;
    double downgrade_threshold_ = 0.01;
    double ema_alpha_ = 0.1;  // EMA smoothing factor
    
    // Decide precision based on importance
    PrecisionLevel importance_to_precision(double importance);
};

} // namespace kv_cache
} // namespace hnf
