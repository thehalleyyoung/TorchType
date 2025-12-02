#pragma once

#include "kv_cache_types.hpp"
#include "kv_cache_analyzer.hpp"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <map>

namespace hnf {
namespace kv_cache {

/**
 * Real-world validation using actual language model sequences
 * 
 * This validates the HNF theorem predictions on:
 * 1. Real text sequences (WikiText, code, conversations)
 * 2. Actual attention patterns from real models
 * 3. Quality degradation when using predicted precisions
 * 4. Memory savings in practice
 */
class RealDataValidator {
public:
    struct ValidationConfig {
        std::string dataset_name;  // "wikitext", "code", "conversation"
        int num_samples;
        int max_sequence_length;
        double quality_threshold;
        bool measure_perplexity;
        bool measure_next_token_accuracy;
        bool measure_downstream_task_performance;
    };
    
    struct ValidationMetrics {
        // Compression metrics
        double compression_ratio;
        double memory_saved_gb;
        
        // Quality metrics
        double perplexity_degradation;  // % increase
        double next_token_accuracy;     // exact match %
        double bleu_score;              // for generation quality
        double output_similarity;       // cosine sim of outputs
        
        // Precision distribution
        std::map<PrecisionLevel, double> precision_distribution;
        
        // Per-layer metrics
        std::vector<double> layer_wise_compression;
        std::vector<double> layer_wise_quality;
        
        // HNF theorem validation
        struct {
            bool all_positions_meet_bound;
            double avg_bound_sharpness;
            int positions_violating_bound;
            double max_observed_error;
            double max_theoretical_error;
        } theorem_validation;
        
        // Comparison to baselines
        struct {
            double uniform_fp16_memory;
            double uniform_int8_memory;
            double uniform_int8_quality_loss;
            double hnf_outperformance;  // quality @ same memory
        } baseline_comparison;
    };
    
    /**
     * Load a real dataset for validation
     */
    static std::vector<torch::Tensor> load_dataset(
        const std::string& dataset_name,
        int num_samples,
        int max_length
    );
    
    /**
     * Run comprehensive validation on real data
     */
    static ValidationMetrics validate_on_dataset(
        KVCacheAnalyzer& analyzer,
        const ValidationConfig& config
    );
    
    /**
     * Measure perplexity degradation with adaptive precision
     */
    static double measure_perplexity_degradation(
        const torch::Tensor& logits_full_precision,
        const torch::Tensor& logits_adaptive_precision,
        const torch::Tensor& targets
    );
    
    /**
     * Measure next-token prediction accuracy
     */
    static double measure_next_token_accuracy(
        const torch::Tensor& logits_full,
        const torch::Tensor& logits_adaptive,
        const torch::Tensor& targets
    );
    
    /**
     * Compute BLEU score for generated sequences
     */
    static double compute_bleu_score(
        const std::vector<std::vector<int>>& references,
        const std::vector<std::vector<int>>& hypotheses
    );
    
    /**
     * Verify HNF theorem predictions on real attention patterns
     */
    static bool verify_theorem_on_real_data(
        const std::vector<PositionCurvature>& curvatures,
        const std::vector<PrecisionLevel>& precisions,
        const torch::Tensor& observed_errors
    );
    
    /**
     * Compare to baseline methods
     */
    static void compare_to_baselines(
        const torch::Tensor& full_precision_kv,
        const torch::Tensor& adaptive_precision_kv,
        ValidationMetrics& metrics
    );
    
    /**
     * Generate detailed validation report
     */
    static std::string generate_validation_report(
        const ValidationMetrics& metrics
    );

private:
    // Helper: Load WikiText dataset
    static std::vector<torch::Tensor> load_wikitext(int num_samples, int max_length);
    
    // Helper: Load code dataset
    static std::vector<torch::Tensor> load_code_dataset(int num_samples, int max_length);
    
    // Helper: Load conversation dataset
    static std::vector<torch::Tensor> load_conversation_dataset(int num_samples, int max_length);
    
    // Helper: Compute perplexity
    static double compute_perplexity(
        const torch::Tensor& logits,
        const torch::Tensor& targets
    );
};

/**
 * Ablation studies to understand what matters most
 */
class AblationStudy {
public:
    struct AblationConfig {
        bool use_attention_based_curvature;
        bool use_gradient_based_curvature;
        bool use_hessian_based_curvature;
        bool use_dynamic_adjustment;
        bool use_per_layer_optimization;
        bool use_per_head_optimization;
    };
    
    struct AblationResults {
        std::map<std::string, double> component_contributions;
        std::string most_important_component;
        double baseline_compression;
        double full_system_compression;
    };
    
    /**
     * Run ablation study to determine which components matter most
     */
    static AblationResults run_ablation_study(
        KVCacheAnalyzer& analyzer,
        const std::vector<torch::Tensor>& dataset
    );
    
    /**
     * Test sensitivity to hyperparameters
     */
    static std::map<std::string, double> test_hyperparameter_sensitivity(
        KVCacheAnalyzer& analyzer,
        const std::vector<torch::Tensor>& dataset
    );
};

/**
 * Stress tests for edge cases and failure modes
 */
class StressTest {
public:
    /**
     * Test on pathological attention patterns
     */
    static bool test_pathological_attention(
        KVCacheAnalyzer& analyzer
    );
    
    /**
     * Test on very long sequences (> 32K tokens)
     */
    static bool test_ultra_long_sequences(
        KVCacheAnalyzer& analyzer
    );
    
    /**
     * Test numerical stability at extreme precisions
     */
    static bool test_numerical_stability(
        KVCacheAnalyzer& analyzer
    );
    
    /**
     * Test recovery from precision underallocation
     */
    static bool test_error_recovery(
        KVCacheAnalyzer& analyzer
    );
};

} // namespace kv_cache
} // namespace hnf
