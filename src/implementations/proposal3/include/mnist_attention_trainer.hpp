#pragma once

#include "attention_types.hpp"
#include "attention_analyzer.hpp"
#include "attention_curvature.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <fstream>

namespace hnf {
namespace attention {

/**
 * Real MNIST training with HNF-based attention stability monitoring.
 * 
 * This demonstrates the full power of HNF theory:
 * 1. Pre-training stability analysis predicts failures
 * 2. Real-time monitoring detects emerging instabilities
 * 3. Automated interventions prevent training collapse
 * 4. Validates curvature predictions against actual behavior
 */

struct AttentionLayer : torch::nn::Module {
    AttentionLayer(int64_t dim, int64_t num_heads, double temperature = 1.0);
    
    torch::Tensor forward(torch::Tensor x);
    
    // Get last attention weights for analysis
    torch::Tensor get_last_attention_weights() const { return last_attention_weights_; }
    
    torch::nn::Linear query_proj{nullptr}, key_proj{nullptr}, value_proj{nullptr};
    torch::nn::Linear out_proj{nullptr};
    int64_t num_heads_;
    int64_t head_dim_;
    double temperature_;
    
private:
    torch::Tensor last_attention_weights_;
};

struct VisionTransformerMNIST : torch::nn::Module {
    VisionTransformerMNIST(
        int64_t image_size = 28,
        int64_t patch_size = 7,
        int64_t num_classes = 10,
        int64_t dim = 64,
        int64_t depth = 3,
        int64_t num_heads = 4,
        double temperature = 1.0
    );
    
    torch::Tensor forward(torch::Tensor x);
    
    // Get attention weights from all layers
    std::vector<torch::Tensor> get_all_attention_weights() const;
    
    // Patch embedding
    torch::nn::Conv2d patch_embed{nullptr};
    torch::Tensor pos_embed;
    torch::Tensor cls_token;
    
    // Transformer blocks
    std::vector<std::shared_ptr<AttentionLayer>> attention_layers_;
    std::vector<torch::nn::LayerNorm> layer_norms1_;
    std::vector<torch::nn::LayerNorm> layer_norms2_;
    std::vector<torch::nn::Sequential> mlp_layers_;
    
    // Classification head
    torch::nn::LayerNorm final_norm{nullptr};
    torch::nn::Linear head{nullptr};
    
    int64_t num_patches_;
    int64_t dim_;
    int64_t depth_;
};

struct TrainingMetrics {
    double train_loss = 0.0;
    double train_acc = 0.0;
    double test_loss = 0.0;
    double test_acc = 0.0;
    
    // HNF-specific metrics
    double mean_attention_entropy = 0.0;
    double max_curvature = 0.0;
    double min_precision_required = 0.0;
    int num_overflow_warnings = 0;
    int num_entropy_warnings = 0;
    
    std::vector<double> per_layer_curvature;
    std::vector<double> per_layer_precision;
};

struct TrainingConfig {
    int64_t batch_size = 64;
    int64_t num_epochs = 10;
    double learning_rate = 1e-3;
    double temperature = 1.0;
    int64_t num_heads = 4;
    int64_t dim = 64;
    int64_t depth = 3;
    
    // HNF monitoring config
    bool enable_hnf_monitoring = true;
    int64_t hnf_check_interval = 50;  // Check every N batches
    double entropy_threshold = 0.5;
    double curvature_threshold = 1e6;
    double precision_safety_margin = 1.2;  // Require 20% more bits than minimum
    
    // Intervention config
    bool enable_auto_intervention = true;
    double temperature_adjustment_factor = 1.5;
    double lr_reduction_factor = 0.5;
};

class MNISTAttentionTrainer {
public:
    MNISTAttentionTrainer(const TrainingConfig& config);
    
    // Load MNIST data
    void load_data(const std::string& data_dir);
    
    // Pre-training stability analysis
    StabilityReport analyze_pre_training_stability();
    
    // Train with HNF monitoring
    std::vector<TrainingMetrics> train();
    
    // Evaluate on test set
    TrainingMetrics evaluate();
    
    // Get the model
    std::shared_ptr<VisionTransformerMNIST> get_model() { return model_; }
    
    // Get training history for analysis
    const std::vector<TrainingMetrics>& get_history() const { return history_; }
    
    // Save/load model
    void save_model(const std::string& path);
    void load_model(const std::string& path);
    
private:
    // Training step with HNF monitoring
    TrainingMetrics train_epoch(int64_t epoch);
    
    // Check for instabilities and intervene if needed
    bool check_stability_and_intervene(const TrainingMetrics& metrics);
    
    // Compute HNF metrics for current model state
    void compute_hnf_metrics(TrainingMetrics& metrics);
    
    TrainingConfig config_;
    std::shared_ptr<VisionTransformerMNIST> model_;
    torch::optim::Adam optimizer_{nullptr};
    std::unique_ptr<AttentionAnalyzer> analyzer_;
    
    // Data
    std::vector<std::pair<torch::Tensor, torch::Tensor>> train_data_;
    std::vector<std::pair<torch::Tensor, torch::Tensor>> test_data_;
    
    // Training history
    std::vector<TrainingMetrics> history_;
    
    // Intervention tracking
    int num_temperature_adjustments_ = 0;
    int num_lr_reductions_ = 0;
    double current_temperature_;
    double current_lr_;
};

/**
 * Comparative experiment: train with and without HNF monitoring.
 * Shows that HNF can predict and prevent training failures.
 */
class ComparativeExperiment {
public:
    struct ExperimentResult {
        std::string config_name;
        bool training_succeeded;
        int epochs_completed;
        double final_test_acc;
        double max_curvature_observed;
        int num_instabilities_detected;
        std::vector<TrainingMetrics> history;
    };
    
    // Run multiple configurations
    std::vector<ExperimentResult> run_experiments();
    
    // Generate comparison report
    void generate_report(const std::string& output_path);
    
private:
    std::vector<TrainingConfig> generate_configs();
    ExperimentResult run_single_experiment(const TrainingConfig& config);
};

/**
 * MNIST data loader.
 * Downloads and processes MNIST for vision transformer.
 */
class MNISTLoader {
public:
    static std::pair<
        std::vector<std::pair<torch::Tensor, torch::Tensor>>,
        std::vector<std::pair<torch::Tensor, torch::Tensor>>
    > load(const std::string& data_dir, int64_t batch_size);
    
private:
    static torch::Tensor load_images(const std::string& path);
    static torch::Tensor load_labels(const std::string& path);
};

} // namespace attention
} // namespace hnf
