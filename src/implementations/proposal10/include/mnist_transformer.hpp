#pragma once

#include <torch/torch.h>
#include "stability_linter.hpp"
#include "sheaf_cohomology.hpp"
#include "homotopy_equivalence.hpp"
#include <memory>
#include <vector>

/**
 * Real Transformer Training with Precision Analysis
 * 
 * This module demonstrates the "whole way" - actually training a transformer
 * on MNIST and showing that:
 * 1. HNF curvature bounds predict which precision levels work
 * 2. Unstable implementations fail at predicted precision
 * 3. Stable implementations succeed even at lower precision
 * 
 * This proves HNF theory is not just theoretical but practically useful.
 */

namespace hnf {
namespace mnist_transformer {

using namespace stability_linter;

/**
 * Simple Transformer for MNIST Classification
 * 
 * Implements a minimal transformer architecture:
 * - Patch embedding (28x28 -> 7x7 patches)
 * - Self-attention layers
 * - Classification head
 */
class MNISTTransformer : public torch::nn::Module {
private:
    int patch_size_;
    int embed_dim_;
    int num_heads_;
    int num_layers_;
    
    // Numerical variant
    bool use_stable_softmax_;
    bool use_stable_layernorm_;
    bool use_attention_scaling_;
    
    // Layers
    torch::nn::Linear patch_embed_{nullptr};
    torch::nn::ModuleList attention_layers_;
    torch::nn::ModuleList ffn_layers_;
    torch::nn::Linear classifier_{nullptr};
    
public:
    MNISTTransformer(int patch_size = 4,
                    int embed_dim = 64,
                    int num_heads = 4,
                    int num_layers = 2,
                    bool stable_softmax = true,
                    bool stable_layernorm = true,
                    bool attention_scaling = true);
    
    torch::Tensor forward(torch::Tensor x);
    
    // Attention layer (with numerical variants)
    torch::Tensor attention(torch::Tensor x, bool stable);
    
    // LayerNorm (with numerical variants)
    torch::Tensor layer_norm(torch::Tensor x, bool stable);
    
    // Get computation graph for linting
    std::shared_ptr<ComputationGraph> get_computation_graph();
};

/**
 * MNIST Dataset Loader
 * 
 * Downloads and loads MNIST data for training.
 */
class MNISTDataset {
private:
    std::vector<torch::Tensor> train_images_;
    std::vector<int64_t> train_labels_;
    std::vector<torch::Tensor> test_images_;
    std::vector<int64_t> test_labels_;
    
    bool downloaded_;
    std::string data_dir_;
    
    void download_if_needed();
    void load_data();
    
public:
    MNISTDataset(const std::string& data_dir = "./data/mnist");
    
    struct Batch {
        torch::Tensor images;
        torch::Tensor labels;
        int batch_size;
    };
    
    Batch get_train_batch(int batch_size, int batch_idx) const;
    Batch get_test_batch(int batch_size, int batch_idx) const;
    
    int num_train() const { return train_images_.size(); }
    int num_test() const { return test_images_.size(); }
};

/**
 * Training Configuration
 */
struct TrainingConfig {
    int num_epochs;
    int batch_size;
    double learning_rate;
    torch::DeviceType device;
    
    // Precision settings
    torch::ScalarType dtype;  // torch::kFloat32, torch::kFloat16, etc.
    
    // Numerical variant
    bool use_stable_implementations;
    
    TrainingConfig()
        : num_epochs(10),
          batch_size(128),
          learning_rate(0.001),
          device(torch::kCPU),
          dtype(torch::kFloat32),
          use_stable_implementations(true) {}
};

/**
 * Training Result
 * 
 * Stores training metrics and numerical stability information.
 */
struct TrainingResult {
    std::vector<double> train_losses;
    std::vector<double> train_accuracies;
    std::vector<double> test_accuracies;
    
    // Numerical stability metrics
    bool encountered_nan;
    bool encountered_inf;
    int first_nan_epoch;  // -1 if no NaN
    
    // HNF analysis
    std::vector<double> curvature_estimates;  // per epoch
    std::vector<double> condition_numbers;  // per epoch
    
    std::string summary() const;
    
    // Did training succeed?
    bool successful() const {
        return !encountered_nan && !encountered_inf && 
               !train_losses.empty() && train_losses.back() < 0.5;
    }
};

/**
 * Transformer Trainer
 * 
 * Trains transformer with numerical monitoring.
 */
class TransformerTrainer {
private:
    std::shared_ptr<MNISTTransformer> model_;
    std::shared_ptr<MNISTDataset> dataset_;
    TrainingConfig config_;
    
    // Monitoring
    std::shared_ptr<NumericalLinter> linter_;
    std::shared_ptr<sheaf::SheafLinter> sheaf_linter_;
    
    // Track numerical issues during training
    bool check_for_numerical_issues(const torch::Tensor& tensor) const;
    
    void log_epoch(int epoch, const TrainingResult& result) const;
    
public:
    TransformerTrainer(std::shared_ptr<MNISTTransformer> model,
                      std::shared_ptr<MNISTDataset> dataset,
                      const TrainingConfig& config);
    
    // Train and return results
    TrainingResult train();
    
    // Evaluate on test set
    double evaluate();
    
    // Lint model before training
    LintReport lint_model();
    
    // Analyze precision requirements using HNF
    struct PrecisionAnalysis {
        std::map<std::string, double> layer_precision_requirements;
        double min_precision_for_convergence;
        bool can_use_fp16;
        bool can_use_fp32;
        std::string detailed_report;
    };
    
    PrecisionAnalysis analyze_precision_requirements();
};

/**
 * Comparative Experiment
 * 
 * Compare stable vs unstable implementations at different precisions.
 * This is the main demonstration of HNF theory in practice.
 */
class ComparativeExperiment {
public:
    struct Variant {
        std::string name;
        bool stable_softmax;
        bool stable_layernorm;
        bool attention_scaling;
        torch::ScalarType dtype;
    };
    
    struct ComparisonResult {
        Variant variant;
        TrainingResult training_result;
        LintReport lint_report;
        TransformerTrainer::PrecisionAnalysis precision_analysis;
        
        // Did HNF predictions match reality?
        bool hnf_prediction_correct;
        std::string hnf_explanation;
    };
    
private:
    std::vector<Variant> variants_;
    std::shared_ptr<MNISTDataset> dataset_;
    TrainingConfig base_config_;
    
public:
    ComparativeExperiment(std::shared_ptr<MNISTDataset> dataset,
                         const TrainingConfig& base_config);
    
    // Add variant to test
    void add_variant(const Variant& v);
    
    // Add standard variants
    void add_standard_variants();
    
    // Run all experiments
    std::vector<ComparisonResult> run_all();
    
    // Generate comparison report
    std::string generate_report(const std::vector<ComparisonResult>& results) const;
    
    // Verify HNF predictions
    struct Verification {
        int total_variants;
        int hnf_correct_predictions;
        double accuracy_rate;
        
        std::vector<std::string> hnf_successes;
        std::vector<std::string> hnf_failures;
        
        std::string summary() const;
    };
    
    Verification verify_hnf_predictions(const std::vector<ComparisonResult>& results) const;
};

/**
 * MNIST Transformer Demo
 * 
 * Main entry point for demonstrating HNF theory on real transformers.
 */
class MNISTTransformerDemo {
public:
    static void run_full_demo();
    
    // Individual demonstrations
    static void demo_basic_training();
    static void demo_precision_comparison();
    static void demo_lint_and_fix();
    static void demo_sheaf_analysis();
    static void demo_homotopy_equivalence();
    
    // The grand finale: prove HNF theory works
    static void prove_hnf_theory();
};

} // namespace mnist_transformer
} // namespace hnf
