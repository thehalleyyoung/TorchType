#pragma once

#include "precision_tensor.h"
#include "precision_nn.h"
#include <torch/torch.h>
#include <string>
#include <vector>
#include <memory>

namespace hnf {
namespace proposal1 {

/**
 * MNIST data loader with precision tracking
 * Downloads and loads MNIST dataset for comprehensive experiments
 */
class MNISTDataset {
public:
    struct Sample {
        torch::Tensor image;  // 28x28 flattened to 784
        int64_t label;
    };
    
    MNISTDataset(const std::string& data_dir, bool train = true);
    
    // Get batch of samples
    std::vector<Sample> get_batch(size_t batch_size, size_t offset = 0);
    
    size_t size() const { return images_.size(0); }
    
    // Create synthetic data if MNIST not available
    static MNISTDataset create_synthetic(size_t num_samples = 1000);
    
private:
    torch::Tensor images_;   // [N, 784]
    torch::Tensor labels_;   // [N]
    bool train_;
    
    void load_mnist(const std::string& data_dir);
    void generate_synthetic(size_t num_samples);
};

/**
 * Precision-aware MNIST trainer
 * Implements actual training with precision tracking and validation
 */
class MNISTTrainer {
public:
    struct TrainingConfig {
        size_t batch_size;
        size_t num_epochs;
        double learning_rate;
        double target_accuracy;  // Target numerical accuracy (not classification)
        bool use_mixed_precision;
        bool track_gradients;
        bool verbose;
        
        TrainingConfig() 
            : batch_size(32), num_epochs(5), learning_rate(0.01),
              target_accuracy(1e-6), use_mixed_precision(false),
              track_gradients(true), verbose(true) {}
    };
    
    struct TrainingStats {
        std::vector<double> train_losses;
        std::vector<double> train_accuracies;
        std::vector<double> val_accuracies;
        std::vector<double> max_curvatures;  // Per epoch
        std::vector<int> max_precision_bits; // Per epoch
        std::vector<double> gradient_norms;
        
        // Precision tracking
        std::map<std::string, Precision> operation_precisions;
        std::map<std::string, double> operation_curvatures;
        
        void print_summary() const;
    };
    
    MNISTTrainer(
        std::shared_ptr<PrecisionModule> model,
        const TrainingConfig& config = TrainingConfig()
    );
    
    // Train model and return statistics
    TrainingStats train(MNISTDataset& train_data, MNISTDataset& val_data);
    
    // Evaluate model
    double evaluate(MNISTDataset& test_data);
    
    // Test precision predictions
    struct PrecisionTest {
        Precision predicted_min_precision;
        std::map<Precision, bool> compatibility;
        std::map<Precision, double> actual_accuracies;
        bool prediction_correct;
    };
    
    PrecisionTest test_precision_predictions(MNISTDataset& test_data);
    
    // Experiment: Train at different precisions and compare
    struct ComparativeExperiment {
        std::map<Precision, double> final_accuracies;
        std::map<Precision, double> training_times;
        std::map<Precision, bool> numerical_stability;
        Precision hnf_recommendation;
        bool hnf_correct;
        
        void print_results() const;
    };
    
    ComparativeExperiment run_comparative_experiment(
        MNISTDataset& train_data,
        MNISTDataset& val_data
    );
    
private:
    std::shared_ptr<PrecisionModule> model_;
    TrainingConfig config_;
    
    // Track precision requirements during training
    void track_precision_stats(TrainingStats& stats);
    
    // Apply mixed precision based on HNF analysis
    void apply_mixed_precision();
};

/**
 * Gradient precision analyzer
 * Extends HNF to backpropagation - tracks precision through gradient computation
 */
class GradientPrecisionAnalyzer {
public:
    struct GradientStats {
        double max_gradient_curvature;
        int required_bits_forward;
        int required_bits_backward;
        std::map<std::string, double> per_layer_gradient_curvature;
        std::map<std::string, int> per_layer_gradient_bits;
        
        void print() const;
    };
    
    GradientPrecisionAnalyzer(std::shared_ptr<PrecisionModule> model);
    
    // Analyze gradient precision requirements
    GradientStats analyze(const PrecisionTensor& loss);
    
    // Check if gradients will be stable at given precision
    bool are_gradients_stable(Precision p) const;
    
private:
    std::shared_ptr<PrecisionModule> model_;
    GradientStats latest_stats_;
};

/**
 * Adversarial precision tester
 * Creates challenging numerical scenarios to validate HNF predictions
 */
class AdversarialPrecisionTester {
public:
    enum class TestCase {
        CATASTROPHIC_CANCELLATION,    // Gallery Example 1
        EXPONENTIAL_EXPLOSION,        // High curvature
        NEAR_SINGULAR_MATRIX,         // Matrix inversion
        EXTREME_SOFTMAX,              // Large logits
        DEEP_COMPOSITION,             // Error accumulation
        GRADIENT_VANISHING,           // Backprop through deep network
        GRADIENT_EXPLOSION            // Unstable gradients
    };
    
    struct TestResult {
        TestCase test_case;
        std::string description;
        double predicted_required_bits;
        double actual_required_bits;
        bool prediction_accurate;  // Within 2x factor
        double error_ratio;        // actual/predicted
        
        void print() const;
    };
    
    AdversarialPrecisionTester(std::shared_ptr<PrecisionModule> model);
    
    // Run specific test case
    TestResult run_test(TestCase test_case);
    
    // Run all adversarial tests
    std::vector<TestResult> run_all_tests();
    
    // Success rate of HNF predictions
    double compute_prediction_accuracy(const std::vector<TestResult>& results);
    
private:
    std::shared_ptr<PrecisionModule> model_;
    
    // Individual test implementations
    TestResult test_catastrophic_cancellation();
    TestResult test_exponential_explosion();
    TestResult test_near_singular_matrix();
    TestResult test_extreme_softmax();
    TestResult test_deep_composition();
    TestResult test_gradient_vanishing();
    TestResult test_gradient_explosion();
    
    // Helper: Measure actual bits needed via binary search
    double measure_actual_bits_needed(
        std::function<double(Precision)> test_func,
        double tolerance = 0.1
    );
};

} // namespace proposal1
} // namespace hnf
