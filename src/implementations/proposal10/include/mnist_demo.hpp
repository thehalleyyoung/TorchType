#pragma once

#include "stability_linter.hpp"
#include <torch/torch.h>
#include <vector>
#include <string>

namespace hnf {
namespace stability_linter {

// Demonstrate actual numerical impact on MNIST classification
class MNISTNumericalDemo {
public:
    // Simple feedforward network for MNIST
    struct SimpleNet : torch::nn::Module {
        torch::nn::Linear fc1{nullptr}, fc2{nullptr}, fc3{nullptr};
        
        SimpleNet(int64_t input_size = 784, int64_t hidden1 = 256, 
                  int64_t hidden2 = 128, int64_t num_classes = 10);
        
        torch::Tensor forward(torch::Tensor x);
        
        // Forward pass with explicit precision control
        torch::Tensor forward_mixed_precision(torch::Tensor x, 
                                              const std::map<std::string, torch::ScalarType>& layer_precision);
    };
    
    struct TrainingConfig {
        int64_t batch_size = 64;
        int64_t epochs = 5;
        double learning_rate = 0.001;
        std::string device = "cpu";
        bool use_mixed_precision = false;
    };
    
    struct PrecisionExperiment {
        std::string name;
        torch::ScalarType fc1_precision;
        torch::ScalarType fc2_precision;
        torch::ScalarType fc3_precision;
        double final_accuracy;
        double avg_loss;
        std::vector<double> layer_condition_numbers;
        int total_bits_used;
    };
    
    MNISTNumericalDemo();
    
    // Download and prepare MNIST data
    bool prepare_mnist_data(const std::string& data_dir = "./data");
    
    // Train network with specified precision
    PrecisionExperiment train_with_precision(
        const std::string& experiment_name,
        torch::ScalarType fc1_prec,
        torch::ScalarType fc2_prec,
        torch::ScalarType fc3_prec,
        const TrainingConfig& config = TrainingConfig()
    );
    
    // Run comprehensive precision comparison
    struct ComparisonResults {
        std::vector<PrecisionExperiment> experiments;
        std::string summary_table;
        std::string hnf_predictions;
        std::string recommendations;
    };
    
    ComparisonResults run_precision_comparison();
    
    // Analyze network using HNF linter
    LintReport analyze_network_stability(SimpleNet& net);
    
    // Compute HNF-predicted precision requirements
    std::map<std::string, int> predict_precision_requirements(
        const SimpleNet& net, 
        double target_accuracy = 1e-3
    );
    
    // Verify HNF predictions against actual experiments
    struct VerificationResult {
        bool hnf_predictions_accurate;
        std::map<std::string, double> prediction_errors;
        std::string verification_summary;
    };
    
    VerificationResult verify_hnf_predictions(const ComparisonResults& results);
    
private:
    std::shared_ptr<torch::data::datasets::MNIST> train_dataset_;
    std::shared_ptr<torch::data::datasets::MNIST> test_dataset_;
    bool data_ready_ = false;
    
    // Helper: train one epoch
    double train_epoch(SimpleNet& net, 
                      torch::optim::Optimizer& optimizer,
                      const TrainingConfig& config);
    
    // Helper: evaluate on test set
    double evaluate_accuracy(SimpleNet& net);
    
    // Helper: compute condition numbers for network layers
    std::vector<double> compute_layer_condition_numbers(const SimpleNet& net);
    
    // Helper: convert network to computation graph
    std::shared_ptr<ComputationGraph> net_to_graph(const SimpleNet& net);
};

} // namespace stability_linter
} // namespace hnf
