#pragma once

#include "precision_tensor.h"
#include "precision_nn.h"
#include "rigorous_curvature.h"
#include <torch/torch.h>
#include <vector>
#include <chrono>
#include <fstream>

namespace hnf {
namespace proposal1 {

/**
 * @brief Actual Training Demonstration Framework
 * 
 * This class implements REAL training experiments that demonstrate:
 * 1. Wall-clock time improvements from mixed precision
 * 2. Memory savings from precision-aware computation  
 * 3. Numerical stability improvements from curvature tracking
 * 4. Concrete accuracy vs. precision trade-offs
 * 
 * Unlike toy examples, this uses actual PyTorch training loops on real tasks.
 */
class ActualTrainingDemo {
public:
    struct TrainingConfig {
        int batch_size = 128;
        int num_epochs = 10;
        double learning_rate = 0.001;
        Precision forward_precision = Precision::FLOAT32;
        Precision backward_precision = Precision::FLOAT32;
        bool track_curvature = true;
        bool use_mps = false;  // Apple Metal Performance Shaders
        std::string device = "cpu";
    };
    
    struct TrainingMetrics {
        std::vector<double> train_losses;
        std::vector<double> test_accuracies;
        std::vector<double> curvatures;
        std::vector<double> gradient_norms;
        std::vector<double> wall_clock_times;
        double total_training_time_ms = 0.0;
        double peak_memory_mb = 0.0;
        int num_nan_events = 0;
        int num_precision_escalations = 0;
        
        void save_to_csv(const std::string& filename) const;
        void print_summary() const;
    };
    
    /**
     * @brief Train a simple CNN on MNIST with precision tracking
     * 
     * This demonstrates:
     * - How curvature evolves during training
     * - When precision escalation is needed
     * - Concrete speedup from mixed precision
     */
    static TrainingMetrics train_mnist_cnn(const TrainingConfig& config);
    
    /**
     * @brief Train a small transformer on sequence modeling
     * 
     * Demonstrates attention layer precision requirements
     */
    static TrainingMetrics train_small_transformer(
        const TrainingConfig& config,
        int seq_length = 32,
        int vocab_size = 1000,
        int d_model = 128,
        int num_heads = 4
    );
    
    /**
     * @brief Compare different precision configurations
     * 
     * Runs the same training with different precision settings and
     * compares: wall time, memory, final accuracy, stability
     */
    static std::map<std::string, TrainingMetrics> compare_precision_configs(
        const std::string& task,  // "mnist" or "transformer"
        const std::vector<std::pair<Precision, Precision>>& configs
    );
    
    /**
     * @brief Demonstrate curvature-guided learning rate scheduling
     * 
     * Uses κ(t) to adjust LR during training:
     * - High curvature → reduce LR
     * - Low curvature → increase LR
     * 
     * Shows concrete improvement over constant LR
     */
    static std::pair<TrainingMetrics, TrainingMetrics> demonstrate_curvature_lr_scheduling();
    
    /**
     * @brief Demonstrate automatic precision escalation
     * 
     * Start with FP16, automatically escalate to FP32 when NaNs detected.
     * Shows this prevents training failure.
     */
    static TrainingMetrics demonstrate_auto_precision_escalation();
    
    /**
     * @brief Stress test: train a network that REQUIRES FP64
     * 
     * Demonstrates a case where FP32 provably fails (predicted by curvature),
     * but FP64 succeeds.
     */
    static std::pair<TrainingMetrics, TrainingMetrics> stress_test_high_curvature_network();

private:
    // Helper: Load MNIST data
    static std::pair<torch::Tensor, torch::Tensor> load_mnist_subset(int num_samples = 1000);
    
    // Helper: Generate synthetic sequence data
    static std::pair<torch::Tensor, torch::Tensor> generate_sequence_data(
        int num_sequences, int seq_length, int vocab_size
    );
    
    // Helper: Compute current memory usage
    static double get_memory_usage_mb();
    
    // Helper: Check for NaN/Inf in gradients
    static bool has_nan_or_inf(torch::nn::Module& model);
};

/**
 * @brief Concrete Wall-Clock Benchmarks
 * 
 * These are NOT synthetic microbenchmarks - they measure actual training time
 * improvements on real tasks with real data.
 */
class WallClockBenchmarks {
public:
    struct BenchmarkResult {
        std::string operation;
        std::string precision_config;
        double time_ms;
        double memory_mb;
        double numerical_error;
        
        void print() const;
    };
    
    /**
     * @brief Benchmark matrix multiplication at different precisions
     * 
     * Measures actual wall-clock time for different sizes and precisions on MPS/CPU
     */
    static std::vector<BenchmarkResult> benchmark_matmul(
        const std::vector<int>& sizes,
        const std::vector<Precision>& precisions,
        const std::string& device = "cpu"
    );
    
    /**
     * @brief Benchmark attention computation
     * 
     * Critical for transformers - shows where FP32 is actually needed
     */
    static std::vector<BenchmarkResult> benchmark_attention(
        const std::vector<int>& seq_lengths,
        int d_model = 512,
        const std::string& device = "cpu"
    );
    
    /**
     * @brief Benchmark full forward+backward pass
     * 
     * Most realistic - includes gradient computation
     */
    static std::vector<BenchmarkResult> benchmark_forward_backward(
        const std::string& model_type,  // "cnn", "transformer", "resnet"
        const std::vector<std::pair<Precision, Precision>>& configs
    );
    
    /**
     * @brief Memory bandwidth benchmark
     * 
     * Shows memory savings from reduced precision
     */
    static std::vector<BenchmarkResult> benchmark_memory_bandwidth(
        const std::vector<Precision>& precisions
    );
};

/**
 * @brief Numerical Stability Demonstrations
 * 
 * Concrete examples where curvature tracking prevents numerical failure
 */
class StabilityDemonstrations {
public:
    /**
     * @brief Demonstrate gradient explosion prevention
     * 
     * Train a deep network (50+ layers) with and without curvature monitoring.
     * Show that curvature tracking predicts explosions before they happen.
     */
    static void demonstrate_gradient_explosion_prevention();
    
    /**
     * @brief Demonstrate attention NaN prevention
     * 
     * Train transformer with very long sequences. Show that curvature-based
     * precision requirements prevent the NaNs that occur with naive FP16.
     */
    static void demonstrate_attention_nan_prevention();
    
    /**
     * @brief Demonstrate catastrophic cancellation detection
     * 
     * Example from HNF paper: exp(-100) via Taylor series vs. 1/exp(100).
     * Show curvature correctly predicts which fails.
     */
    static void demonstrate_catastrophic_cancellation();
    
    /**
     * @brief Demonstrate BatchNorm precision requirements
     * 
     * BatchNorm with small batch size can have numerical issues.
     * Show curvature predicts the minimum precision needed.
     */
    static void demonstrate_batchnorm_stability();
};

/**
 * @brief Real-World Application Scenarios
 * 
 * End-to-end demonstrations on actual ML tasks
 */
class RealWorldScenarios {
public:
    /**
     * @brief Scenario 1: Deploy a model to edge device
     * 
     * Given: Trained FP32 model
     * Goal: Minimize bit-width while maintaining accuracy
     * 
     * Uses curvature analysis to determine per-layer precision,
     * then validates on actual test set.
     */
    struct DeploymentReport {
        std::map<std::string, Precision> layer_precisions;
        double original_accuracy;
        double quantized_accuracy;
        double memory_savings_percent;
        std::vector<std::string> warnings;
        
        void print() const;
    };
    
    static DeploymentReport edge_deployment_scenario(
        torch::nn::Module& model,
        const torch::Tensor& test_data,
        const torch::Tensor& test_labels,
        double min_acceptable_accuracy = 0.95
    );
    
    /**
     * @brief Scenario 2: Debug training instability
     * 
     * Given: Model that produces NaNs during training
     * Goal: Identify which layer(s) need higher precision
     * 
     * Returns diagnostic report pointing to problematic operations
     */
    struct DebugReport {
        std::vector<std::pair<std::string, double>> culprit_layers;  // (name, curvature)
        std::vector<std::string> recommendations;
        int first_nan_epoch;
        std::string root_cause;
        
        void print() const;
    };
    
    static DebugReport debug_training_instability(
        torch::nn::Module& model,
        const torch::Tensor& train_data,
        const torch::Tensor& train_labels,
        int max_epochs = 10
    );
    
    /**
     * @brief Scenario 3: Optimize inference latency
     * 
     * Given: Trained model, latency budget
     * Goal: Maximize throughput while meeting accuracy target
     * 
     * Searches over mixed-precision configurations guided by curvature
     */
    struct OptimizationReport {
        std::map<std::string, Precision> best_config;
        double throughput_samples_per_sec;
        double accuracy;
        double latency_ms;
        
        void print() const;
    };
    
    static OptimizationReport optimize_inference_latency(
        torch::nn::Module& model,
        const torch::Tensor& test_data,
        const torch::Tensor& test_labels,
        double latency_budget_ms = 10.0
    );
};

/**
 * @brief Comprehensive Test Suite Generator
 * 
 * Generates extensive test cases that stress-test the curvature computations
 */
class ComprehensiveTestGenerator {
public:
    /**
     * @brief Generate adversarial test cases
     * 
     * Creates inputs designed to trigger edge cases:
     * - Near-zero denominators
     * - Very large/small exponents
     * - Ill-conditioned matrices
     * - Extreme attention logits
     */
    static std::vector<torch::Tensor> generate_adversarial_inputs(
        const std::string& operation_type
    );
    
    /**
     * @brief Generate property-based tests
     * 
     * Tests that curvature satisfies expected properties:
     * - κ(f ∘ g) ≤ κ(f) · L_g² + κ(g) · ||Df||
     * - κ(cf) = c² · κ(f) for linear scaling
     * - κ(f + g) ≤ κ(f) + κ(g) + interaction term
     */
    static bool test_curvature_composition_property(int num_trials = 100);
    static bool test_curvature_scaling_property(int num_trials = 100);
    static bool test_curvature_addition_property(int num_trials = 100);
    
    /**
     * @brief Fuzzing-based tests
     * 
     * Random inputs to find corner cases
     */
    static std::vector<std::string> fuzz_curvature_computations(
        int num_trials = 10000
    );
};

} // namespace proposal1
} // namespace hnf
