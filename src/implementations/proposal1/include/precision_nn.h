#pragma once

#include "precision_tensor.h"
#include <torch/torch.h>
#include <vector>
#include <memory>
#include <map>

namespace hnf {
namespace proposal1 {

// Forward declaration
class PrecisionModule;

// Computation graph for tracking operations
class ComputationGraph {
public:
    struct Node {
        std::string name;
        std::string op_type;
        int required_bits;
        Precision recommended_precision;
        double curvature;
        double lipschitz;
        std::vector<std::shared_ptr<Node>> inputs;
        
        Node(const std::string& n, const std::string& type, 
             int bits, Precision prec, double curv, double lip)
            : name(n), op_type(type), required_bits(bits),
              recommended_precision(prec), curvature(curv), lipschitz(lip) {}
    };
    
    std::vector<std::shared_ptr<Node>> nodes_;
    std::map<std::string, std::shared_ptr<Node>> node_map_;
    
    void add_operation(const std::string& name, const std::string& op_type,
                      const PrecisionTensor& output);
    
    // Access nodes
    const std::map<std::string, std::shared_ptr<Node>>& get_nodes() const { return node_map_; }
    
    // Compute global error bound using Theorem 3.8
    double compute_total_error(double input_error, Precision H) const;
    
    // Find nodes that require more precision than available
    std::vector<std::shared_ptr<Node>> find_critical_nodes(Precision H) const;
    
    // Generate mixed-precision configuration
    std::map<std::string, Precision> generate_precision_config() const;
    
    // Statistics
    double max_curvature() const;
    double total_lipschitz() const;
    int max_required_bits() const;
    
    // Pretty print
    std::string to_string() const;
    void print() const;
};

// Base class for neural network modules with precision tracking
class PrecisionModule {
protected:
    std::string name_;
    ComputationGraph graph_;
    bool is_training_;
    int operation_counter_;
    
    std::string get_unique_op_name(const std::string& base);

public:
    explicit PrecisionModule(const std::string& name = "module")
        : name_(name), is_training_(true), operation_counter_(0) {}
    
    virtual ~PrecisionModule() = default;
    
    // Forward pass - to be implemented by subclasses
    virtual PrecisionTensor forward(const PrecisionTensor& input) = 0;
    
    // Training mode control
    void train() { is_training_ = true; }
    void eval() { is_training_ = false; }
    bool is_training() const { return is_training_; }
    
    // Access computation graph
    const ComputationGraph& graph() const { return graph_; }
    ComputationGraph& graph() { return graph_; }
    
    // Generate precision report
    std::string precision_report() const;
    void print_precision_report() const;
    
    // Check if model can run on given hardware
    bool can_run_on(Precision p) const;
    
    // Get recommended precision configuration
    std::map<std::string, Precision> get_precision_config() const;
};

// Linear layer with precision tracking
class PrecisionLinear : public PrecisionModule {
private:
    torch::nn::Linear impl_;
    int in_features_;
    int out_features_;
    
public:
    PrecisionLinear(int in_features, int out_features, bool bias = true,
                   const std::string& name = "linear");
    
    PrecisionTensor forward(const PrecisionTensor& input) override;
    
    torch::nn::Linear& impl() { return impl_; }
};

// Conv2d layer with precision tracking
class PrecisionConv2d : public PrecisionModule {
private:
    torch::nn::Conv2d impl_;
    
public:
    PrecisionConv2d(int in_channels, int out_channels, int kernel_size,
                   int stride = 1, int padding = 0, bool bias = true,
                   const std::string& name = "conv2d");
    
    PrecisionTensor forward(const PrecisionTensor& input) override;
    
    torch::nn::Conv2d& impl() { return impl_; }
};

// Multi-head attention with precision tracking
class PrecisionMultiHeadAttention : public PrecisionModule {
private:
    int embed_dim_;
    int num_heads_;
    int head_dim_;
    
    torch::nn::Linear q_proj_;
    torch::nn::Linear k_proj_;
    torch::nn::Linear v_proj_;
    torch::nn::Linear out_proj_;
    
public:
    PrecisionMultiHeadAttention(int embed_dim, int num_heads,
                               const std::string& name = "attention");
    
    PrecisionTensor forward(const PrecisionTensor& input) override;
    
    PrecisionTensor forward_qkv(const PrecisionTensor& Q,
                               const PrecisionTensor& K,
                               const PrecisionTensor& V);
};

// Sequential container for chaining modules
class PrecisionSequential : public PrecisionModule {
private:
    std::vector<std::shared_ptr<PrecisionModule>> modules_;
    
public:
    explicit PrecisionSequential(const std::string& name = "sequential");
    
    void add_module(std::shared_ptr<PrecisionModule> module);
    
    template<typename ModuleType, typename... Args>
    void add(Args&&... args) {
        add_module(std::make_shared<ModuleType>(std::forward<Args>(args)...));
    }
    
    PrecisionTensor forward(const PrecisionTensor& input) override;
    
    // Aggregate graphs from all submodules
    void aggregate_graphs();
};

// Simple feedforward network for testing
class SimpleFeedForward : public PrecisionModule {
private:
    std::vector<std::shared_ptr<PrecisionLinear>> layers_;
    std::string activation_;
    
public:
    SimpleFeedForward(const std::vector<int>& layer_sizes,
                     const std::string& activation = "relu",
                     const std::string& name = "feedforward");
    
    PrecisionTensor forward(const PrecisionTensor& input) override;
};

// ResNet block with precision tracking
class ResidualBlock : public PrecisionModule {
private:
    std::shared_ptr<PrecisionConv2d> conv1_;
    std::shared_ptr<PrecisionConv2d> conv2_;
    std::shared_ptr<PrecisionConv2d> shortcut_;  // For dimension matching
    bool use_shortcut_;
    
public:
    ResidualBlock(int in_channels, int out_channels, int stride = 1,
                 const std::string& name = "resblock");
    
    PrecisionTensor forward(const PrecisionTensor& input) override;
};

// Transformer encoder layer
class TransformerEncoderLayer : public PrecisionModule {
private:
    std::shared_ptr<PrecisionMultiHeadAttention> attention_;
    std::shared_ptr<PrecisionLinear> ff1_;
    std::shared_ptr<PrecisionLinear> ff2_;
    int embed_dim_;
    int ff_dim_;
    
public:
    TransformerEncoderLayer(int embed_dim, int num_heads, int ff_dim = 2048,
                           const std::string& name = "transformer_layer");
    
    PrecisionTensor forward(const PrecisionTensor& input) override;
};

} // namespace proposal1
} // namespace hnf
