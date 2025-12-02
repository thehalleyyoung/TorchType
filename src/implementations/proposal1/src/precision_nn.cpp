#include "precision_nn.h"
#include <iomanip>
#include <sstream>

namespace hnf {
namespace proposal1 {

// ============================================================================
// ComputationGraph Implementation
// ============================================================================

void ComputationGraph::add_operation(const std::string& name, const std::string& op_type,
                                     const PrecisionTensor& output) {
    auto node = std::make_shared<Node>(
        name, op_type,
        output.required_bits(),
        output.recommend_precision(),
        output.curvature(),
        output.lipschitz()
    );
    
    nodes_.push_back(node);
    node_map_[name] = node;
}

double ComputationGraph::compute_total_error(double input_error, Precision H) const {
    // Start with input error and propagate through graph
    double current_error = input_error;
    
    for (const auto& node : nodes_) {
        // Accumulate error using Lipschitz constant
        current_error = current_error * node->lipschitz + machine_epsilon(H);
    }
    
    return current_error;
}

std::vector<std::shared_ptr<ComputationGraph::Node>> 
ComputationGraph::find_critical_nodes(Precision H) const {
    std::vector<std::shared_ptr<Node>> critical;
    int available_bits = mantissa_bits(H);
    
    for (const auto& node : nodes_) {
        if (node->required_bits > available_bits) {
            critical.push_back(node);
        }
    }
    
    return critical;
}

std::map<std::string, Precision> ComputationGraph::generate_precision_config() const {
    std::map<std::string, Precision> config;
    
    for (const auto& node : nodes_) {
        config[node->name] = node->recommended_precision;
    }
    
    return config;
}

double ComputationGraph::max_curvature() const {
    double max_curv = 0.0;
    for (const auto& node : nodes_) {
        if (node->curvature > max_curv && !std::isinf(node->curvature)) {
            max_curv = node->curvature;
        }
    }
    return max_curv;
}

double ComputationGraph::total_lipschitz() const {
    double L_total = 1.0;
    for (const auto& node : nodes_) {
        L_total *= node->lipschitz;
    }
    return L_total;
}

int ComputationGraph::max_required_bits() const {
    int max_bits = 0;
    for (const auto& node : nodes_) {
        if (node->required_bits > max_bits && node->required_bits < 200) {
            max_bits = node->required_bits;
        }
    }
    return max_bits;
}

std::string ComputationGraph::to_string() const {
    std::ostringstream oss;
    oss << "\n╔══════════════════════════════════════════════════════════════════════════════╗\n";
    oss << "║                          COMPUTATION GRAPH ANALYSIS                          ║\n";
    oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    
    oss << std::left << std::setw(20) << "║ Operation"
        << std::setw(15) << "Type"
        << std::setw(15) << "Curvature"
        << std::setw(12) << "Bits Req."
        << std::setw(14) << "Recommend" << " ║\n";
    oss << "╟──────────────────────────────────────────────────────────────────────────────╢\n";
    
    for (const auto& node : nodes_) {
        oss << "║ " << std::left << std::setw(18) << node->name.substr(0, 18)
            << std::setw(15) << node->op_type.substr(0, 15);
        
        if (std::isinf(node->curvature)) {
            oss << std::setw(15) << "∞";
        } else if (node->curvature > 1e6) {
            oss << std::scientific << std::setprecision(1) << std::setw(15) << node->curvature;
        } else {
            oss << std::fixed << std::setprecision(2) << std::setw(15) << node->curvature;
        }
        
        oss << std::setw(12) << node->required_bits
            << std::setw(14) << precision_name(node->recommended_precision) << " ║\n";
    }
    
    oss << "╠══════════════════════════════════════════════════════════════════════════════╣\n";
    oss << "║ GLOBAL STATISTICS                                                            ║\n";
    oss << "╟──────────────────────────────────────────────────────────────────────────────╢\n";
    oss << "║ Total Operations:     " << std::setw(53) << nodes_.size() << " ║\n";
    oss << "║ Max Curvature:        " << std::scientific << std::setprecision(2) 
        << std::setw(53) << max_curvature() << " ║\n";
    oss << "║ Total Lipschitz:      " << std::scientific << std::setprecision(2) 
        << std::setw(53) << total_lipschitz() << " ║\n";
    oss << "║ Max Required Bits:    " << std::setw(53) << max_required_bits() << " ║\n";
    oss << "║ Min Precision:        " << std::setw(53) 
        << precision_name(static_cast<Precision>(max_required_bits() <= 7 ? 1 : 
                                                 max_required_bits() <= 10 ? 2 :
                                                 max_required_bits() <= 23 ? 3 :
                                                 max_required_bits() <= 52 ? 4 : 5)) << " ║\n";
    oss << "╚══════════════════════════════════════════════════════════════════════════════╝\n";
    
    return oss.str();
}

void ComputationGraph::print() const {
    std::cout << to_string();
}

// ============================================================================
// PrecisionModule Implementation
// ============================================================================

std::string PrecisionModule::get_unique_op_name(const std::string& base) {
    return base + "_" + std::to_string(operation_counter_++);
}

std::string PrecisionModule::precision_report() const {
    return graph_.to_string();
}

void PrecisionModule::print_precision_report() const {
    graph_.print();
}

bool PrecisionModule::can_run_on(Precision p) const {
    auto critical = graph_.find_critical_nodes(p);
    return critical.empty();
}

std::map<std::string, Precision> PrecisionModule::get_precision_config() const {
    return graph_.generate_precision_config();
}

// ============================================================================
// PrecisionLinear Implementation
// ============================================================================

PrecisionLinear::PrecisionLinear(int in_features, int out_features, bool bias,
                                const std::string& name)
    : PrecisionModule(name),
      impl_(torch::nn::LinearOptions(in_features, out_features).bias(bias)),
      in_features_(in_features),
      out_features_(out_features)
{
}

PrecisionTensor PrecisionLinear::forward(const PrecisionTensor& input) {
    auto output = ops::matmul(input, PrecisionTensor(impl_->weight.t()));
    
    if (impl_->options.bias()) {
        output = ops::add(output, PrecisionTensor(impl_->bias));
    }
    
    graph_.add_operation(get_unique_op_name(name_), "linear", output);
    
    return output;
}

// ============================================================================
// PrecisionConv2d Implementation
// ============================================================================

PrecisionConv2d::PrecisionConv2d(int in_channels, int out_channels, int kernel_size,
                                int stride, int padding, bool bias,
                                const std::string& name)
    : PrecisionModule(name),
      impl_(torch::nn::Conv2dOptions(in_channels, out_channels, kernel_size)
                .stride(stride)
                .padding(padding)
                .bias(bias))
{
}

PrecisionTensor PrecisionConv2d::forward(const PrecisionTensor& input) {
    torch::Tensor bias = impl_->options.bias() ? impl_->bias : torch::Tensor();
    auto output = ops::conv2d(input, impl_->weight, bias);
    
    graph_.add_operation(get_unique_op_name(name_), "conv2d", output);
    
    return output;
}

// ============================================================================
// PrecisionMultiHeadAttention Implementation
// ============================================================================

PrecisionMultiHeadAttention::PrecisionMultiHeadAttention(int embed_dim, int num_heads,
                                                        const std::string& name)
    : PrecisionModule(name),
      embed_dim_(embed_dim),
      num_heads_(num_heads),
      head_dim_(embed_dim / num_heads),
      q_proj_(torch::nn::LinearOptions(embed_dim, embed_dim)),
      k_proj_(torch::nn::LinearOptions(embed_dim, embed_dim)),
      v_proj_(torch::nn::LinearOptions(embed_dim, embed_dim)),
      out_proj_(torch::nn::LinearOptions(embed_dim, embed_dim))
{
}

PrecisionTensor PrecisionMultiHeadAttention::forward(const PrecisionTensor& input) {
    // Self-attention: Q=K=V=input
    auto Q = ops::matmul(input, PrecisionTensor(q_proj_->weight.t()));
    auto K = ops::matmul(input, PrecisionTensor(k_proj_->weight.t()));
    auto V = ops::matmul(input, PrecisionTensor(v_proj_->weight.t()));
    
    return forward_qkv(Q, K, V);
}

PrecisionTensor PrecisionMultiHeadAttention::forward_qkv(
    const PrecisionTensor& Q,
    const PrecisionTensor& K,
    const PrecisionTensor& V
) {
    auto attn_output = ops::attention(Q, K, V);
    auto output = ops::matmul(attn_output, PrecisionTensor(out_proj_->weight.t()));
    
    graph_.add_operation(get_unique_op_name(name_), "attention", output);
    
    return output;
}

// ============================================================================
// PrecisionSequential Implementation
// ============================================================================

PrecisionSequential::PrecisionSequential(const std::string& name)
    : PrecisionModule(name)
{
}

void PrecisionSequential::add_module(std::shared_ptr<PrecisionModule> module) {
    modules_.push_back(module);
}

PrecisionTensor PrecisionSequential::forward(const PrecisionTensor& input) {
    PrecisionTensor output = input;
    
    for (auto& module : modules_) {
        output = module->forward(output);
    }
    
    aggregate_graphs();
    
    return output;
}

void PrecisionSequential::aggregate_graphs() {
    // Merge all submodule graphs into this graph
    graph_.nodes_.clear();
    graph_.node_map_.clear();
    
    for (const auto& module : modules_) {
        const auto& subgraph = module->graph();
        for (const auto& node : subgraph.nodes_) {
            graph_.nodes_.push_back(node);
            graph_.node_map_[node->name] = node;
        }
    }
}

// ============================================================================
// SimpleFeedForward Implementation
// ============================================================================

SimpleFeedForward::SimpleFeedForward(const std::vector<int>& layer_sizes,
                                    const std::string& activation,
                                    const std::string& name)
    : PrecisionModule(name), activation_(activation)
{
    for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
        layers_.push_back(std::make_shared<PrecisionLinear>(
            layer_sizes[i], layer_sizes[i+1], true,
            "fc_" + std::to_string(i)
        ));
    }
}

PrecisionTensor SimpleFeedForward::forward(const PrecisionTensor& input) {
    PrecisionTensor x = input;
    
    for (size_t i = 0; i < layers_.size(); ++i) {
        x = layers_[i]->forward(x);
        
        // Apply activation (except last layer)
        if (i < layers_.size() - 1) {
            if (activation_ == "relu") {
                x = ops::relu(x);
            } else if (activation_ == "tanh") {
                x = ops::tanh(x);
            } else if (activation_ == "sigmoid") {
                x = ops::sigmoid(x);
            } else if (activation_ == "gelu") {
                x = ops::gelu(x);
            }
            
            graph_.add_operation(get_unique_op_name(activation_), activation_, x);
        }
    }
    
    // Aggregate all layer graphs
    for (const auto& layer : layers_) {
        const auto& subgraph = layer->graph();
        for (const auto& node : subgraph.nodes_) {
            if (graph_.node_map_.find(node->name) == graph_.node_map_.end()) {
                graph_.nodes_.push_back(node);
                graph_.node_map_[node->name] = node;
            }
        }
    }
    
    return x;
}

// ============================================================================
// ResidualBlock Implementation
// ============================================================================

ResidualBlock::ResidualBlock(int in_channels, int out_channels, int stride,
                            const std::string& name)
    : PrecisionModule(name),
      use_shortcut_(in_channels != out_channels || stride != 1)
{
    conv1_ = std::make_shared<PrecisionConv2d>(in_channels, out_channels, 3, stride, 1, false, "conv1");
    conv2_ = std::make_shared<PrecisionConv2d>(out_channels, out_channels, 3, 1, 1, false, "conv2");
    
    if (use_shortcut_) {
        shortcut_ = std::make_shared<PrecisionConv2d>(in_channels, out_channels, 1, stride, 0, false, "shortcut");
    }
}

PrecisionTensor ResidualBlock::forward(const PrecisionTensor& input) {
    auto out = conv1_->forward(input);
    out = ops::relu(out);
    out = conv2_->forward(out);
    
    PrecisionTensor residual = input;
    if (use_shortcut_) {
        residual = shortcut_->forward(input);
    }
    
    out = ops::add(out, residual);
    out = ops::relu(out);
    
    // Aggregate graphs
    for (const auto& subgraph : {conv1_->graph(), conv2_->graph()}) {
        for (const auto& node : subgraph.nodes_) {
            if (graph_.node_map_.find(node->name) == graph_.node_map_.end()) {
                graph_.nodes_.push_back(node);
                graph_.node_map_[node->name] = node;
            }
        }
    }
    if (use_shortcut_) {
        for (const auto& node : shortcut_->graph().nodes_) {
            if (graph_.node_map_.find(node->name) == graph_.node_map_.end()) {
                graph_.nodes_.push_back(node);
                graph_.node_map_[node->name] = node;
            }
        }
    }
    
    return out;
}

// ============================================================================
// TransformerEncoderLayer Implementation
// ============================================================================

TransformerEncoderLayer::TransformerEncoderLayer(int embed_dim, int num_heads, int ff_dim,
                                                const std::string& name)
    : PrecisionModule(name),
      embed_dim_(embed_dim),
      ff_dim_(ff_dim)
{
    attention_ = std::make_shared<PrecisionMultiHeadAttention>(embed_dim, num_heads, "mha");
    ff1_ = std::make_shared<PrecisionLinear>(embed_dim, ff_dim, true, "ff1");
    ff2_ = std::make_shared<PrecisionLinear>(ff_dim, embed_dim, true, "ff2");
}

PrecisionTensor TransformerEncoderLayer::forward(const PrecisionTensor& input) {
    // Multi-head attention
    auto attn_out = attention_->forward(input);
    
    // Add & Norm (residual connection)
    auto x = ops::add(input, attn_out);
    x = ops::layer_norm(x, {embed_dim_});
    
    // Feed-forward
    auto ff_out = ff1_->forward(x);
    ff_out = ops::gelu(ff_out);
    ff_out = ff2_->forward(ff_out);
    
    // Add & Norm
    auto output = ops::add(x, ff_out);
    output = ops::layer_norm(output, {embed_dim_});
    
    // Aggregate graphs
    for (const auto& submodule : {
        std::static_pointer_cast<PrecisionModule>(attention_),
        std::static_pointer_cast<PrecisionModule>(ff1_),
        std::static_pointer_cast<PrecisionModule>(ff2_)
    }) {
        for (const auto& node : submodule->graph().nodes_) {
            if (graph_.node_map_.find(node->name) == graph_.node_map_.end()) {
                graph_.nodes_.push_back(node);
                graph_.node_map_[node->name] = node;
            }
        }
    }
    
    return output;
}

} // namespace proposal1
} // namespace hnf
