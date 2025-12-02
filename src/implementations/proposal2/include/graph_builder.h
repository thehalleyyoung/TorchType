#pragma once

#include "computation_graph.h"
#include <torch/torch.h>
#include <torch/script.h>

namespace hnf {
namespace sheaf {

// Builder for computation graphs from PyTorch modules
class GraphBuilder {
public:
    // Build graph from transformer attention layer
    static ComputationGraph build_attention_graph(
        int64_t seq_len,
        int64_t d_model,
        int64_t num_heads
    ) {
        ComputationGraph graph;
        
        int64_t d_head = d_model / num_heads;
        
        // Input nodes
        auto q_node = std::make_shared<ComputationNode>("Q", "input");
        q_node->output_shape = {seq_len, d_model};
        q_node->curvature = 0.0;
        q_node->lipschitz = 1.0;
        q_node->diameter = 2.0;  // Assuming normalized inputs
        graph.add_node(q_node);
        
        auto k_node = std::make_shared<ComputationNode>("K", "input");
        k_node->output_shape = {seq_len, d_model};
        k_node->curvature = 0.0;
        k_node->lipschitz = 1.0;
        k_node->diameter = 2.0;
        graph.add_node(k_node);
        
        auto v_node = std::make_shared<ComputationNode>("V", "input");
        v_node->output_shape = {seq_len, d_model};
        v_node->curvature = 0.0;
        v_node->lipschitz = 1.0;
        v_node->diameter = 2.0;
        graph.add_node(v_node);
        
        // QK^T computation
        auto qk_node = std::make_shared<ComputationNode>("QK_T", "matmul");
        qk_node->output_shape = {num_heads, seq_len, seq_len};
        qk_node->curvature = 0.0;  // Bilinear
        qk_node->lipschitz = std::sqrt(static_cast<double>(d_head));
        qk_node->diameter = 4.0 * d_head;
        graph.add_node(qk_node);
        graph.add_edge("Q", "QK_T");
        graph.add_edge("K", "QK_T");
        
        // Scale by 1/sqrt(d_head)
        auto scale_node = std::make_shared<ComputationNode>("scale", "div");
        scale_node->output_shape = {num_heads, seq_len, seq_len};
        scale_node->curvature = 2.0 / (d_head * std::sqrt(static_cast<double>(d_head)));
        scale_node->lipschitz = 1.0 / std::sqrt(static_cast<double>(d_head));
        scale_node->diameter = 4.0 * std::sqrt(static_cast<double>(d_head));
        graph.add_node(scale_node);
        graph.add_edge("QK_T", "scale");
        
        // Softmax - THIS IS THE CRITICAL HIGH-PRECISION OPERATION
        auto softmax_node = std::make_shared<ComputationNode>("softmax", "softmax");
        softmax_node->output_shape = {num_heads, seq_len, seq_len};
        
        // Softmax curvature: κ = 0.5 (from paper Example 4)
        // But composed with QK^T, it's much higher
        double qk_norm = 4.0 * std::sqrt(static_cast<double>(d_head));
        softmax_node->curvature = 0.5 * qk_norm * qk_norm;  // ~362.5 for d=64
        softmax_node->lipschitz = 1.0;  // Softmax is 1-Lipschitz
        softmax_node->diameter = 1.0;  // Output is probability distribution
        graph.add_node(softmax_node);
        graph.add_edge("scale", "softmax");
        
        // Multiply by V
        auto attn_v_node = std::make_shared<ComputationNode>("attn_V", "matmul");
        attn_v_node->output_shape = {seq_len, d_model};
        attn_v_node->curvature = 0.0;  // Bilinear
        attn_v_node->lipschitz = std::sqrt(static_cast<double>(d_head));
        attn_v_node->diameter = std::sqrt(static_cast<double>(d_head));
        graph.add_node(attn_v_node);
        graph.add_edge("softmax", "attn_V");
        graph.add_edge("V", "attn_V");
        
        // Output projection (simplified)
        auto output_node = std::make_shared<ComputationNode>("output", "linear");
        output_node->output_shape = {seq_len, d_model};
        output_node->curvature = 0.0;
        output_node->lipschitz = std::sqrt(static_cast<double>(d_model));
        output_node->diameter = std::sqrt(static_cast<double>(d_model));
        graph.add_node(output_node);
        graph.add_edge("attn_V", "output");
        
        return graph;
    }
    
    // Build graph for feed-forward network
    static ComputationGraph build_ffn_graph(
        int64_t d_model,
        int64_t d_ff
    ) {
        ComputationGraph graph;
        
        // Input
        auto input_node = std::make_shared<ComputationNode>("input", "input");
        input_node->output_shape = {1, d_model};
        input_node->curvature = 0.0;
        input_node->lipschitz = 1.0;
        input_node->diameter = 2.0;
        graph.add_node(input_node);
        
        // First linear layer
        auto linear1_node = std::make_shared<ComputationNode>("linear1", "linear");
        linear1_node->output_shape = {1, d_ff};
        linear1_node->curvature = 0.0;  // Linear
        linear1_node->lipschitz = std::sqrt(static_cast<double>(d_model));
        linear1_node->diameter = 2.0 * std::sqrt(static_cast<double>(d_model));
        graph.add_node(linear1_node);
        graph.add_edge("input", "linear1");
        
        // ReLU activation
        auto relu_node = std::make_shared<ComputationNode>("relu", "relu");
        relu_node->output_shape = {1, d_ff};
        relu_node->curvature = 0.0;  // Piecewise linear
        relu_node->lipschitz = 1.0;
        relu_node->diameter = 2.0 * std::sqrt(static_cast<double>(d_model));
        graph.add_node(relu_node);
        graph.add_edge("linear1", "relu");
        
        // Second linear layer
        auto linear2_node = std::make_shared<ComputationNode>("linear2", "linear");
        linear2_node->output_shape = {1, d_model};
        linear2_node->curvature = 0.0;
        linear2_node->lipschitz = std::sqrt(static_cast<double>(d_ff));
        linear2_node->diameter = 2.0 * std::sqrt(static_cast<double>(d_ff));
        graph.add_node(linear2_node);
        graph.add_edge("relu", "linear2");
        
        return graph;
    }
    
    // Build complete transformer block
    static ComputationGraph build_transformer_block(
        int64_t seq_len,
        int64_t d_model,
        int64_t num_heads,
        int64_t d_ff
    ) {
        ComputationGraph graph;
        
        // Input
        auto input_node = std::make_shared<ComputationNode>("input", "input");
        input_node->output_shape = {seq_len, d_model};
        input_node->curvature = 0.0;
        input_node->lipschitz = 1.0;
        input_node->diameter = 2.0;
        graph.add_node(input_node);
        
        // Build attention subgraph
        auto attn_graph = build_attention_graph(seq_len, d_model, num_heads);
        
        // Merge attention graph
        for (const auto& [name, node] : attn_graph.nodes) {
            auto new_node = std::make_shared<ComputationNode>(*node);
            new_node->name = "attn_" + name;
            graph.add_node(new_node);
        }
        
        for (const auto& edge : attn_graph.edges) {
            graph.add_edge("attn_" + edge.source, "attn_" + edge.target, edge.tolerance);
        }
        
        // Connect input to attention Q, K, V
        graph.add_edge("input", "attn_Q");
        graph.add_edge("input", "attn_K");
        graph.add_edge("input", "attn_V");
        
        // Layer norm after attention
        auto ln1_node = std::make_shared<ComputationNode>("ln1", "layer_norm");
        ln1_node->output_shape = {seq_len, d_model};
        ln1_node->curvature = 10.0;  // Layer norm has moderate curvature
        ln1_node->lipschitz = 2.0;
        ln1_node->diameter = 2.0;
        graph.add_node(ln1_node);
        graph.add_edge("attn_output", "ln1");
        graph.add_edge("input", "ln1");  // Residual connection
        
        // Build FFN subgraph
        auto ffn_graph = build_ffn_graph(d_model, d_ff);
        
        // Merge FFN graph
        for (const auto& [name, node] : ffn_graph.nodes) {
            if (name == "input") continue;  // Skip input
            
            auto new_node = std::make_shared<ComputationNode>(*node);
            new_node->name = "ffn_" + name;
            graph.add_node(new_node);
        }
        
        for (const auto& edge : ffn_graph.edges) {
            if (edge.source == "input") {
                graph.add_edge("ln1", "ffn_" + edge.target, edge.tolerance);
            } else {
                graph.add_edge("ffn_" + edge.source, "ffn_" + edge.target, edge.tolerance);
            }
        }
        
        // Layer norm after FFN
        auto ln2_node = std::make_shared<ComputationNode>("ln2", "layer_norm");
        ln2_node->output_shape = {seq_len, d_model};
        ln2_node->curvature = 10.0;
        ln2_node->lipschitz = 2.0;
        ln2_node->diameter = 2.0;
        graph.add_node(ln2_node);
        graph.add_edge("ffn_linear2", "ln2");
        graph.add_edge("ln1", "ln2");  // Residual connection
        
        return graph;
    }
    
    // Build simple convolutional network
    static ComputationGraph build_convnet_graph(
        int64_t num_layers,
        int64_t channels,
        int64_t height,
        int64_t width
    ) {
        ComputationGraph graph;
        
        // Input
        auto input_node = std::make_shared<ComputationNode>("input", "input");
        input_node->output_shape = {channels, height, width};
        input_node->curvature = 0.0;
        input_node->lipschitz = 1.0;
        input_node->diameter = 2.0;
        graph.add_node(input_node);
        
        std::string prev_name = "input";
        int64_t current_h = height;
        int64_t current_w = width;
        int64_t current_c = channels;
        
        for (int64_t i = 0; i < num_layers; ++i) {
            // Convolution
            std::string conv_name = "conv" + std::to_string(i);
            auto conv_node = std::make_shared<ComputationNode>(conv_name, "conv2d");
            current_c = std::min<int64_t>(current_c * 2, 512L);  // Increase channels
            conv_node->output_shape = {current_c, current_h, current_w};
            conv_node->curvature = 0.0;  // Linear
            conv_node->lipschitz = std::sqrt(static_cast<double>(current_c));
            conv_node->diameter = 2.0 * std::sqrt(static_cast<double>(current_c));
            graph.add_node(conv_node);
            graph.add_edge(prev_name, conv_name);
            
            // ReLU
            std::string relu_name = "relu" + std::to_string(i);
            auto relu_node = std::make_shared<ComputationNode>(relu_name, "relu");
            relu_node->output_shape = {current_c, current_h, current_w};
            relu_node->curvature = 0.0;
            relu_node->lipschitz = 1.0;
            relu_node->diameter = 2.0 * std::sqrt(static_cast<double>(current_c));
            graph.add_node(relu_node);
            graph.add_edge(conv_name, relu_name);
            
            // Max pooling (every other layer)
            if (i % 2 == 1) {
                std::string pool_name = "pool" + std::to_string(i / 2);
                auto pool_node = std::make_shared<ComputationNode>(pool_name, "maxpool");
                current_h /= 2;
                current_w /= 2;
                pool_node->output_shape = {current_c, current_h, current_w};
                pool_node->curvature = 0.0;  // Piecewise linear
                pool_node->lipschitz = 1.0;
                pool_node->diameter = 2.0 * std::sqrt(static_cast<double>(current_c));
                graph.add_node(pool_node);
                graph.add_edge(relu_name, pool_name);
                prev_name = pool_name;
            } else {
                prev_name = relu_name;
            }
        }
        
        // Global average pooling
        auto gap_node = std::make_shared<ComputationNode>("gap", "global_avg_pool");
        gap_node->output_shape = {current_c};
        gap_node->curvature = 0.0;
        gap_node->lipschitz = 1.0 / std::sqrt(static_cast<double>(current_h * current_w));
        gap_node->diameter = 2.0 * std::sqrt(static_cast<double>(current_c));
        graph.add_node(gap_node);
        graph.add_edge(prev_name, "gap");
        
        // Final linear layer
        auto fc_node = std::make_shared<ComputationNode>("fc", "linear");
        fc_node->output_shape = {10};  // 10 classes
        fc_node->curvature = 0.0;
        fc_node->lipschitz = std::sqrt(static_cast<double>(current_c));
        fc_node->diameter = 10.0;
        graph.add_node(fc_node);
        graph.add_edge("gap", "fc");
        
        return graph;
    }
    
    // Build pathological network (from proposal example)
    static ComputationGraph build_pathological_network() {
        ComputationGraph graph;
        
        // Input
        auto input_node = std::make_shared<ComputationNode>("input", "input");
        input_node->output_shape = {128};
        input_node->curvature = 0.0;
        input_node->lipschitz = 1.0;
        input_node->diameter = 2.0;
        graph.add_node(input_node);
        
        // Low precision OK
        auto linear1_node = std::make_shared<ComputationNode>("linear1", "linear");
        linear1_node->output_shape = {64};
        linear1_node->curvature = 0.0;
        linear1_node->lipschitz = std::sqrt(128.0);
        linear1_node->diameter = 2.0 * std::sqrt(128.0);
        graph.add_node(linear1_node);
        graph.add_edge("input", "linear1");
        
        auto relu1_node = std::make_shared<ComputationNode>("relu1", "relu");
        relu1_node->output_shape = {64};
        relu1_node->curvature = 0.0;
        relu1_node->lipschitz = 1.0;
        relu1_node->diameter = 2.0 * std::sqrt(128.0);
        graph.add_node(relu1_node);
        graph.add_edge("linear1", "relu1");
        
        // CRITICAL: Double exponential - MUST be high precision
        // exp(exp(x)) has κ ~ e^(e^x)
        auto exp1_node = std::make_shared<ComputationNode>("exp1", "exp");
        exp1_node->output_shape = {64};
        double x_max = 5.0;  // Assume clipped inputs
        exp1_node->curvature = std::exp(x_max);  // e^5 ≈ 148
        exp1_node->lipschitz = std::exp(x_max);
        exp1_node->diameter = std::exp(x_max);
        graph.add_node(exp1_node);
        graph.add_edge("relu1", "exp1");
        
        auto exp2_node = std::make_shared<ComputationNode>("exp2", "exp");
        exp2_node->output_shape = {64};
        double exp_x_max = std::exp(x_max);
        exp2_node->curvature = std::exp(exp_x_max);  // Huge!
        exp2_node->lipschitz = std::exp(exp_x_max);
        exp2_node->diameter = std::exp(exp_x_max);
        graph.add_node(exp2_node);
        graph.add_edge("exp1", "exp2");
        
        // Low precision OK again
        auto linear2_node = std::make_shared<ComputationNode>("linear2", "linear");
        linear2_node->output_shape = {10};
        linear2_node->curvature = 0.0;
        linear2_node->lipschitz = std::sqrt(64.0);
        linear2_node->diameter = 10.0;
        graph.add_node(linear2_node);
        graph.add_edge("exp2", "linear2");
        
        return graph;
    }
};

} // namespace sheaf
} // namespace hnf
