#include "transformer_analyzer.hpp"
#include "patterns.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <sstream>

namespace hnf {
namespace stability_linter {

TransformerAnalyzer::TransformerAnalyzer(const AttentionConfig& config)
    : config_(config) {}

std::shared_ptr<ComputationGraph> TransformerAnalyzer::build_attention_graph() {
    auto graph = std::make_shared<ComputationGraph>();
    
    // Input: Q, K, V matrices
    auto q_node = std::make_shared<Node>("Q", OpType::PLACEHOLDER);
    auto k_node = std::make_shared<Node>("K", OpType::PLACEHOLDER);
    auto v_node = std::make_shared<Node>("V", OpType::PLACEHOLDER);
    
    q_node->value_range = {-10.0, 10.0};  // typical after embedding
    k_node->value_range = {-10.0, 10.0};
    v_node->value_range = {-10.0, 10.0};
    
    graph->add_node(q_node);
    graph->add_node(k_node);
    graph->add_node(v_node);
    
    // K transpose
    auto kt_node = std::make_shared<Node>("K_transpose", OpType::TRANSPOSE);
    kt_node->input_ids = {"K"};
    kt_node->lipschitz_constant = 1.0;
    graph->add_node(kt_node);
    graph->add_edge("K", "K_transpose");
    
    // Q @ K^T
    auto qk_node = std::make_shared<Node>("QK_scores", OpType::MATMUL);
    qk_node->input_ids = {"Q", "K_transpose"};
    // Lipschitz constant for matmul is ||K||_op * ||Q||_op
    // For attention, typically bounded by sqrt(d_k) after initialization
    qk_node->lipschitz_constant = std::sqrt(static_cast<double>(config_.d_k));
    graph->add_node(qk_node);
    graph->add_edge("Q", "QK_scores");
    graph->add_edge("K_transpose", "QK_scores");
    
    // Scale by 1/sqrt(d_k) if enabled
    if (config_.scaled) {
        auto scale_node = std::make_shared<Node>("scale", OpType::DIV);
        scale_node->input_ids = {"QK_scores"};
        scale_node->lipschitz_constant = 1.0 / std::sqrt(static_cast<double>(config_.d_k));
        scale_node->kwargs["divisor"] = std::to_string(std::sqrt(static_cast<double>(config_.d_k)));
        graph->add_node(scale_node);
        graph->add_edge("QK_scores", "scale");
        
        // Softmax on scaled scores
        auto softmax_node = std::make_shared<Node>("attention_weights", OpType::SOFTMAX);
        softmax_node->input_ids = {"scale"};
        softmax_node->lipschitz_constant = 1.0;  // softmax is 1-Lipschitz
        graph->add_node(softmax_node);
        graph->add_edge("scale", "attention_weights");
    } else {
        // Softmax directly on QK
        auto softmax_node = std::make_shared<Node>("attention_weights", OpType::SOFTMAX);
        softmax_node->input_ids = {"QK_scores"};
        softmax_node->lipschitz_constant = 1.0;
        graph->add_node(softmax_node);
        graph->add_edge("QK_scores", "attention_weights");
    }
    
    // Attention @ V
    auto attn_v_node = std::make_shared<Node>("attention_output", OpType::MATMUL);
    attn_v_node->input_ids = {"attention_weights", "V"};
    attn_v_node->lipschitz_constant = 1.0;  // attention weights sum to 1
    graph->add_node(attn_v_node);
    graph->add_edge("attention_weights", "attention_output");
    graph->add_edge("V", "attention_output");
    
    // Propagate ranges
    graph->propagate_ranges({-10.0, 10.0});
    
    return graph;
}

std::shared_ptr<ComputationGraph> TransformerAnalyzer::build_ffn_graph(int hidden_dim) {
    auto graph = std::make_shared<ComputationGraph>();
    
    // Input from attention
    auto input_node = std::make_shared<Node>("ffn_input", OpType::PLACEHOLDER);
    input_node->value_range = {-10.0, 10.0};
    graph->add_node(input_node);
    
    // First linear layer
    auto fc1_node = std::make_shared<Node>("fc1", OpType::MATMUL);
    fc1_node->input_ids = {"ffn_input"};
    // Weight matrix typically Xavier/He initialized
    fc1_node->lipschitz_constant = std::sqrt(static_cast<double>(hidden_dim) / config_.d_model);
    graph->add_node(fc1_node);
    graph->add_edge("ffn_input", "fc1");
    
    // ReLU activation
    auto relu_node = std::make_shared<Node>("relu", OpType::RELU);
    relu_node->input_ids = {"fc1"};
    relu_node->lipschitz_constant = 1.0;
    relu_node->curvature = 0.0;  // ReLU is piecewise linear
    graph->add_node(relu_node);
    graph->add_edge("fc1", "relu");
    
    // Second linear layer
    auto fc2_node = std::make_shared<Node>("fc2", OpType::MATMUL);
    fc2_node->input_ids = {"relu"};
    fc2_node->lipschitz_constant = std::sqrt(static_cast<double>(config_.d_model) / hidden_dim);
    graph->add_node(fc2_node);
    graph->add_edge("relu", "fc2");
    
    graph->propagate_ranges({-10.0, 10.0});
    
    return graph;
}

double TransformerAnalyzer::compute_attention_curvature() {
    // From HNF paper Example 4 (Section 2, Gallery)
    // Attention softmax has curvature κ_softmax = 1/2
    // But composition with QK^T matters:
    // κ_attn = κ_softmax * L_QK^2 + L_softmax * κ_QK
    
    double L_QK = std::sqrt(static_cast<double>(config_.d_k));
    double kappa_softmax = 0.5;  // HNF Theorem
    double L_softmax = 1.0;
    double kappa_QK = 0.0;  // matmul is bilinear, zero curvature
    
    // For unscaled attention
    if (!config_.scaled) {
        // Without scaling, scores can be very large
        L_QK = L_QK * config_.d_k;  // no division by sqrt(d_k)
    }
    
    double kappa_attn = kappa_softmax * L_QK * L_QK + L_softmax * kappa_QK;
    
    return kappa_attn;
}

double TransformerAnalyzer::estimate_layer_lipschitz(const std::string& layer_type) {
    if (layer_type == "attention") {
        // Attention is approximately 1-Lipschitz when properly normalized
        return 1.1;  // slight buffer for numerical errors
    } else if (layer_type == "ffn") {
        // FFN Lipschitz = product of layer Lipschitz constants
        // With residual connections, bounded by ||W2|| * ||W1||
        // Typical values after initialization: ~2-4
        return 3.0;
    } else if (layer_type == "layernorm") {
        return 1.0;  // LayerNorm is 1-Lipschitz
    }
    return 1.0;
}

TransformerAnalyzer::AnalysisResult TransformerAnalyzer::analyze_attention_block() {
    AnalysisResult result;
    
    auto graph = build_attention_graph();
    
    // Compute curvatures for key operations
    result.layer_curvatures["attention_softmax"] = compute_attention_curvature();
    result.layer_curvatures["qk_matmul"] = 0.0;  // bilinear
    result.layer_curvatures["attn_v_matmul"] = 0.0;
    
    // Compute precision requirements from HNF Theorem 4.3
    // p >= log2(c * κ * D^2 / ε)
    double c = 0.125;  // from HNF proof
    double D = 20.0;   // diameter of typical value range [-10, 10]
    double target_eps = 1e-3;  // typical inference accuracy
    
    double kappa_attn = result.layer_curvatures["attention_softmax"];
    int p_attn = static_cast<int>(std::ceil(
        std::log2(c * kappa_attn * D * D / target_eps)
    ));
    
    result.precision_requirements["attention_weights"] = p_attn;
    result.precision_requirements["qk_scores"] = p_attn - 5;  // less critical
    result.precision_requirements["output"] = p_attn;
    
    // Determine minimum safe precision
    result.min_safe_precision = p_attn;
    
    // Generate recommendations
    if (p_attn > 23) {  // exceeds fp16
        result.recommendations["attention_weights"] = 
            "Requires FP32 (need " + std::to_string(p_attn) + " bits, fp16 has only 10 mantissa bits)";
    } else {
        result.recommendations["attention_weights"] = "Can use FP16 safely";
    }
    
    if (!config_.scaled) {
        result.recommendations["scaling"] = 
            "WARNING: Unscaled attention has curvature " + 
            std::to_string(kappa_attn) + " which is " + 
            std::to_string(kappa_attn / compute_attention_curvature()) + 
            "x worse than scaled. MUST use scaled dot-product!";
    }
    
    // Run pattern-based linting
    NumericalLinter linter;
    auto lint_report = linter.lint_graph(graph, 1e-3);
    result.issues = lint_report.results;
    
    return result;
}

TransformerAnalyzer::AnalysisResult TransformerAnalyzer::analyze_transformer_layer(int ffn_hidden_dim) {
    AnalysisResult result;
    
    // Analyze attention
    auto attn_result = analyze_attention_block();
    
    // Analyze FFN
    auto ffn_graph = build_ffn_graph(ffn_hidden_dim);
    NumericalLinter linter;
    auto ffn_lint = linter.lint_graph(ffn_graph, 1e-3);
    
    // Merge results
    result.layer_curvatures = attn_result.layer_curvatures;
    result.layer_curvatures["ffn_fc1"] = 0.0;  // linear
    result.layer_curvatures["ffn_relu"] = 0.0;  // piecewise linear
    result.layer_curvatures["ffn_fc2"] = 0.0;
    
    // Composition curvature (HNF Theorem 3.2)
    // κ_{g∘f} ≤ κ_g * L_f^2 + L_g * κ_f
    double L_attn = estimate_layer_lipschitz("attention");
    double L_ffn = estimate_layer_lipschitz("ffn");
    double kappa_attn = attn_result.layer_curvatures["attention_softmax"];
    double kappa_ffn = 0.0;  // FFN is piecewise linear
    
    result.total_composition_curvature = 
        kappa_ffn * L_attn * L_attn + L_ffn * kappa_attn;
    
    // Precision requirements
    double c = 0.125;
    double D = 20.0;
    double target_eps = 1e-3;
    
    int p_layer = static_cast<int>(std::ceil(
        std::log2(c * result.total_composition_curvature * D * D / target_eps)
    ));
    
    result.precision_requirements["full_layer"] = p_layer;
    result.min_safe_precision = p_layer;
    
    // Merge issues
    result.issues = attn_result.issues;
    result.issues.insert(result.issues.end(), ffn_lint.results.begin(), ffn_lint.results.end());
    
    // Recommendations
    result.recommendations = attn_result.recommendations;
    if (p_layer > 23) {
        result.recommendations["layer_norm"] = "Use FP32 for LayerNorm to maintain accuracy";
    }
    
    return result;
}

TransformerAnalyzer::AnalysisResult TransformerAnalyzer::analyze_stacked_transformer(int num_layers) {
    AnalysisResult result;
    
    // Analyze single layer first
    auto single_layer = analyze_transformer_layer(4 * config_.d_model);
    
    // Composition through stack
    // Error amplification: Π L_i for i=1..num_layers
    double L_per_layer = estimate_layer_lipschitz("attention") * 
                        estimate_layer_lipschitz("ffn");
    
    double total_lipschitz = std::pow(L_per_layer, num_layers);
    
    // Curvature composition (conservative bound)
    // Sum of individual curvatures weighted by downstream Lipschitz products
    double total_curvature = 0.0;
    for (int i = 0; i < num_layers; ++i) {
        double downstream_lipschitz = std::pow(L_per_layer, num_layers - i - 1);
        total_curvature += single_layer.total_composition_curvature * downstream_lipschitz;
    }
    
    result.total_composition_curvature = total_curvature;
    
    // Precision requirements
    double c = 0.125;
    double D = 20.0;
    double target_eps = 1e-3;
    
    int p_total = static_cast<int>(std::ceil(
        std::log2(c * total_curvature * D * D / target_eps)
    ));
    
    result.min_safe_precision = p_total;
    
    // Layer-specific recommendations
    for (int i = 0; i < num_layers; ++i) {
        std::string layer_name = "layer_" + std::to_string(i);
        
        // Early layers more critical (more downstream amplification)
        double layer_weight = std::pow(L_per_layer, num_layers - i - 1);
        int layer_precision = static_cast<int>(std::ceil(
            std::log2(c * single_layer.total_composition_curvature * layer_weight * D * D / target_eps)
        ));
        
        result.precision_requirements[layer_name] = layer_precision;
        
        if (i < num_layers / 3) {
            result.recommendations[layer_name] = "Critical early layer - use FP32";
        } else if (i < 2 * num_layers / 3) {
            result.recommendations[layer_name] = "Middle layer - FP16 acceptable";
        } else {
            result.recommendations[layer_name] = "Late layer - can use lower precision";
        }
    }
    
    // Overall summary
    std::stringstream summary;
    summary << "Stack of " << num_layers << " transformer layers:\n";
    summary << "  Total Lipschitz constant: " << total_lipschitz << "\n";
    summary << "  Total composition curvature: " << total_curvature << "\n";
    summary << "  Minimum safe precision: " << p_total << " mantissa bits\n";
    summary << "  Error amplification: " << std::pow(L_per_layer, num_layers) << "x\n";
    
    result.recommendations["summary"] = summary.str();
    
    return result;
}

TransformerAnalyzer::QuantizationAnalysis TransformerAnalyzer::analyze_quantization_safety(double target_accuracy) {
    QuantizationAnalysis analysis;
    
    // Analyze full layer
    auto layer_result = analyze_transformer_layer(4 * config_.d_model);
    
    // INT8 has ~7 bits of precision (8 bits, 1 for sign)
    // FP16 has 10 mantissa bits
    // FP32 has 23 mantissa bits
    
    for (const auto& [component, required_bits] : layer_result.precision_requirements) {
        analysis.can_use_int8[component] = (required_bits <= 7);
        analysis.can_use_fp16[component] = (required_bits <= 10);
        analysis.requires_fp32[component] = (required_bits > 10);
    }
    
    // Estimate accuracy degradation
    double worst_case_degradation = 0.0;
    for (const auto& [component, required_bits] : layer_result.precision_requirements) {
        if (required_bits > 10) {
            // If we use FP16 but need FP32, estimate degradation
            double bit_shortfall = required_bits - 10;
            worst_case_degradation = std::max(worst_case_degradation, 
                                             std::pow(2.0, -10 + bit_shortfall));
        }
    }
    
    analysis.accuracy_degradation_estimate = worst_case_degradation;
    
    // Generate summary
    std::stringstream summary;
    summary << "Quantization Analysis for target accuracy " << target_accuracy << ":\n\n";
    
    int int8_safe = 0, fp16_safe = 0, fp32_needed = 0;
    for (const auto& [comp, _] : layer_result.precision_requirements) {
        if (analysis.can_use_int8[comp]) int8_safe++;
        else if (analysis.can_use_fp16[comp]) fp16_safe++;
        else fp32_needed++;
    }
    
    summary << "Component Precision Requirements:\n";
    summary << "  " << int8_safe << " components can use INT8\n";
    summary << "  " << fp16_safe << " components can use FP16\n";
    summary << "  " << fp32_needed << " components require FP32\n\n";
    
    summary << "Critical Components (require FP32):\n";
    for (const auto& [comp, req] : analysis.requires_fp32) {
        if (req) {
            summary << "  - " << comp << " (needs " 
                   << layer_result.precision_requirements[comp] << " bits)\n";
        }
    }
    
    if (worst_case_degradation > target_accuracy) {
        summary << "\n⚠️ WARNING: Quantizing to FP16 may degrade accuracy by up to " 
               << worst_case_degradation << "\n";
        summary << "   This exceeds target accuracy " << target_accuracy << "\n";
    } else {
        summary << "\n✓ Mixed precision FP32/FP16 quantization is safe for target accuracy\n";
    }
    
    analysis.summary = summary.str();
    
    return analysis;
}

// ModelVariantAnalyzer implementations

TransformerAnalyzer::ModelSpec ModelVariantAnalyzer::get_bert_base() {
    ModelSpec spec;
    spec.type = ModelType::BERT;
    spec.num_layers = 12;
    spec.d_model = 768;
    spec.n_heads = 12;
    spec.ffn_hidden_dim = 3072;
    spec.vocab_size = 30522;
    spec.max_seq_len = 512;
    return spec;
}

TransformerAnalyzer::ModelSpec ModelVariantAnalyzer::get_gpt2_small() {
    ModelSpec spec;
    spec.type = ModelType::GPT2;
    spec.num_layers = 12;
    spec.d_model = 768;
    spec.n_heads = 12;
    spec.ffn_hidden_dim = 3072;
    spec.vocab_size = 50257;
    spec.max_seq_len = 1024;
    return spec;
}

TransformerAnalyzer::ModelSpec ModelVariantAnalyzer::get_llama2_7b() {
    ModelSpec spec;
    spec.type = ModelType::LLAMA;
    spec.num_layers = 32;
    spec.d_model = 4096;
    spec.n_heads = 32;
    spec.ffn_hidden_dim = 11008;
    spec.vocab_size = 32000;
    spec.max_seq_len = 4096;
    return spec;
}

TransformerAnalyzer::ModelSpec ModelVariantAnalyzer::get_vit_base() {
    ModelSpec spec;
    spec.type = ModelType::VISION_TRANSFORMER;
    spec.num_layers = 12;
    spec.d_model = 768;
    spec.n_heads = 12;
    spec.ffn_hidden_dim = 3072;
    spec.vocab_size = 0;  // no vocab for vision
    spec.max_seq_len = 197;  // 14x14 patches + 1 cls token
    return spec;
}

TransformerAnalyzer::AnalysisResult ModelVariantAnalyzer::analyze_model(const ModelSpec& spec) {
    TransformerAnalyzer::AttentionConfig config;
    config.d_model = spec.d_model;
    config.n_heads = spec.n_heads;
    config.d_k = spec.d_model / spec.n_heads;
    config.d_v = spec.d_model / spec.n_heads;
    config.seq_len = spec.max_seq_len;
    config.scaled = true;
    
    TransformerAnalyzer analyzer(config);
    return analyzer.analyze_stacked_transformer(spec.num_layers);
}

std::vector<std::string> ModelVariantAnalyzer::identify_critical_layers(const ModelSpec& spec) {
    std::vector<std::string> critical;
    
    // First 1/3 of layers are critical
    int critical_threshold = spec.num_layers / 3;
    
    for (int i = 0; i < critical_threshold; ++i) {
        critical.push_back("layer_" + std::to_string(i));
        critical.push_back("layer_" + std::to_string(i) + ".attention");
        critical.push_back("layer_" + std::to_string(i) + ".layernorm");
    }
    
    // Embedding layer always critical
    critical.push_back("embedding");
    
    // Output layer always critical
    critical.push_back("output_layer");
    critical.push_back("lm_head");
    
    return critical;
}

} // namespace stability_linter
} // namespace hnf
