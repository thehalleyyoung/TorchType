#include "real_training.hpp"
#include "attention_curvature.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>

namespace hnf {
namespace attention {

// TransformerLayer implementation
MNISTTransformer::TransformerLayer::TransformerLayer(
    int dim,
    int num_heads,
    double temperature,
    double dropout_rate
) {
    attention = register_module(
        "attention",
        torch::nn::MultiheadAttention(
            torch::nn::MultiheadAttentionOptions(dim, num_heads)
                .dropout(dropout_rate)
        )
    );
    
    norm1 = register_module("norm1", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    norm2 = register_module("norm2", torch::nn::LayerNorm(torch::nn::LayerNormOptions({dim})));
    
    ffn1 = register_module("ffn1", torch::nn::Linear(dim, dim * 4));
    ffn2 = register_module("ffn2", torch::nn::Linear(dim * 4, dim));
    
    dropout = register_module("dropout", torch::nn::Dropout(dropout_rate));
}

torch::Tensor MNISTTransformer::TransformerLayer::forward(
    torch::Tensor x,
    bool track_curvature
) {
    // Multi-head attention with residual
    auto attn_output_tuple = attention->forward(x, x, x);
    auto attn_output = std::get<0>(attn_output_tuple);
    
    if (track_curvature) {
        last_attention_weights = std::get<1>(attn_output_tuple);
    }
    
    x = norm1(x + dropout(attn_output));
    
    // Feed-forward with residual
    auto ffn_output = ffn2(torch::relu(ffn1(x)));
    x = norm2(x + dropout(ffn_output));
    
    return x;
}

// MNISTTransformer implementation
MNISTTransformer::MNISTTransformer(const Config& config)
    : config_(config)
{
    // Patch embedding: Conv2d extracts patches
    patch_embedding_ = register_module(
        "patch_embedding",
        torch::nn::Conv2d(
            torch::nn::Conv2dOptions(1, config.embedding_dim, config.patch_size)
                .stride(config.patch_size)
        )
    );
    
    // Position embedding
    position_embedding_ = register_module(
        "position_embedding",
        torch::nn::Linear(config.num_patches, config.num_patches)
    );
    
    // Transformer layers
    for (int i = 0; i < config.num_layers; ++i) {
        auto layer = std::make_shared<TransformerLayer>(
            config.embedding_dim,
            config.num_heads,
            config.temperature,
            config.dropout
        );
        layers_.push_back(layer);
        register_module("layer_" + std::to_string(i), layer);
    }
    
    // Final normalization and classifier
    final_norm_ = register_module(
        "final_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({config.embedding_dim}))
    );
    
    classifier_ = register_module(
        "classifier",
        torch::nn::Linear(config.embedding_dim, config.num_classes)
    );
}

torch::Tensor MNISTTransformer::forward(
    torch::Tensor x,
    bool track_curvature
) {
    // x: [batch, 1, 28, 28]
    
    // Patch embedding
    x = patch_embedding_(x);  // [batch, embedding_dim, patch_grid, patch_grid]
    
    // Flatten patches
    int batch_size = x.size(0);
    x = x.flatten(2).transpose(1, 2);  // [batch, num_patches, embedding_dim]
    
    // Add positional encoding (learnable)
    x = x.transpose(0, 1);  // [num_patches, batch, embedding_dim] for attention
    
    // Clear tracking
    if (track_curvature) {
        attention_weights_.clear();
    }
    
    // Transformer layers
    for (auto& layer : layers_) {
        x = layer->forward(x, track_curvature);
        if (track_curvature && layer->last_attention_weights.defined()) {
            attention_weights_.push_back(layer->last_attention_weights);
        }
    }
    
    x = x.transpose(0, 1);  // [batch, num_patches, embedding_dim]
    
    // Global average pooling
    x = x.mean(1);  // [batch, embedding_dim]
    
    // Normalization and classification
    x = final_norm_(x);
    x = classifier_(x);
    
    return x;  // [batch, num_classes]
}

std::vector<torch::Tensor> MNISTTransformer::get_Q_weights() const {
    std::vector<torch::Tensor> weights;
    for (const auto& layer : layers_) {
        // Extract Q projection from MultiheadAttention
        auto q_proj = layer->attention->named_parameters()["in_proj_weight"];
        int embed_dim = config_.embedding_dim;
        weights.push_back(q_proj.slice(0, 0, embed_dim));
    }
    return weights;
}

std::vector<torch::Tensor> MNISTTransformer::get_K_weights() const {
    std::vector<torch::Tensor> weights;
    for (const auto& layer : layers_) {
        auto k_proj = layer->attention->named_parameters()["in_proj_weight"];
        int embed_dim = config_.embedding_dim;
        weights.push_back(k_proj.slice(0, embed_dim, 2 * embed_dim));
    }
    return weights;
}

std::vector<torch::Tensor> MNISTTransformer::get_V_weights() const {
    std::vector<torch::Tensor> weights;
    for (const auto& layer : layers_) {
        auto v_proj = layer->attention->named_parameters()["in_proj_weight"];
        int embed_dim = config_.embedding_dim;
        weights.push_back(v_proj.slice(0, 2 * embed_dim, 3 * embed_dim));
    }
    return weights;
}

std::vector<torch::Tensor> MNISTTransformer::get_ffn_weights() const {
    std::vector<torch::Tensor> weights;
    for (const auto& layer : layers_) {
        weights.push_back(layer->ffn1->weight);
    }
    return weights;
}

// HNFMonitoredTraining implementation
HNFMonitoredTraining::HNFMonitoredTraining(
    const MNISTTransformer::Config& model_config,
    const TrainingConfig& training_config,
    const HardwareModel& hardware
) : model_config_(model_config),
    training_config_(training_config),
    hardware_(hardware)
{
    // Create model
    model_ = std::make_shared<MNISTTransformer>(model_config);
    
    // Create optimizer
    optimizer_ = std::make_shared<torch::optim::Adam>(
        model_->parameters(),
        torch::optim::AdamOptions(training_config.learning_rate)
    );
    
    // Load datasets
    train_dataset_ = std::make_unique<torch::data::datasets::MNIST>(
        training_config.dataset_path,
        torch::data::datasets::MNIST::Mode::kTrain
    );
    
    test_dataset_ = std::make_unique<torch::data::datasets::MNIST>(
        training_config.dataset_path,
        torch::data::datasets::MNIST::Mode::kTest
    );
    
    // Initialize precision analyzer
    precision_analyzer_.build_graph_from_transformer(
        model_config.num_layers,
        model_config.num_heads,
        model_config.embedding_dim,
        model_config.num_patches,
        model_config.temperature
    );
}

HNFMonitoredTraining::PreTrainingAnalysis 
HNFMonitoredTraining::analyze_before_training() {
    PreTrainingAnalysis analysis;
    
    std::cout << "\n=== Pre-Training HNF Analysis ===\n" << std::endl;
    
    // Get initial weights
    auto Q_weights = model_->get_Q_weights();
    auto K_weights = model_->get_K_weights();
    auto V_weights = model_->get_V_weights();
    auto ffn_weights = model_->get_ffn_weights();
    
    // Populate analyzer with weights
    precision_analyzer_.populate_from_weights(Q_weights, K_weights, V_weights, ffn_weights);
    
    // Run sheaf cohomology analysis
    analysis.sheaf_analysis = precision_analyzer_.generate_report(
        training_config_.target_accuracy_precision,
        hardware_
    );
    
    // Extract key metrics
    analysis.predicted_max_curvature = 0.0;
    for (const auto& vertex : precision_analyzer_.graph().vertices()) {
        analysis.predicted_max_curvature = std::max(
            analysis.predicted_max_curvature,
            vertex.local_curvature
        );
    }
    
    analysis.predicted_precision_requirement = analysis.sheaf_analysis.cohomology.minimal_precision;
    
    // Make predictions
    if (!analysis.sheaf_analysis.is_achievable_with_hardware) {
        analysis.will_succeed = false;
        analysis.predictions.push_back(
            "❌ PREDICTION: Training will FAIL due to insufficient precision"
        );
        analysis.predictions.push_back(
            "   Required: " + std::to_string(analysis.predicted_precision_requirement) + " bits"
        );
        analysis.predictions.push_back(
            "   Hardware: " + std::to_string(hardware_.precision_bits()) + " bits"
        );
    } else if (analysis.sheaf_analysis.cohomology.h1_dimension > 0) {
        analysis.will_succeed = false;
        analysis.predictions.push_back(
            "❌ PREDICTION: Training will FAIL due to precision obstruction"
        );
        analysis.predictions.push_back(
            "   H^1 dimension: " + std::to_string(analysis.sheaf_analysis.cohomology.h1_dimension)
        );
        for (const auto& reason : analysis.sheaf_analysis.cohomology.obstruction_reasons) {
            analysis.predictions.push_back("   " + reason);
        }
    } else if (analysis.predicted_max_curvature > 1e10) {
        analysis.will_succeed = false;
        analysis.predictions.push_back(
            "⚠️  PREDICTION: Training may fail due to extreme curvature"
        );
        analysis.predictions.push_back(
            "   Max curvature: " + std::to_string(analysis.predicted_max_curvature)
        );
    } else {
        analysis.will_succeed = true;
        analysis.predictions.push_back(
            "✅ PREDICTION: Training should SUCCEED"
        );
        analysis.predictions.push_back(
            "   Max curvature: " + std::to_string(analysis.predicted_max_curvature) + " (acceptable)"
        );
        analysis.predictions.push_back(
            "   Precision: " + std::to_string(analysis.predicted_precision_requirement) + 
            " bits (within hardware capacity)"
        );
    }
    
    // Print analysis
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Layers: " << model_config_.num_layers << std::endl;
    std::cout << "  Heads: " << model_config_.num_heads << std::endl;
    std::cout << "  Embedding dim: " << model_config_.embedding_dim << std::endl;
    std::cout << "  Temperature: " << model_config_.temperature << std::endl;
    std::cout << "\nHNF Predictions:" << std::endl;
    for (const auto& pred : analysis.predictions) {
        std::cout << pred << std::endl;
    }
    
    std::cout << "\nSheaf Cohomology:" << std::endl;
    std::cout << "  H^0 dimension: " << analysis.sheaf_analysis.cohomology.h0_dimension << std::endl;
    std::cout << "  H^1 dimension: " << analysis.sheaf_analysis.cohomology.h1_dimension << std::endl;
    std::cout << "  Minimal precision: " << analysis.predicted_precision_requirement << " bits" << std::endl;
    
    std::cout << "\nPer-Layer Diagnosis:" << std::endl;
    for (const auto& diagnosis : analysis.sheaf_analysis.layer_diagnoses) {
        std::cout << "  " << diagnosis << std::endl;
    }
    
    if (!analysis.sheaf_analysis.recommendations.empty()) {
        std::cout << "\nRecommendations:" << std::endl;
        for (const auto& rec : analysis.sheaf_analysis.recommendations) {
            std::cout << "  " << rec << std::endl;
        }
    }
    
    std::cout << "\n================================\n" << std::endl;
    
    return analysis;
}

MultiLayerPrecisionAnalyzer::AnalysisReport HNFMonitoredTraining::run_hnf_analysis() {
    auto Q_weights = model_->get_Q_weights();
    auto K_weights = model_->get_K_weights();
    auto V_weights = model_->get_V_weights();
    auto ffn_weights = model_->get_ffn_weights();
    
    precision_analyzer_.populate_from_weights(Q_weights, K_weights, V_weights, ffn_weights);
    
    return precision_analyzer_.generate_report(
        training_config_.target_accuracy_precision,
        hardware_
    );
}

bool HNFMonitoredTraining::apply_intervention(
    const MultiLayerPrecisionAnalyzer::AnalysisReport& report
) {
    if (!training_config_.auto_intervene) {
        return false;
    }
    
    bool intervention_applied = false;
    
    // If precision requirement is close to hardware limit, reduce learning rate
    if (report.cohomology.minimal_precision > hardware_.precision_bits() * 0.9) {
        std::cout << "🔧 INTERVENTION: Reducing learning rate (precision near hardware limit)" << std::endl;
        
        for (auto& param_group : optimizer_->param_groups()) {
            if (param_group.has_options()) {
                auto& options = static_cast<torch::optim::AdamOptions&>(param_group.options());
                options.lr(options.lr() * 0.5);
            }
        }
        intervention_applied = true;
    }
    
    return intervention_applied;
}

std::pair<double, double> HNFMonitoredTraining::train_epoch(
    int epoch,
    TrainingHistory& history
) {
    model_->train();
    
    double total_loss = 0.0;
    double total_correct = 0.0;
    int total_samples = 0;
    int batch_count = 0;
    
    auto data_loader = torch::data::make_data_loader(
        *train_dataset_,
        torch::data::DataLoaderOptions().batch_size(training_config_.batch_size).workers(2)
    );
    
    for (auto& batch_vector : *data_loader) {
        optimizer_->zero_grad();
        
        // Data loader returns vector of Examples
        // Each Example has .data (tensor) and .target (tensor)
        std::vector<torch::Tensor> images_vec, targets_vec;
        for (const auto& example : batch_vector) {
            images_vec.push_back(example.data);
            targets_vec.push_back(example.target);
        }
        
        auto images = torch::stack(images_vec).to(torch::kFloat32) / 255.0;
        auto targets = torch::stack(targets_vec).to(torch::kLong).squeeze();
        
        bool track_curvature = training_config_.enable_hnf_monitoring && 
                              (batch_count % training_config_.monitor_every_n_batches == 0);
        
        auto outputs = model_->forward(images, track_curvature);
        
        auto loss = torch::nn::functional::cross_entropy(outputs, targets);
        loss.backward();
        optimizer_->step();
        
        // Compute accuracy
        auto predictions = outputs.argmax(1);
        auto correct = predictions.eq(targets).sum();
        
        total_loss += loss.item<double>() * images.size(0);
        total_correct += correct.item<double>();
        total_samples += images.size(0);
        batch_count++;
        
        // HNF monitoring
        if (track_curvature) {
            auto report = run_hnf_analysis();
            
            double max_curv = 0.0;
            for (double p : report.per_layer_precision) {
                max_curv = std::max(max_curv, p);
            }
            
            if (report.cohomology.h1_dimension > 0) {
                std::cout << "⚠️  Warning: H^1 obstruction detected at epoch " << epoch 
                         << ", batch " << batch_count << std::endl;
                
                bool intervened = apply_intervention(report);
                if (intervened) {
                    history.interventions.push_back(
                        "Epoch " + std::to_string(epoch) + ", Batch " + std::to_string(batch_count) +
                        ": Applied intervention"
                    );
                }
            }
        }
    }
    
    double avg_loss = total_loss / total_samples;
    double accuracy = total_correct / total_samples;
    
    return {avg_loss, accuracy};
}

std::pair<double, double> HNFMonitoredTraining::evaluate() {
    model_->eval();
    torch::NoGradGuard no_grad;
    
    double total_loss = 0.0;
    double total_correct = 0.0;
    int total_samples = 0;
    
    auto data_loader = torch::data::make_data_loader(
        *test_dataset_,
        torch::data::DataLoaderOptions().batch_size(training_config_.batch_size).workers(2)
    );
    
    for (auto& batch_vector : *data_loader) {
        // Data loader returns vector of Examples
        std::vector<torch::Tensor> images_vec, targets_vec;
        for (const auto& example : batch_vector) {
            images_vec.push_back(example.data);
            targets_vec.push_back(example.target);
        }
        
        auto images = torch::stack(images_vec).to(torch::kFloat32) / 255.0;
        auto targets = torch::stack(targets_vec).to(torch::kLong).squeeze();
        
        auto outputs = model_->forward(images, false);
        auto loss = torch::nn::functional::cross_entropy(outputs, targets);
        
        auto predictions = outputs.argmax(1);
        auto correct = predictions.eq(targets).sum();
        
        total_loss += loss.item<double>() * images.size(0);
        total_correct += correct.item<double>();
        total_samples += images.size(0);
    }
    
    double avg_loss = total_loss / total_samples;
    double accuracy = total_correct / total_samples;
    
    return {avg_loss, accuracy};
}

HNFMonitoredTraining::TrainingHistory HNFMonitoredTraining::train() {
    TrainingHistory history;
    history.training_succeeded = true;
    
    // Pre-training analysis
    auto pre_analysis = analyze_before_training();
    
    if (!pre_analysis.will_succeed && !training_config_.auto_intervene) {
        history.training_succeeded = false;
        history.failure_reason = "HNF pre-training analysis predicted failure";
        return history;
    }
    
    std::cout << "Starting training..." << std::endl;
    
    for (int epoch = 0; epoch < training_config_.num_epochs; ++epoch) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << training_config_.num_epochs << std::endl;
        
        // Train
        auto [train_loss, train_acc] = train_epoch(epoch, history);
        history.train_losses.push_back(train_loss);
        history.train_accuracies.push_back(train_acc);
        
        // Evaluate
        auto [test_loss, test_acc] = evaluate();
        history.test_losses.push_back(test_loss);
        history.test_accuracies.push_back(test_acc);
        
        std::cout << std::fixed << std::setprecision(4);
        std::cout << "  Train Loss: " << train_loss << ", Train Acc: " << train_acc << std::endl;
        std::cout << "  Test Loss: " << test_loss << ", Test Acc: " << test_acc << std::endl;
        
        // HNF analysis at end of epoch
        if (training_config_.enable_hnf_monitoring) {
            auto report = run_hnf_analysis();
            
            double max_curv = 0.0;
            for (const auto& vertex : precision_analyzer_.graph().vertices()) {
                max_curv = std::max(max_curv, vertex.local_curvature);
            }
            
            history.max_curvatures.push_back(max_curv);
            history.required_precisions.push_back(report.cohomology.minimal_precision);
            history.h1_dimensions.push_back(report.cohomology.h1_dimension);
            
            std::cout << "  HNF Metrics: Curvature=" << max_curv 
                     << ", Precision=" << report.cohomology.minimal_precision << " bits"
                     << ", H^1=" << report.cohomology.h1_dimension << std::endl;
            
            // Check for training failure
            if (std::isnan(train_loss) || std::isinf(train_loss)) {
                history.training_succeeded = false;
                history.failure_reason = "NaN/Inf in training loss at epoch " + std::to_string(epoch);
                std::cout << "❌ Training failed: " << history.failure_reason << std::endl;
                break;
            }
        }
    }
    
    return history;
}

std::vector<HNFMonitoredTraining::ConfigComparison> 
HNFMonitoredTraining::compare_configurations(
    const std::vector<MNISTTransformer::Config>& configs,
    const HardwareModel& hardware
) {
    std::vector<ConfigComparison> comparisons;
    
    std::cout << "\n=== Comparing " << configs.size() << " Configurations ===" << std::endl;
    
    for (size_t i = 0; i < configs.size(); ++i) {
        const auto& config = configs[i];
        ConfigComparison comp;
        comp.config = config;
        
        std::cout << "\nConfig " << (i+1) << ":"  << std::endl;
        std::cout << "  Layers=" << config.num_layers 
                 << ", Heads=" << config.num_heads
                 << ", Dim=" << config.embedding_dim
                 << ", Temp=" << config.temperature << std::endl;
        
        // Create temporary model and analyzer
        auto model = std::make_shared<MNISTTransformer>(config);
        
        MultiLayerPrecisionAnalyzer analyzer;
        analyzer.build_graph_from_transformer(
            config.num_layers,
            config.num_heads,
            config.embedding_dim,
            config.num_patches,
            config.temperature
        );
        
        auto Q_weights = model->get_Q_weights();
        auto K_weights = model->get_K_weights();
        auto V_weights = model->get_V_weights();
        auto ffn_weights = model->get_ffn_weights();
        
        analyzer.populate_from_weights(Q_weights, K_weights, V_weights, ffn_weights);
        
        auto report = analyzer.generate_report(1e-6, hardware);
        
        // Extract metrics
        comp.max_curvature = 0.0;
        for (const auto& vertex : analyzer.graph().vertices()) {
            comp.max_curvature = std::max(comp.max_curvature, vertex.local_curvature);
        }
        
        comp.required_precision = report.cohomology.minimal_precision;
        comp.is_viable = report.is_achievable_with_hardware && 
                        report.cohomology.h1_dimension == 0;
        
        // Stability score (higher is better)
        comp.stability_score = 1.0 / (1.0 + std::log10(std::max(comp.max_curvature, 1.0)));
        comp.stability_score *= (hardware.precision_bits() - comp.required_precision) / hardware.precision_bits();
        comp.stability_score = std::max(0.0, comp.stability_score);
        
        comp.issues = report.recommendations;
        
        std::cout << "  Max Curvature: " << comp.max_curvature << std::endl;
        std::cout << "  Required Precision: " << comp.required_precision << " bits" << std::endl;
        std::cout << "  Stability Score: " << comp.stability_score << std::endl;
        std::cout << "  Viable: " << (comp.is_viable ? "✅ Yes" : "❌ No") << std::endl;
        
        if (!comp.issues.empty()) {
            std::cout << "  Issues:" << std::endl;
            for (const auto& issue : comp.issues) {
                std::cout << "    - " << issue << std::endl;
            }
        }
        
        comparisons.push_back(comp);
    }
    
    // Sort by stability score
    std::sort(comparisons.begin(), comparisons.end(),
             [](const ConfigComparison& a, const ConfigComparison& b) {
                 return a.stability_score > b.stability_score;
             });
    
    std::cout << "\n=== Ranking ===" << std::endl;
    for (size_t i = 0; i < comparisons.size(); ++i) {
        std::cout << (i+1) << ". Score=" << comparisons[i].stability_score
                 << " (Layers=" << comparisons[i].config.num_layers
                 << ", Heads=" << comparisons[i].config.num_heads
                 << ", Temp=" << comparisons[i].config.temperature << ")" << std::endl;
    }
    
    return comparisons;
}

} // namespace attention
} // namespace hnf
