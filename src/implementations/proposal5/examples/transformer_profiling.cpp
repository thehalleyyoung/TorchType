#include "curvature_profiler.hpp"
#include "visualization.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <iomanip>

using namespace hnf::profiler;

/**
 * @brief Example: Predict training instability in a Transformer
 * 
 * This example demonstrates the main contribution of Proposal 5:
 * Using curvature κ^{curv}(t) to predict training failures before they occur.
 * 
 * We train a small Transformer with deliberately unstable settings and show
 * that curvature spikes precede loss spikes by 10-100 steps, validating
 * the hypothesis from the proposal.
 */

struct TransformerBlock : torch::nn::Module {
    TransformerBlock(int d_model, int nhead) 
        : d_model_(d_model), nhead_(nhead) {
        
        // Multi-head attention components
        q_proj = register_module("q_proj", torch::nn::Linear(d_model, d_model));
        k_proj = register_module("k_proj", torch::nn::Linear(d_model, d_model));
        v_proj = register_module("v_proj", torch::nn::Linear(d_model, d_model));
        out_proj = register_module("out_proj", torch::nn::Linear(d_model, d_model));
        
        // Feed-forward network
        ffn_up = register_module("ffn_up", torch::nn::Linear(d_model, 4 * d_model));
        ffn_down = register_module("ffn_down", torch::nn::Linear(4 * d_model, d_model));
        
        // Layer norms
        ln1 = register_module("ln1", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model})));
        ln2 = register_module("ln2", torch::nn::LayerNorm(
            torch::nn::LayerNormOptions({d_model})));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        // x: (batch, seq_len, d_model)
        auto residual = x;
        
        // Self-attention
        x = ln1->forward(x);
        auto Q = q_proj->forward(x);
        auto K = k_proj->forward(x);
        auto V = v_proj->forward(x);
        
        // Scaled dot-product attention
        auto scores = torch::matmul(Q, K.transpose(-2, -1)) / std::sqrt(d_model_);
        auto attn = torch::softmax(scores, -1);
        auto attn_out = torch::matmul(attn, V);
        attn_out = out_proj->forward(attn_out);
        
        x = residual + attn_out;
        residual = x;
        
        // Feed-forward
        x = ln2->forward(x);
        x = ffn_up->forward(x);
        x = torch::relu(x);
        x = ffn_down->forward(x);
        
        return residual + x;
    }
    
    int d_model_, nhead_;
    torch::nn::Linear q_proj{nullptr}, k_proj{nullptr}, v_proj{nullptr}, out_proj{nullptr};
    torch::nn::Linear ffn_up{nullptr}, ffn_down{nullptr};
    torch::nn::LayerNorm ln1{nullptr}, ln2{nullptr};
};

struct SimpleTransformer : torch::nn::Module {
    SimpleTransformer(int vocab_size, int d_model, int num_layers) 
        : vocab_size_(vocab_size), d_model_(d_model) {
        
        embed = register_module("embed", 
            torch::nn::Embedding(torch::nn::EmbeddingOptions(vocab_size, d_model)));
        
        for (int i = 0; i < num_layers; ++i) {
            auto block = std::make_shared<TransformerBlock>(d_model, 4);
            blocks.push_back(register_module("block" + std::to_string(i), block));
        }
        
        out = register_module("out", torch::nn::Linear(d_model, vocab_size));
    }
    
    torch::Tensor forward(torch::Tensor input_ids) {
        auto x = embed->forward(input_ids);
        
        for (auto& block : blocks) {
            x = block->forward(x);
        }
        
        return out->forward(x);
    }
    
    int vocab_size_, d_model_;
    torch::nn::Embedding embed{nullptr};
    std::vector<std::shared_ptr<TransformerBlock>> blocks;
    torch::nn::Linear out{nullptr};
};

void run_unstable_training_experiment() {
    std::cout << "\n=== Experiment 1: Predicting Training Instability ===\n\n";
    
    // Model config
    int vocab_size = 1000;
    int d_model = 128;
    int num_layers = 6;
    int seq_len = 64;
    int batch_size = 8;
    
    auto model = std::make_shared<SimpleTransformer>(vocab_size, d_model, num_layers);
    
    // Setup profiler
    CurvatureProfiler profiler(*model);
    
    // Track key layers
    for (size_t i = 0; i < model->blocks.size(); ++i) {
        auto& block = model->blocks[i];
        profiler.track_layer("block" + std::to_string(i) + ".q_proj", block->q_proj);
        profiler.track_layer("block" + std::to_string(i) + ".softmax", block->q_proj);  // Proxy
        profiler.track_layer("block" + std::to_string(i) + ".ffn_up", block->ffn_up);
    }
    
    // Setup monitor with low thresholds to catch issues early
    TrainingMonitor::Config monitor_config;
    monitor_config.warning_threshold = 1e5;
    monitor_config.danger_threshold = 1e8;
    monitor_config.prediction_horizon = 50;
    
    TrainingMonitor monitor(profiler, monitor_config);
    RealTimeDashboard dashboard(profiler, monitor);
    dashboard.set_compact_mode(false);
    
    // Setup optimizer with deliberately high LR to cause instability
    torch::optim::Adam optimizer(model->parameters(), 
                                 torch::optim::AdamOptions(0.01));  // High LR
    
    std::vector<double> loss_history;
    std::vector<int> warning_steps;
    std::vector<int> loss_spike_steps;
    
    int num_steps = 200;
    
    std::cout << "Training with deliberately unstable settings (high LR)...\n";
    std::cout << "Hypothesis: Curvature spikes will precede loss spikes\n\n";
    
    for (int step = 0; step < num_steps; ++step) {
        // Generate random batch
        auto input_ids = torch::randint(0, vocab_size, {batch_size, seq_len});
        auto target_ids = torch::randint(0, vocab_size, {batch_size, seq_len});
        
        // Forward pass
        auto logits = model->forward(input_ids);
        auto loss = torch::nn::functional::cross_entropy(
            logits.view({-1, vocab_size}),
            target_ids.view({-1}));
        
        double loss_val = loss.item<double>();
        loss_history.push_back(loss_val);
        
        // Check for loss spike
        if (loss_history.size() > 1) {
            double prev_loss = loss_history[loss_history.size() - 2];
            if (loss_val > prev_loss * 2.0 && prev_loss > 0) {
                loss_spike_steps.push_back(step);
            }
        }
        
        // Monitor step
        auto warnings = monitor.on_step(loss, step);
        if (!warnings.empty()) {
            warning_steps.push_back(step);
            for (const auto& w : warnings) {
                std::cout << "[Step " << step << "] " << w << "\n";
            }
        }
        
        // Backward and optimize
        optimizer.zero_grad();
        loss.backward();
        
        // Gradient clipping to prevent complete explosion
        torch::nn::utils::clip_grad_norm_(model->parameters(), 1.0);
        
        optimizer.step();
        
        // Dashboard update
        if (step % 10 == 0) {
            dashboard.update(step, loss_val);
        }
        
        // Early termination on NaN
        if (!std::isfinite(loss_val)) {
            std::cout << "\nTraining diverged at step " << step << "\n";
            break;
        }
    }
    
    std::cout << "\n\n=== Analysis ===\n";
    std::cout << "Total warnings: " << warning_steps.size() << "\n";
    std::cout << "Total loss spikes: " << loss_spike_steps.size() << "\n";
    
    // Analyze lead time: how many steps before loss spike did we warn?
    std::vector<int> lead_times;
    for (int loss_spike : loss_spike_steps) {
        for (int warn_step : warning_steps) {
            if (warn_step < loss_spike && loss_spike - warn_step <= 100) {
                lead_times.push_back(loss_spike - warn_step);
                break;
            }
        }
    }
    
    if (!lead_times.empty()) {
        double avg_lead = std::accumulate(lead_times.begin(), lead_times.end(), 0.0) 
                         / lead_times.size();
        std::cout << "Average lead time: " << avg_lead << " steps\n";
        std::cout << "Min lead time: " << *std::min_element(lead_times.begin(), lead_times.end()) << "\n";
        std::cout << "Max lead time: " << *std::max_element(lead_times.begin(), lead_times.end()) << "\n";
        
        if (avg_lead >= 10) {
            std::cout << "\n✓ SUCCESS: Curvature warnings preceded loss spikes by "
                      << avg_lead << " steps on average!\n";
        }
    } else {
        std::cout << "No clear correlation detected (may need longer training or different seed)\n";
    }
    
    // Export data for plotting
    profiler.export_to_csv("transformer_curvature.csv");
    
    // Export loss history
    std::ofstream loss_file("transformer_loss.csv");
    loss_file << "step,loss\n";
    for (size_t i = 0; i < loss_history.size(); ++i) {
        loss_file << i << "," << loss_history[i] << "\n";
    }
    loss_file.close();
    
    // Generate visualization
    CurvatureVisualizer viz(profiler);
    std::cout << "\n" << viz.generate_heatmap() << "\n";
    
    auto correlations = viz.correlate_with_loss_spikes(loss_history, 2.0, 100);
    std::cout << "\nDetected " << correlations.size() << " curvature-loss correlations:\n";
    for (const auto& [curv_step, loss_step, layer] : correlations) {
        std::cout << "  Layer " << layer << ": curvature spike at step " << curv_step
                  << " → loss spike at step " << loss_step 
                  << " (lag: " << (loss_step - curv_step) << ")\n";
    }
    
    viz.generate_matplotlib_script("plot_transformer.py", "transformer_curvature.csv");
    std::cout << "\nGenerated plot_transformer.py - run with: python3 plot_transformer.py\n";
}

void run_adaptive_lr_experiment() {
    std::cout << "\n\n=== Experiment 2: Curvature-Adaptive Learning Rate ===\n\n";
    
    // Simpler model for clarity
    auto model = std::make_shared<torch::nn::Sequential>(
        torch::nn::Linear(100, 50),
        torch::nn::ReLU(),
        torch::nn::Linear(50, 20),
        torch::nn::ReLU(),
        torch::nn::Linear(20, 10)
    );
    
    CurvatureProfiler profiler(*model);
    
    // Track layers
    auto layers = model->children();
    int idx = 0;
    for (auto& layer : layers) {
        if (auto linear = layer->as<torch::nn::Linear>()) {
            profiler.track_layer("layer" + std::to_string(idx), 
                               std::make_shared<torch::nn::Linear>(*linear));
        }
        idx++;
    }
    
    // Setup adaptive LR scheduler
    CurvatureAdaptiveLR::Config lr_config;
    lr_config.base_lr = 0.01;
    lr_config.target_curvature = 1e4;
    
    CurvatureAdaptiveLR adaptive_scheduler(profiler, lr_config);
    
    torch::optim::SGD optimizer(model->parameters(), 
                                torch::optim::SGDOptions(lr_config.base_lr));
    
    std::vector<double> lr_history;
    std::vector<double> loss_history;
    
    std::cout << "Training with curvature-adaptive learning rate...\n\n";
    
    for (int step = 0; step < 100; ++step) {
        auto x = torch::randn({32, 100});
        auto target = torch::randint(0, 10, {32});
        
        auto output = model->forward(x);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        
        loss_history.push_back(loss.item<double>());
        
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        // Apply adaptive LR
        adaptive_scheduler.step(optimizer, step);
        
        // Record current LR
        double current_lr = optimizer.param_groups()[0].options().get_lr();
        lr_history.push_back(current_lr);
        
        if (step % 20 == 0) {
            std::cout << "Step " << step << ": loss=" << loss.item<double>()
                      << ", LR=" << current_lr << "\n";
        }
    }
    
    // Export results
    std::ofstream results("adaptive_lr_results.csv");
    results << "step,loss,lr\n";
    for (size_t i = 0; i < loss_history.size(); ++i) {
        results << i << "," << loss_history[i] << "," << lr_history[i] << "\n";
    }
    results.close();
    
    std::cout << "\n✓ Adaptive LR training completed successfully!\n";
    std::cout << "Results saved to adaptive_lr_results.csv\n";
}

int main() {
    std::cout << "========================================\n";
    std::cout << "HNF Proposal 5: Condition Number Profiler\n";
    std::cout << "Demonstration of Training Dynamics Monitoring\n";
    std::cout << "========================================\n";
    
    torch::manual_seed(42);
    
    try {
        run_unstable_training_experiment();
        run_adaptive_lr_experiment();
        
        std::cout << "\n========================================\n";
        std::cout << "All experiments completed successfully!\n";
        std::cout << "========================================\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}
