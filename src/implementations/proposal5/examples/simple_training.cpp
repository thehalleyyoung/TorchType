#include "curvature_profiler.hpp"
#include "visualization.hpp"
#include <torch/torch.h>
#include <iostream>

using namespace hnf::profiler;

int main() {
    std::cout << "========================================\n";
    std::cout << "HNF Proposal 5: Condition Number Profiler\n";
    std::cout << "Simple Training Example\n";
    std::cout << "========================================\n\n";
    
    torch::manual_seed(42);
    
    // Create a simple feedforward network
    auto model = torch::nn::Sequential(
        torch::nn::Linear(100, 50),
        torch::nn::ReLU(),
        torch::nn::Linear(50, 20),
        torch::nn::ReLU(),
        torch::nn::Linear(20, 10)
    );
    
    // Setup profiler
    CurvatureProfiler profiler(*model);
    
    // Track all linear layers
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        if (auto* linear = dynamic_cast<torch::nn::LinearImpl*>(child.get())) {
            profiler.track_layer("layer" + std::to_string(idx), child.get());
        }
        idx++;
    }
    
    // Setup monitor and dashboard
    TrainingMonitor::Config monitor_config;
    monitor_config.warning_threshold = 1e5;
    monitor_config.danger_threshold = 1e8;
    
    TrainingMonitor monitor(profiler, monitor_config);
    RealTimeDashboard dashboard(profiler, monitor);
    dashboard.set_compact_mode(true);
    
    // Setup optimizer
    torch::optim::Adam optimizer(model->parameters(), torch::optim::AdamOptions(0.001));
    
    std::vector<double> loss_history;
    
    std::cout << "Training for 100 steps...\n\n";
    
    for (int step = 0; step < 100; ++step) {
        // Generate synthetic data
        auto x = torch::randn({32, 100});
        auto target = torch::randint(0, 10, {32});
        
        // Forward pass
        auto output = model->forward(x);
        auto loss = torch::nn::functional::cross_entropy(output, target);
        
        loss_history.push_back(loss.item<double>());
        
        // Monitor and warn
        auto warnings = monitor.on_step(loss, step);
        for (const auto& w : warnings) {
            std::cout << "\n[Step " << step << "] " << w;
        }
        
        // Backward and optimize
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        
        // Update dashboard
        dashboard.update(step, loss.item<double>());
    }
    
    std::cout << "\n\n=== Training completed! ===\n\n";
    
    // Export results
    profiler.export_to_csv("training_curvature.csv");
    std::cout << "Exported curvature data to training_curvature.csv\n";
    
    // Generate visualization
    CurvatureVisualizer viz(profiler);
    std::cout << viz.generate_heatmap() << "\n";
    std::cout << viz.generate_summary_report() << "\n";
    
    viz.generate_matplotlib_script("plot_training.py", "training_curvature.csv");
    std::cout << "Generated plot_training.py - run with: python3 plot_training.py\n";
    
    std::cout << "\n========================================\n";
    std::cout << "Example completed successfully!\n";
    std::cout << "========================================\n";
    
    return 0;
}
