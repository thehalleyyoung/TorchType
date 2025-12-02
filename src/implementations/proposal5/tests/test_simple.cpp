#include "curvature_profiler.hpp"
#include <torch/torch.h>
#include <iostream>

using namespace hnf::profiler;

int main() {
    std::cout << "Simple test of curvature profiler\n";
    
    // Create a simple linear layer
    auto model = torch::nn::Sequential(
        torch::nn::Linear(10, 5),
        torch::nn::ReLU()
    );
    
    CurvatureProfiler profiler(*model);
    
    // Track the first layer
    auto children = model->children();
    int idx = 0;
    for (auto& child : children) {
        profiler.track_layer("layer" + std::to_string(idx++), child.get());
    }
    
    // Generate some data
    auto x = torch::randn({8, 10});
    auto y = model->forward(x);
    auto loss = y.pow(2).sum();
    
    // Compute curvature
    auto metrics = profiler.compute_curvature(loss, 0);
    
    std::cout << "Computed curvature for " << metrics.size() << " layers\n";
    for (const auto& [name, m] : metrics) {
        std::cout << name << ": kappa=" << m.kappa_curv << "\n";
    }
    
    std::cout << "Test passed!\n";
    return 0;
}
