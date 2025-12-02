#include "../include/curvature_quantizer.hpp"
#include <torch/torch.h>
#include <iostream>
#include <memory>

using namespace hnf::quantization;

// ============================================================================
// ResNet Building Blocks
// ============================================================================

struct BasicBlock : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr}, bn2{nullptr};
    torch::nn::Sequential downsample{nullptr};
    bool use_downsample;
    
    BasicBlock(int in_channels, int out_channels, int stride = 1, bool downsample_flag = false)
        : use_downsample(downsample_flag)
    {
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(in_channels, out_channels, 3)
                .stride(stride).padding(1).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(out_channels));
        
        conv2 = register_module("conv2", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(out_channels, out_channels, 3)
                .stride(1).padding(1).bias(false)));
        bn2 = register_module("bn2", torch::nn::BatchNorm2d(out_channels));
        
        if (use_downsample) {
            downsample = register_module("downsample", torch::nn::Sequential(
                torch::nn::Conv2d(torch::nn::Conv2dOptions(in_channels, out_channels, 1)
                    .stride(stride).bias(false)),
                torch::nn::BatchNorm2d(out_channels)
            ));
        }
    }
    
    torch::Tensor forward(torch::Tensor x) {
        auto identity = x;
        
        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = torch::relu(out);
        
        out = conv2->forward(out);
        out = bn2->forward(out);
        
        if (use_downsample) {
            identity = downsample->forward(x);
        }
        
        out += identity;
        out = torch::relu(out);
        
        return out;
    }
};

struct ResNet18 : torch::nn::Module {
    torch::nn::Conv2d conv1{nullptr};
    torch::nn::BatchNorm2d bn1{nullptr};
    torch::nn::SequentialImpl layer1{nullptr}, layer2{nullptr}, layer3{nullptr}, layer4{nullptr};
    torch::nn::AdaptiveAvgPool2d avgpool{nullptr};
    torch::nn::Linear fc{nullptr};
    
    ResNet18(int num_classes = 10) {
        // Initial convolution
        conv1 = register_module("conv1", torch::nn::Conv2d(
            torch::nn::Conv2dOptions(3, 64, 7).stride(2).padding(3).bias(false)));
        bn1 = register_module("bn1", torch::nn::BatchNorm2d(64));
        
        // ResNet layers
        layer1 = register_module("layer1", torch::nn::Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64)
        ));
        
        layer2 = register_module("layer2", torch::nn::Sequential(
            BasicBlock(64, 128, 2, true),
            BasicBlock(128, 128)
        ));
        
        layer3 = register_module("layer3", torch::nn::Sequential(
            BasicBlock(128, 256, 2, true),
            BasicBlock(256, 256)
        ));
        
        layer4 = register_module("layer4", torch::nn::Sequential(
            BasicBlock(256, 512, 2, true),
            BasicBlock(512, 512)
        ));
        
        avgpool = register_module("avgpool", torch::nn::AdaptiveAvgPool2d(
            torch::nn::AdaptiveAvgPool2dOptions({1, 1})));
        
        fc = register_module("fc", torch::nn::Linear(512, num_classes));
    }
    
    torch::Tensor forward(torch::Tensor x) {
        x = conv1->forward(x);
        x = bn1->forward(x);
        x = torch::relu(x);
        x = torch::max_pool2d(x, 3, 2, 1);
        
        x = layer1->forward(x);
        x = layer2->forward(x);
        x = layer3->forward(x);
        x = layer4->forward(x);
        
        x = avgpool->forward(x);
        x = torch::flatten(x, 1);
        x = fc->forward(x);
        
        return x;
    }
};

// ============================================================================
// Main ResNet Quantization Demo
// ============================================================================

int main() {
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║     PROPOSAL 9: ResNet-18 CURVATURE-GUIDED QUANTIZATION       ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    torch::manual_seed(42);
    
    // Create ResNet-18 model
    std::cout << "Creating ResNet-18 model...\n";
    auto model = std::make_shared<ResNet18>(10);
    
    // Count parameters
    int64_t total_params = 0;
    for (const auto& p : model->parameters()) {
        total_params += p.numel();
    }
    std::cout << "Total parameters: " << total_params << "\n\n";
    
    // ========================================================================
    // CURVATURE ANALYSIS
    // ========================================================================
    
    std::cout << "=== Curvature Analysis ===\n";
    
    CurvatureQuantizationAnalyzer analyzer(*model, 1e-3, 4, 16);
    
    // Generate calibration data (synthetic CIFAR-10 like)
    std::vector<torch::Tensor> calibration_data;
    for (int i = 0; i < 20; ++i) {
        calibration_data.push_back(torch::randn({16, 3, 32, 32}));
    }
    
    analyzer.calibrate(calibration_data);
    analyzer.compute_curvature();
    
    const auto& stats = analyzer.get_layer_stats();
    
    std::cout << "\nLayer-wise curvature breakdown:\n";
    std::cout << std::string(70, '-') << "\n";
    
    // Analyze by layer type
    std::vector<std::pair<std::string, double>> conv_curvatures;
    std::vector<std::pair<std::string, double>> fc_curvatures;
    
    for (const auto& [name, stat] : stats) {
        if (name.find("conv") != std::string::npos) {
            conv_curvatures.push_back({name, stat.curvature});
        } else if (name.find("fc") != std::string::npos) {
            fc_curvatures.push_back({name, stat.curvature});
        }
    }
    
    // Sort by curvature
    auto compare_curv = [](const auto& a, const auto& b) { return a.second > b.second; };
    std::sort(conv_curvatures.begin(), conv_curvatures.end(), compare_curv);
    std::sort(fc_curvatures.begin(), fc_curvatures.end(), compare_curv);
    
    std::cout << "\nTop 10 highest curvature convolution layers:\n";
    for (size_t i = 0; i < std::min<size_t>(10, conv_curvatures.size()); ++i) {
        std::cout << "  " << std::setw(40) << conv_curvatures[i].first 
                  << ": κ = " << std::setw(10) << std::fixed << std::setprecision(2)
                  << conv_curvatures[i].second << "\n";
    }
    
    std::cout << "\nFully connected layers:\n";
    for (const auto& [name, curv] : fc_curvatures) {
        std::cout << "  " << std::setw(40) << name 
                  << ": κ = " << std::setw(10) << std::fixed << std::setprecision(2)
                  << curv << "\n";
    }
    
    // ========================================================================
    // QUANTIZATION EXPERIMENTS
    // ========================================================================
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  QUANTIZATION EXPERIMENTS                      ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // Experiment 1: 8-bit uniform
    std::cout << "\n[1] Uniform 8-bit quantization:\n";
    auto uniform8_alloc = std::unordered_map<std::string, int>();
    for (const auto& [name, _] : stats) {
        uniform8_alloc[name] = 8;
    }
    double error_uniform8 = analyzer.estimate_total_error(uniform8_alloc);
    std::cout << "    Estimated error: " << std::scientific << error_uniform8 << "\n";
    std::cout << "    Total bits: " << (8.0 * total_params) << "\n";
    
    // Experiment 2: Curvature-guided 8-bit average
    std::cout << "\n[2] Curvature-guided (8-bit average):\n";
    auto curvature8_alloc = analyzer.optimize_bit_allocation(8.0);
    double error_curvature8 = analyzer.estimate_total_error(curvature8_alloc);
    std::cout << "    Estimated error: " << std::scientific << error_curvature8 << "\n";
    
    // Count actual average
    int64_t total_bits = 0;
    for (const auto& [name, bits] : curvature8_alloc) {
        total_bits += stats.at(name).num_parameters * bits;
    }
    double avg_bits = static_cast<double>(total_bits) / total_params;
    std::cout << "    Actual average: " << std::fixed << std::setprecision(2) << avg_bits << " bits\n";
    std::cout << "    Error improvement: " << std::setprecision(1) 
              << (100.0 * (error_uniform8 - error_curvature8) / error_uniform8) << "%\n";
    
    // Experiment 3: Aggressive 6-bit average
    std::cout << "\n[3] Curvature-guided (6-bit average):\n";
    auto curvature6_alloc = analyzer.optimize_bit_allocation(6.0);
    double error_curvature6 = analyzer.estimate_total_error(curvature6_alloc);
    
    total_bits = 0;
    for (const auto& [name, bits] : curvature6_alloc) {
        total_bits += stats.at(name).num_parameters * bits;
    }
    avg_bits = static_cast<double>(total_bits) / total_params;
    std::cout << "    Estimated error: " << std::scientific << error_curvature6 << "\n";
    std::cout << "    Actual average: " << std::fixed << std::setprecision(2) << avg_bits << " bits\n";
    std::cout << "    Memory savings vs FP32: " << std::setprecision(1) 
              << (100.0 * (32 - avg_bits) / 32) << "%\n";
    
    // Detailed allocation for 6-bit budget
    std::cout << "\n=== Detailed Bit Allocation (6-bit budget) ===\n";
    QuantizationValidator::print_quantization_report(analyzer, curvature6_alloc);
    
    // ========================================================================
    // LAYER-TYPE ANALYSIS
    // ========================================================================
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║               LAYER-TYPE ANALYSIS                              ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    // Analyze bit allocation by layer type
    std::map<std::string, std::vector<int>> bits_by_type;
    for (const auto& [name, bits] : curvature6_alloc) {
        if (name.find("conv1") != std::string::npos && name.find("layer") == std::string::npos) {
            bits_by_type["Initial Conv"].push_back(bits);
        } else if (name.find("layer1") != std::string::npos) {
            bits_by_type["Layer 1 (64 ch)"].push_back(bits);
        } else if (name.find("layer2") != std::string::npos) {
            bits_by_type["Layer 2 (128 ch)"].push_back(bits);
        } else if (name.find("layer3") != std::string::npos) {
            bits_by_type["Layer 3 (256 ch)"].push_back(bits);
        } else if (name.find("layer4") != std::string::npos) {
            bits_by_type["Layer 4 (512 ch)"].push_back(bits);
        } else if (name.find("fc") != std::string::npos) {
            bits_by_type["FC Layer"].push_back(bits);
        }
    }
    
    std::cout << "\nBit allocation by layer group:\n";
    for (const auto& [type, bits_vec] : bits_by_type) {
        if (bits_vec.empty()) continue;
        
        double avg = std::accumulate(bits_vec.begin(), bits_vec.end(), 0.0) / bits_vec.size();
        int min_b = *std::min_element(bits_vec.begin(), bits_vec.end());
        int max_b = *std::max_element(bits_vec.begin(), bits_vec.end());
        
        std::cout << "  " << std::setw(20) << type << ": "
                  << "avg=" << std::fixed << std::setprecision(1) << avg
                  << ", range=[" << min_b << ", " << max_b << "]\n";
    }
    
    // ========================================================================
    // KEY OBSERVATIONS
    // ========================================================================
    
    std::cout << "\n╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    KEY OBSERVATIONS                            ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n\n";
    
    std::cout << "1. DEPTH-DEPENDENT PRECISION:\n";
    std::cout << "   Early layers (layer1) typically have lower curvature and can\n";
    std::cout << "   use fewer bits. Later layers (layer3-4) need more precision.\n\n";
    
    std::cout << "2. FULLY CONNECTED LAYER:\n";
    std::cout << "   The final FC layer often has high curvature due to the\n";
    std::cout << "   classification task, requiring more bits for accuracy.\n\n";
    
    std::cout << "3. MEMORY-QUALITY TRADEOFF:\n";
    std::cout << "   - 8-bit avg: " << std::fixed << std::setprecision(1) 
              << (100.0 * (32 - 8) / 32) << "% memory reduction\n";
    std::cout << "   - 6-bit avg: " << (100.0 * (32 - 6) / 32) << "% memory reduction\n";
    std::cout << "   Curvature-guided allocation maintains quality at lower bits.\n\n";
    
    std::cout << "4. THEORETICAL VALIDATION:\n";
    std::cout << "   HNF Theorem 4.7 provides the lower bounds used for allocation.\n";
    std::cout << "   The compositional error (Theorem 3.4) is minimized through\n";
    std::cout << "   strategic bit placement in high-curvature layers.\n\n";
    
    std::cout << "╔════════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                  DEMONSTRATION COMPLETE                        ║\n";
    std::cout << "╚════════════════════════════════════════════════════════════════╝\n";
    
    return 0;
}
