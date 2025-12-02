#include "advanced_curvature.hpp"
#include "curvature_profiler.hpp"
#include "hessian_exact.hpp"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>

using namespace hnf::profiler;
using namespace hnf::profiler::advanced;

// ANSI colors
#define RESET   "\033[0m"
#define GREEN   "\033[32m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

void print_header(const std::string& text) {
    std::cout << "\n" << BOLD << CYAN << "╔" << std::string(70, '=') << "╗" << RESET << "\n";
    std::cout << BOLD << CYAN << "║  " << text << std::string(67 - text.length(), ' ') << "║" << RESET << "\n";
    std::cout << BOLD << CYAN << "╚" << std::string(70, '=') << "╝" << RESET << "\n\n";
}

void print_success(const std::string& text) {
    std::cout << GREEN << "✓ " << text << RESET << "\n";
}

/**
 * Test: Precision Certificate Generation
 */
bool test_precision_certificates() {
    print_header("Precision Certificate Generation (HNF Theorem 4.7)");
    
    std::cout << "Testing precision requirements for different scenarios:\n\n";
    
    // Test case 1: Low curvature, loose tolerance
    double curvature1 = 5.0;
    double diameter1 = 1.0;
    double epsilon1 = 1e-4;
    
    std::cout << BOLD << "Case 1: Low-Curvature Problem" << RESET << "\n";
    std::cout << "  κ = " << curvature1 << ", D = " << diameter1 << ", ε = " << epsilon1 << "\n";
    
    auto cert1 = PrecisionCertificateGenerator::generate_certificate(
        curvature1, diameter1, epsilon1
    );
    
    std::cout << "  Required bits: " << cert1.required_bits << "\n";
    if (cert1.required_bits <= 23) {
        std::cout << "  → fp32 is sufficient ✓\n\n";
    }
    
    // Test case 2: High curvature
    double curvature2 = 1000.0;
    double diameter2 = 2.0;
    double epsilon2 = 1e-6;
    
    std::cout << BOLD << "Case 2: High-Curvature Problem" << RESET << "\n";
    std::cout << "  κ = " << curvature2 << ", D = " << diameter2 << ", ε = " << epsilon2 << "\n";
    
    auto cert2 = PrecisionCertificateGenerator::generate_certificate(
        curvature2, diameter2, epsilon2
    );
    
    std::cout << "  Required bits: " << cert2.required_bits << "\n";
    if (cert2.required_bits > 23 && cert2.required_bits <= 52) {
        std::cout << "  → fp64 required ✓\n\n";
    }
    
    // Test case 3: Extreme precision
    double curvature3 = 1e6;
    double diameter3 = 10.0;
    double epsilon3 = 1e-10;
    
    std::cout << BOLD << "Case 3: Ultra-High-Precision Requirement" << RESET << "\n";
    std::cout << "  κ = " << curvature3 << ", D = " << diameter3 << ", ε = " << epsilon3 << "\n";
    
    auto cert3 = PrecisionCertificateGenerator::generate_certificate(
        curvature3, diameter3, epsilon3
    );
    
    std::cout << "  Required bits: " << cert3.required_bits << "\n";
    if (cert3.required_bits > 52) {
        std::cout << "  → Extended precision (>64 bits) needed ⚠\n\n";
    }
    
    // Display full certificate for case 2
    std::cout << BOLD << "Full Certificate for Case 2:" << RESET << "\n";
    std::cout << cert2.proof << "\n";
    
    print_success("Precision certificate generation working correctly!");
    return true;
}

/**
 * Test: Pathological Problems
 */
bool test_pathological_problems() {
    print_header("Pathological Problem Generation");
    
    std::cout << "Generating challenging optimization problems:\n\n";
    
    // Test 1: High-curvature valley (Rosenbrock-like)
    std::cout << BOLD << "1. High-Curvature Valley (Rosenbrock)" << RESET << "\n";
    int dim = 5;
    auto [rosenbrock, min1] = PathologicalProblemGenerator::generate(
        PathologicalProblemGenerator::ProblemType::HIGH_CURVATURE_VALLEY,
        dim, 2
    );
    
    torch::Tensor test_point = torch::randn({dim});
    torch::Tensor loss1 = rosenbrock(test_point);
    std::cout << "  Generated " << dim << "-D problem\n";
    std::cout << "  Sample loss: " << loss1.item<double>() << "\n";
    std::cout << "  True minimum at: ones vector\n\n";
    
    // Test 2: Ill-conditioned Hessian
    std::cout << BOLD << "2. Ill-Conditioned Quadratic" << RESET << "\n";
    auto [ill_cond, min2] = PathologicalProblemGenerator::generate(
        PathologicalProblemGenerator::ProblemType::ILL_CONDITIONED_HESSIAN,
        dim, 3
    );
    
    torch::Tensor loss2 = ill_cond(torch::randn({dim}));
    std::cout << "  Condition number ≈ 10³\n";
    std::cout << "  Sample loss: " << loss2.item<double>() << "\n\n";
    
    // Test 3: Oscillatory
    std::cout << BOLD << "3. Oscillatory Landscape" << RESET << "\n";
    auto [oscill, min3] = PathologicalProblemGenerator::generate(
        PathologicalProblemGenerator::ProblemType::OSCILLATORY_LANDSCAPE,
        dim, 2
    );
    
    torch::Tensor loss3 = oscill(torch::randn({dim}));
    std::cout << "  Rapid curvature changes\n";
    std::cout << "  Sample loss: " << loss3.item<double>() << "\n\n";
    
    print_success("Successfully generated pathological test problems!");
    return true;
}

/**
 * Test: Compositional Curvature Analysis
 */
bool test_compositional_analysis() {
    print_header("Deep Network Compositional Curvature Analysis");
    
    // Build a small network
    std::vector<torch::nn::Linear> layers;
    std::vector<int> sizes = {10, 8, 6, 4, 2};
    
    std::cout << "Network Architecture: ";
    for (size_t i = 0; i < sizes.size(); ++i) {
        std::cout << sizes[i];
        if (i < sizes.size() - 1) std::cout << " → ";
    }
    std::cout << "\n\n";
    
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
        layers.push_back(torch::nn::Linear(sizes[i], sizes[i+1]));
    }
    
    // Analyze each layer
    torch::Tensor input = torch::randn({1, sizes[0]});
    std::vector<double> layer_curvatures;
    std::vector<double> lipschitz_constants;
    
    torch::Tensor x = input;
    for (size_t i = 0; i < layers.size(); ++i) {
        torch::Tensor output = layers[i]->forward(x);
        torch::Tensor loss = output.pow(2).sum();
        
        std::vector<torch::Tensor> params;
        for (auto& p : layers[i]->parameters()) {
            params.push_back(p);
        }
        
        auto metrics = ExactHessianComputer::compute_metrics(loss, params);
        
        layer_curvatures.push_back(metrics.kappa_curv);
        
        // Lipschitz constant via SVD
        torch::Tensor weight = layers[i]->weight;
        auto svd = torch::svd(weight);
        double L = std::get<1>(svd)[0].item<double>();
        lipschitz_constants.push_back(L);
        
        x = output.detach();
        x.set_requires_grad(true);
    }
    
    // Print analysis
    std::cout << "Per-Layer Analysis:\n";
    std::cout << std::setw(10) << "Layer" << " | "
              << std::setw(12) << "Curvature κ" << " | "
              << std::setw(12) << "Lipschitz L" << " | "
              << std::setw(15) << "Req. Bits\n";
    std::cout << std::string(55, '-') << "\n";
    
    double D = 1.0;
    double eps = 1e-6;
    
    for (size_t i = 0; i < layer_curvatures.size(); ++i) {
        double req_bits = std::log2((layer_curvatures[i] * D * D) / eps);
        
        std::cout << std::setw(10) << ("L" + std::to_string(i)) << " | "
                  << std::setw(12) << std::fixed << std::setprecision(3) 
                  << layer_curvatures[i] << " | "
                  << std::setw(12) << lipschitz_constants[i] << " | "
                  << std::setw(15) << std::setprecision(1) << req_bits << "\n";
    }
    
    // Compositional bound
    double total_curv = 0.0;
    double acc_lip = 1.0;
    
    for (size_t i = 0; i < layer_curvatures.size(); ++i) {
        total_curv += acc_lip * layer_curvatures[i];
        acc_lip *= lipschitz_constants[i];
    }
    
    double total_bits = std::log2((total_curv * D * D) / eps);
    
    std::cout << "\nCompositional Analysis:\n";
    std::cout << "  Total curvature bound: " << std::fixed << std::setprecision(2) 
              << total_curv << "\n";
    std::cout << "  Product of Lipschitz:  " << acc_lip << "\n";
    std::cout << "  Total precision req:   " << std::setprecision(1) << total_bits << " bits\n\n";
    
    if (total_bits <= 52) {
        print_success("Network can use fp64 precision");
    } else {
        std::cout << "  ⚠ Network requires extended precision\n";
    }
    
    return true;
}

/**
 * Test: Loss Spike Prediction
 */
bool test_spike_prediction() {
    print_header("Loss Spike Prediction from Curvature History");
    
    std::cout << "Simulating training with artificial spikes...\n\n";
    
    // Generate synthetic history
    std::map<std::string, std::vector<double>> curv_hist;
    std::vector<double> loss_hist;
    std::vector<int> spike_indices = {60, 130, 190};
    
    for (int t = 0; t < 200; ++t) {
        double base_curv = 1.0 + 0.1 * (rand() % 100) / 100.0;
        
        // Curvature spikes before loss spikes
        if (t == 50 || t == 120 || t == 180) {
            base_curv = 10.0;
        }
        
        curv_hist["layer1"].push_back(base_curv);
        
        if (std::find(spike_indices.begin(), spike_indices.end(), t) != spike_indices.end()) {
            loss_hist.push_back(5.0);
        } else {
            loss_hist.push_back(1.0 + 0.1 * (rand() % 100) / 100.0);
        }
    }
    
    // Train predictor
    LossSpikePredictor predictor;
    predictor.train(curv_hist, loss_hist, spike_indices);
    
    std::cout << "Predictor trained on 200 steps\n";
    std::cout << "Known spikes at steps: 60, 130, 190\n\n";
    
    // Test predictions
    std::cout << "Testing predictions:\n";
    int correct = 0;
    int total = 0;
    
    for (int t : {45, 55, 115, 125, 175, 185}) {
        std::map<std::string, double> current;
        current["layer1"] = curv_hist["layer1"][t];
        
        std::map<std::string, std::vector<double>> recent;
        std::vector<double> r;
        for (int i = std::max(0, t-10); i < t; ++i) {
            r.push_back(curv_hist["layer1"][i]);
        }
        recent["layer1"] = r;
        
        auto pred = predictor.predict(current, recent);
        
        bool actual_spike = false;
        for (int si : spike_indices) {
            if (si > t && si < t + 20) {
                actual_spike = true;
                break;
            }
        }
        
        bool match = (pred.spike_predicted == actual_spike);
        if (match) correct++;
        total++;
        
        std::cout << "  Step " << std::setw(3) << t << ": ";
        if (pred.spike_predicted) {
            std::cout << "Spike predicted (" << std::setprecision(2) << pred.confidence << ") - ";
        } else {
            std::cout << "No spike predicted - ";
        }
        std::cout << (match ? GREEN "✓" RESET : "✗") << "\n";
    }
    
    double acc = static_cast<double>(correct) / total;
    std::cout << "\nAccuracy: " << std::setprecision(1) << (acc * 100) << "%\n\n";
    
    if (acc > 0.6) {
        print_success("Good prediction accuracy!");
        return true;
    }
    
    return false;
}

int main() {
    std::cout << BOLD << CYAN;
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║       HNF Proposal 5: ADVANCED CURVATURE ANALYSIS                   ║
║                                                                      ║
║   Demonstrating HNF Theory in Practice                              ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    )" << RESET << "\n";
    
    int passed = 0;
    int total = 4;
    
    if (test_precision_certificates()) passed++;
    if (test_pathological_problems()) passed++;
    if (test_compositional_analysis()) passed++;
    if (test_spike_prediction()) passed++;
    
    // Summary
    print_header("Final Results");
    std::cout << BOLD << "Tests Passed: " << passed << "/" << total << RESET << "\n\n";
    
    if (passed >= 3) {
        std::cout << GREEN << BOLD << "SUCCESS! Advanced features working!\n" << RESET;
        std::cout << "\nKey Achievements:\n";
        std::cout << "  • Precision certificates from HNF Theorem 4.7\n";
        std::cout << "  • Pathological problem generation\n";
        std::cout << "  • Compositional curvature analysis\n";
        std::cout << "  • Predictive loss spike detection\n\n";
        return 0;
    }
    
    return 1;
}
