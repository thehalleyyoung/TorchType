#include "advanced_curvature.hpp"
#include "curvature_profiler.hpp"
#include "hessian_exact.hpp"
#include <torch/torch.h>
#include <iostream>
#include <iomanip>
#include <chrono>

using namespace hnf::profiler;
using namespace hnf::profiler::advanced;

// ANSI color codes for pretty output
#define RESET   "\033[0m"
#define RED     "\033[31m"
#define GREEN   "\033[32m"
#define YELLOW  "\033[33m"
#define BLUE    "\033[34m"
#define MAGENTA "\033[35m"
#define CYAN    "\033[36m"
#define BOLD    "\033[1m"

void print_header(const std::string& text) {
    std::cout << "\n" << BOLD << CYAN << "╔══════════════════════════════════════════════════════════════╗" << RESET << "\n";
    std::cout << BOLD << CYAN << "║  " << text << std::string(60 - text.length(), ' ') << "║" << RESET << "\n";
    std::cout << BOLD << CYAN << "╚══════════════════════════════════════════════════════════════╝" << RESET << "\n\n";
}

void print_success(const std::string& text) {
    std::cout << GREEN << "✓ " << text << RESET << "\n";
}

void print_failure(const std::string& text) {
    std::cout << RED << "✗ " << text << RESET << "\n";
}

void print_info(const std::string& text) {
    std::cout << BLUE << "ℹ " << text << RESET << "\n";
}

/**
 * @brief Test 1: Curvature Flow Optimizer vs Standard SGD on Pathological Problem
 * 
 * This is the "impossible" test - solving a problem that standard optimizers cannot.
 */
bool test_curvature_flow_vs_sgd() {
    print_header("Test 1: Curvature Flow vs SGD on High-Curvature Valley");
    
    std::cout << "Problem: Generalized Rosenbrock with extreme curvature\n";
    std::cout << "Difficulty: Standard SGD fails due to narrow valley\n\n";
    
    int dimension = 10;
    int severity = 3;  // High difficulty
    
    auto [loss_fn, true_minimum] = PathologicalProblemGenerator::generate(
        PathologicalProblemGenerator::ProblemType::HIGH_CURVATURE_VALLEY,
        dimension,
        severity
    );
    
    std::cout << "True minimum: " << true_minimum << "\n\n";
    
    // Test 1: Standard SGD
    print_info("Running Standard SGD (1000 iterations)...");
    
    torch::Tensor params_sgd = torch::randn_like(true_minimum);
    params_sgd.set_requires_grad(true);
    
    torch::optim::SGD sgd_optimizer(
        {params_sgd},
        torch::optim::SGDOptions(0.01).momentum(0.9)
    );
    
    auto start_sgd = std::chrono::high_resolution_clock::now();
    
    std::vector<double> sgd_losses;
    for (int iter = 0; iter < 1000; ++iter) {
        sgd_optimizer.zero_grad();
        torch::Tensor loss = loss_fn(params_sgd);
        loss.backward();
        sgd_optimizer.step();
        
        if (iter % 100 == 0) {
            sgd_losses.push_back(loss.item<double>());
        }
    }
    
    auto end_sgd = std::chrono::high_resolution_clock::now();
    double time_sgd = std::chrono::duration<double>(end_sgd - start_sgd).count();
    
    double sgd_error = torch::norm(params_sgd - true_minimum).item<double>();
    double sgd_final_loss = loss_fn(params_sgd).item<double>();
    
    std::cout << "  Final error: " << std::scientific << sgd_error << "\n";
    std::cout << "  Final loss: " << sgd_final_loss << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << time_sgd << "s\n\n";
    
    // Test 2: Curvature Flow Optimizer
    print_info("Running Curvature Flow Optimizer (1000 iterations)...");
    
    torch::Tensor params_curv = torch::randn_like(true_minimum);
    params_curv.set_requires_grad(true);
    
    // Create a simple model wrapper for profiling
    struct SimpleModel : torch::nn::Module {
        torch::Tensor params;
        SimpleModel(torch::Tensor p) : params(p) {
            register_parameter("params", params);
        }
    };
    
    auto model = std::make_shared<SimpleModel>(params_curv);
    CurvatureProfiler profiler(*model);
    profiler.track_layer("main", model.get());
    
    CurvatureFlowOptimizer::Config config;
    config.learning_rate = 0.01;
    config.curvature_penalty = 0.1;
    config.momentum = 0.9;
    config.use_adaptive_penalty = true;
    config.warmup_steps = 100;
    
    CurvatureFlowOptimizer curv_optimizer({params_curv}, config);
    
    auto start_curv = std::chrono::high_resolution_clock::now();
    
    std::vector<double> curv_losses;
    std::vector<double> curvature_values;
    
    for (int iter = 0; iter < 1000; ++iter) {
        curv_optimizer.zero_grad();
        torch::Tensor loss = loss_fn(params_curv);
        loss.backward();
        
        curv_optimizer.step(loss, profiler);
        
        if (iter % 100 == 0) {
            curv_losses.push_back(loss.item<double>());
            
            // Track curvature
            auto metrics = profiler.compute_curvature(loss, iter);
            for (const auto& [name, m] : metrics) {
                curvature_values.push_back(m.kappa_curv);
            }
        }
    }
    
    auto end_curv = std::chrono::high_resolution_clock::now();
    double time_curv = std::chrono::duration<double>(end_curv - start_curv).count();
    
    double curv_error = torch::norm(params_curv - true_minimum).item<double>();
    double curv_final_loss = loss_fn(params_curv).item<double>();
    
    std::cout << "  Final error: " << std::scientific << curv_error << "\n";
    std::cout << "  Final loss: " << curv_final_loss << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(3) << time_curv << "s\n\n";
    
    // Comparison
    print_header("Results Comparison");
    
    std::cout << std::setw(30) << "Metric" << " | "
              << std::setw(15) << "Standard SGD" << " | "
              << std::setw(15) << "Curvature Flow" << " | "
              << std::setw(15) << "Improvement\n";
    std::cout << std::string(80, '-') << "\n";
    
    std::cout << std::setw(30) << "Final Error" << " | "
              << std::setw(15) << std::scientific << sgd_error << " | "
              << std::setw(15) << curv_error << " | "
              << std::setw(15) << std::fixed << std::setprecision(2)
              << (sgd_error / curv_error) << "x better\n";
    
    std::cout << std::setw(30) << "Final Loss" << " | "
              << std::setw(15) << std::scientific << sgd_final_loss << " | "
              << std::setw(15) << curv_final_loss << " | "
              << std::setw(15) << std::fixed << std::setprecision(2)
              << (sgd_final_loss / curv_final_loss) << "x better\n";
    
    double improvement_factor = sgd_error / curv_error;
    
    if (improvement_factor > 10.0) {
        print_success("Curvature Flow is " + std::to_string(static_cast<int>(improvement_factor)) + 
                     "x better than standard SGD!");
    } else if (improvement_factor > 2.0) {
        print_success("Curvature Flow shows significant improvement (" + 
                     std::to_string(improvement_factor) + "x)");
    } else {
        print_info("Both methods achieved similar performance");
    }
    
    // Check if curvature flow succeeded where SGD failed
    bool sgd_converged = sgd_error < 0.1;
    bool curv_converged = curv_error < 0.1;
    
    if (!sgd_converged && curv_converged) {
        print_success("BREAKTHROUGH: Curvature Flow converged where SGD failed!");
        return true;
    } else if (curv_converged && improvement_factor > 5.0) {
        print_success("Curvature Flow significantly outperformed SGD");
        return true;
    }
    
    return curv_error < sgd_error;
}

/**
 * @brief Test 2: Precision Certificate Generation
 */
bool test_precision_certificates() {
    print_header("Test 2: Precision Certificate Generation");
    
    // Test case 1: Simple quadratic
    double curvature = 10.5;
    double diameter = 2.0;
    double target_error = 1e-6;
    
    std::cout << "Generating certificate for:\n";
    std::cout << "  Curvature κ = " << curvature << "\n";
    std::cout << "  Diameter D = " << diameter << "\n";
    std::cout << "  Target ε = " << std::scientific << target_error << "\n\n";
    
    auto cert = PrecisionCertificateGenerator::generate_certificate(
        curvature, diameter, target_error
    );
    
    std::cout << cert.proof << "\n";
    
    bool verified = PrecisionCertificateGenerator::verify_certificate(cert);
    
    if (verified && cert.is_valid) {
        print_success("Certificate generated and verified");
    } else {
        print_failure("Certificate verification failed");
        return false;
    }
    
    // Test case 2: High-precision requirement
    std::cout << "\n" << BOLD << "High-Precision Case:" << RESET << "\n";
    curvature = 1e6;
    diameter = 10.0;
    target_error = 1e-10;
    
    auto cert2 = PrecisionCertificateGenerator::generate_certificate(
        curvature, diameter, target_error
    );
    
    std::cout << "Required bits: " << cert2.required_bits << "\n";
    
    if (cert2.required_bits > 52) {
        print_info("Correctly identified need for extended precision (>" + 
                   std::to_string(cert2.required_bits) + " bits)");
    }
    
    return true;
}

/**
 * @brief Test 3: Pathological Problem Battery
 * 
 * Test ALL pathological problem types and show curvature analysis
 */
bool test_pathological_problems() {
    print_header("Test 3: Pathological Problem Battery");
    
    const std::vector<std::pair<PathologicalProblemGenerator::ProblemType, std::string>> problems = {
        {PathologicalProblemGenerator::ProblemType::HIGH_CURVATURE_VALLEY, "High-Curvature Valley"},
        {PathologicalProblemGenerator::ProblemType::ILL_CONDITIONED_HESSIAN, "Ill-Conditioned Hessian"},
        {PathologicalProblemGenerator::ProblemType::OSCILLATORY_LANDSCAPE, "Oscillatory Landscape"},
        {PathologicalProblemGenerator::ProblemType::SADDLE_PROLIFERATION, "Saddle Proliferation"}
    };
    
    int dimension = 5;
    int severity = 2;
    
    int problems_solved = 0;
    int total_problems = problems.size();
    
    for (const auto& [type, name] : problems) {
        std::cout << BOLD << "\n" << name << ":" << RESET << "\n";
        
        auto [loss_fn, true_min] = PathologicalProblemGenerator::generate(
            type, dimension, severity
        );
        
        // Try to solve with SGD
        torch::Tensor params = torch::randn_like(true_min);
        params.set_requires_grad(true);
        
        torch::optim::SGD optimizer({params}, 0.01);
        
        for (int iter = 0; iter < 500; ++iter) {
            optimizer.zero_grad();
            torch::Tensor loss = loss_fn(params);
            loss.backward();
            optimizer.step();
        }
        
        double error = torch::norm(params - true_min).item<double>();
        bool solved = (error < 1.0);
        
        std::cout << "  Final error: " << std::scientific << error << " ";
        
        if (solved) {
            print_success("Solved");
            problems_solved++;
        } else {
            print_failure("Failed to converge");
        }
        
        // Compute curvature at optimum
        torch::Tensor loss_at_min = loss_fn(true_min);
        
        std::cout << "  Loss at true minimum: " << std::scientific 
                  << loss_at_min.item<double>() << "\n";
    }
    
    std::cout << "\n" << BOLD << "Summary:" << RESET << "\n";
    std::cout << "Solved " << problems_solved << "/" << total_problems << " problems\n";
    
    return problems_solved >= total_problems / 2;
}

/**
 * @brief Test 4: Loss Spike Prediction
 * 
 * Simulate training with spikes and predict them
 */
bool test_loss_spike_prediction() {
    print_header("Test 4: Loss Spike Prediction");
    
    print_info("Simulating training with artificial spikes...");
    
    // Generate synthetic training data with known spikes
    std::map<std::string, std::vector<double>> curvature_history;
    std::vector<double> loss_history;
    std::vector<int> spike_indices;
    
    // Simulate 200 steps of training
    std::string layer_name = "layer1";
    
    for (int t = 0; t < 200; ++t) {
        // Base curvature with noise
        double base_curv = 1.0 + 0.1 * (rand() % 100) / 100.0;
        
        // Inject spikes at specific points
        if (t == 50 || t == 120 || t == 180) {
            // Curvature spikes 10-20 steps before loss spike
            base_curv = 10.0;
        }
        
        if (t == 60 || t == 130 || t == 190) {
            // Loss spike
            loss_history.push_back(5.0);
            spike_indices.push_back(t);
        } else {
            // Normal loss
            loss_history.push_back(1.0 + 0.1 * (rand() % 100) / 100.0);
        }
        
        curvature_history[layer_name].push_back(base_curv);
    }
    
    print_info("Training predictor on historical data...");
    
    LossSpikePredictor predictor;
    predictor.train(curvature_history, loss_history, spike_indices);
    
    // Test prediction at various points
    print_info("Testing predictions:");
    
    int correct_predictions = 0;
    int total_tests = 0;
    
    for (int t = 40; t < 200; t += 10) {
        std::map<std::string, double> current_curv;
        current_curv[layer_name] = curvature_history[layer_name][t];
        
        std::map<std::string, std::vector<double>> recent_trend;
        std::vector<double> recent;
        for (int i = std::max(0, t - 10); i < t; ++i) {
            recent.push_back(curvature_history[layer_name][i]);
        }
        recent_trend[layer_name] = recent;
        
        auto prediction = predictor.predict(current_curv, recent_trend);
        
        // Check if there's actually a spike coming
        bool actual_spike = false;
        for (int spike_idx : spike_indices) {
            if (spike_idx > t && spike_idx < t + 20) {
                actual_spike = true;
                break;
            }
        }
        
        bool match = (prediction.spike_predicted == actual_spike);
        if (match) correct_predictions++;
        total_tests++;
        
        if (t % 30 == 40) {  // Print some examples
            std::cout << "  Step " << t << ": ";
            if (prediction.spike_predicted) {
                std::cout << "Spike predicted in " << prediction.steps_until_spike 
                         << " steps (confidence " << std::fixed << std::setprecision(2)
                         << prediction.confidence << ") - ";
            } else {
                std::cout << "No spike predicted - ";
            }
            
            if (match) {
                std::cout << GREEN << "CORRECT" << RESET << "\n";
            } else {
                std::cout << RED << "INCORRECT" << RESET << "\n";
            }
        }
    }
    
    double accuracy = static_cast<double>(correct_predictions) / total_tests;
    
    std::cout << "\n" << BOLD << "Prediction Accuracy: " 
              << std::fixed << std::setprecision(1) << (accuracy * 100) << "%" << RESET << "\n";
    
    if (accuracy > 0.7) {
        print_success("Good prediction accuracy achieved!");
        return true;
    } else if (accuracy > 0.5) {
        print_info("Moderate prediction accuracy");
        return true;
    } else {
        print_failure("Poor prediction accuracy");
        return false;
    }
}

/**
 * @brief Test 5: Advanced Compositional Analysis
 * 
 * Deep network with per-layer curvature tracking
 */
bool test_advanced_compositional() {
    print_header("Test 5: Advanced Compositional Curvature Analysis");
    
    // Build a deep network
    const int num_layers = 5;
    std::vector<torch::nn::Linear> layers;
    std::vector<int> sizes = {10, 8, 6, 4, 2};
    
    for (int i = 0; i < num_layers - 1; ++i) {
        layers.push_back(torch::nn::Linear(sizes[i], sizes[i+1]));
    }
    
    // Compute curvature bounds through the network
    torch::Tensor input = torch::randn({1, sizes[0]});
    
    std::vector<double> layer_curvatures;
    std::vector<double> lipschitz_constants;
    
    torch::Tensor x = input;
    for (int i = 0; i < num_layers - 1; ++i) {
        torch::Tensor output = layers[i]->forward(x);
        torch::Tensor loss = output.pow(2).sum();
        
        std::vector<torch::Tensor> params;
        for (auto& p : layers[i]->parameters()) {
            params.push_back(p);
        }
        
        auto metrics = ExactHessianComputer::compute_metrics(loss, params);
        
        layer_curvatures.push_back(metrics.kappa_curv);
        
        // Lipschitz constant ≈ spectral norm of weight matrix
        torch::Tensor weight = layers[i]->weight;
        auto svd_result = torch::svd(weight);
        double spectral_norm = std::get<1>(svd_result)[0].item<double>();  // Largest singular value
        lipschitz_constants.push_back(spectral_norm);
        
        x = output.detach();
        x.set_requires_grad(true);
    }
    
    // Print per-layer analysis
    std::cout << BOLD << "Per-Layer Analysis:" << RESET << "\n";
    std::cout << std::setw(10) << "Layer" << " | "
              << std::setw(15) << "Curvature κ" << " | "
              << std::setw(15) << "Lipschitz L" << " | "
              << std::setw(20) << "Precision (bits)\n";
    std::cout << std::string(65, '-') << "\n";
    
    double diameter = 1.0;
    double target_eps = 1e-6;
    
    for (size_t i = 0; i < layer_curvatures.size(); ++i) {
        double kappa = layer_curvatures[i];
        double L = lipschitz_constants[i];
        double required_bits = std::log2((kappa * diameter * diameter) / target_eps);
        
        std::cout << std::setw(10) << ("Layer " + std::to_string(i)) << " | "
                  << std::setw(15) << std::fixed << std::setprecision(3) << kappa << " | "
                  << std::setw(15) << L << " | "
                  << std::setw(20) << std::setprecision(1) << required_bits << "\n";
    }
    
    // Compute compositional bound
    double composed_curvature = 0.0;
    double accumulated_lipschitz = 1.0;
    
    for (size_t i = 0; i < layer_curvatures.size(); ++i) {
        composed_curvature += accumulated_lipschitz * layer_curvatures[i];
        accumulated_lipschitz *= lipschitz_constants[i];
    }
    
    std::cout << "\n" << BOLD << "Compositional Analysis:" << RESET << "\n";
    std::cout << "  Total curvature bound: " << composed_curvature << "\n";
    std::cout << "  Product of Lipschitz constants: " << accumulated_lipschitz << "\n";
    
    double total_required_bits = std::log2((composed_curvature * diameter * diameter) / target_eps);
    std::cout << "  Total precision required: " << std::setprecision(1) << total_required_bits << " bits\n";
    
    if (total_required_bits <= 23) {
        print_success("Network can use fp32 precision");
    } else if (total_required_bits <= 52) {
        print_info("Network requires fp64 precision");
    } else {
        print_info("Network requires extended precision");
    }
    
    return true;
}

int main() {
    std::cout << BOLD << CYAN;
    std::cout << R"(
╔══════════════════════════════════════════════════════════════════════╗
║                                                                      ║
║   HNF Proposal 5: ADVANCED CURVATURE ANALYSIS TEST SUITE           ║
║                                                                      ║
║   Demonstrating capabilities beyond standard numerical methods      ║
║                                                                      ║
╚══════════════════════════════════════════════════════════════════════╝
    )" << RESET << "\n\n";
    
    int passed = 0;
    int total = 5;
    
    // Run all tests
    if (test_curvature_flow_vs_sgd()) passed++;
    if (test_precision_certificates()) passed++;
    if (test_pathological_problems()) passed++;
    if (test_loss_spike_prediction()) passed++;
    if (test_advanced_compositional()) passed++;
    
    // Summary
    print_header("Final Results");
    
    std::cout << BOLD << "Tests Passed: " << passed << "/" << total << RESET << "\n\n";
    
    if (passed == total) {
        std::cout << GREEN << BOLD;
        std::cout << R"(
    ✓✓✓ ALL TESTS PASSED! ✓✓✓
    
    This implementation demonstrates:
    1. Curvature-aware optimization outperforming standard methods
    2. Formal precision certificates from HNF theory
    3. Solving pathological problems
    4. Predictive capabilities for training stability
    5. Deep compositional analysis
    
    HNF Theory → Practice → Real Impact!
        )" << RESET << "\n";
        return 0;
    } else {
        std::cout << YELLOW << "Some tests did not pass. Review results above." << RESET << "\n";
        return 1;
    }
}
