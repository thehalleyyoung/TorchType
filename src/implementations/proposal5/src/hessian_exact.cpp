#include "hessian_exact.hpp"
#include <cmath>
#include <stdexcept>
#include <sstream>
#include <iostream>

namespace hnf {
namespace profiler {

// ============================================================================
// ExactHessianComputer Implementation
// ============================================================================

torch::Tensor ExactHessianComputer::flatten_parameters(
    const std::vector<torch::Tensor>& params) {
    std::vector<torch::Tensor> flat_params;
    for (const auto& p : params) {
        if (p.defined() && p.requires_grad()) {
            flat_params.push_back(p.flatten().detach());
        }
    }
    if (flat_params.empty()) {
        return torch::zeros({0});
    }
    return torch::cat(flat_params);
}

torch::Tensor ExactHessianComputer::compute_gradient(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters) {
    auto grads = torch::autograd::grad({loss}, parameters,
                                       {}, /*retain_graph=*/true,
                                       /*create_graph=*/false,
                                       /*allow_unused=*/true);
    std::vector<torch::Tensor> flat_grads;
    for (const auto& g : grads) {
        if (g.defined()) {
            flat_grads.push_back(g.flatten());
        } else {
            // If gradient is undefined, use zeros
            flat_grads.push_back(torch::zeros({1}));
        }
    }
    return torch::cat(flat_grads);
}

Eigen::MatrixXd ExactHessianComputer::compute_hessian_matrix(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters) {
    
    // Count total number of parameters
    int64_t total_params = 0;
    for (const auto& p : parameters) {
        if (p.defined() && p.requires_grad()) {
            total_params += p.numel();
        }
    }
    
    if (total_params == 0) {
        return Eigen::MatrixXd(0, 0);
    }
    
    if (total_params > 10000) {
        std::cerr << "WARNING: Computing exact Hessian for " << total_params 
                  << " parameters. This will be slow and memory-intensive." << std::endl;
    }
    
    // Compute first derivatives (gradient)
    auto grads = torch::autograd::grad({loss}, parameters,
                                       {}, /*retain_graph=*/true,
                                       /*create_graph=*/true,  // Need 2nd order
                                       /*allow_unused=*/true);
    
    // Flatten gradients
    std::vector<torch::Tensor> grad_list;
    for (const auto& g : grads) {
        if (g.defined()) {
            grad_list.push_back(g.flatten());
        }
    }
    
    if (grad_list.empty()) {
        return Eigen::MatrixXd(0, 0);
    }
    
    auto flat_grad = torch::cat(grad_list);
    
    // Compute Hessian: H[i,j] = ∂²L/∂θ_i∂θ_j = ∂(∂L/∂θ_i)/∂θ_j
    Eigen::MatrixXd hessian = Eigen::MatrixXd::Zero(total_params, total_params);
    
    for (int64_t i = 0; i < total_params; ++i) {
        // Compute gradient of flat_grad[i] w.r.t. all parameters
        auto grad_i = flat_grad[i];
        
        auto second_grads = torch::autograd::grad({grad_i}, parameters,
                                                   {}, /*retain_graph=*/true,
                                                   /*create_graph=*/false,
                                                   /*allow_unused=*/true);
        
        // Flatten and store in hessian matrix
        std::vector<torch::Tensor> second_grad_list;
        for (const auto& sg : second_grads) {
            if (sg.defined()) {
                second_grad_list.push_back(sg.flatten());
            } else {
                // Fill with zeros if undefined
                int64_t size = 0;
                for (const auto& p : parameters) {
                    if (p.defined() && p.requires_grad()) {
                        size += p.numel();
                        break;
                    }
                }
                second_grad_list.push_back(torch::zeros({size}));
            }
        }
        
        if (!second_grad_list.empty()) {
            auto flat_second_grad = torch::cat(second_grad_list);
            auto cpu_data = flat_second_grad.cpu().contiguous();
            
            // Handle both float and double tensors
            for (int64_t j = 0; j < total_params && j < cpu_data.size(0); ++j) {
                if (cpu_data.scalar_type() == torch::kFloat) {
                    hessian(i, j) = static_cast<double>(cpu_data.data_ptr<float>()[j]);
                } else if (cpu_data.scalar_type() == torch::kDouble) {
                    hessian(i, j) = cpu_data.data_ptr<double>()[j];
                } else {
                    // Fallback: convert to double
                    hessian(i, j) = cpu_data[j].item<double>();
                }
            }
        }
    }
    
    // Make Hessian symmetric (it should be, but numerical errors can cause asymmetry)
    hessian = (hessian + hessian.transpose()) / 2.0;
    
    return hessian;
}

ExactHessianComputer::HessianMetrics ExactHessianComputer::compute_metrics(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters) {
    
    HessianMetrics metrics;
    
    // Compute exact Hessian matrix
    Eigen::MatrixXd H = compute_hessian_matrix(loss, parameters);
    
    if (H.rows() == 0) {
        // No parameters, return zero metrics
        metrics.spectral_norm = 0.0;
        metrics.frobenius_norm = 0.0;
        metrics.trace = 0.0;
        metrics.determinant = 0.0;
        metrics.condition_number = 0.0;
        metrics.rank = 0;
        metrics.is_positive_definite = false;
        metrics.kappa_curv = 0.0;
        return metrics;
    }
    
    // Compute eigenvalue decomposition
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(H);
    
    if (eigensolver.info() != Eigen::Success) {
        throw std::runtime_error("Eigenvalue decomposition failed");
    }
    
    Eigen::VectorXd eigenvalues = eigensolver.eigenvalues();
    
    // Extract metrics
    metrics.spectral_norm = eigenvalues.cwiseAbs().maxCoeff();
    metrics.frobenius_norm = H.norm();
    metrics.trace = H.trace();
    metrics.determinant = eigenvalues.prod();
    
    // Store eigenvalues
    metrics.eigenvalues.resize(eigenvalues.size());
    for (int i = 0; i < eigenvalues.size(); ++i) {
        metrics.eigenvalues[i] = eigenvalues(i);
    }
    
    // Condition number (ratio of largest to smallest nonzero eigenvalue)
    double max_eval = eigenvalues.cwiseAbs().maxCoeff();
    double min_eval = 0.0;
    for (int i = 0; i < eigenvalues.size(); ++i) {
        if (std::abs(eigenvalues(i)) > 1e-10) {
            if (min_eval == 0.0 || std::abs(eigenvalues(i)) < min_eval) {
                min_eval = std::abs(eigenvalues(i));
            }
        }
    }
    
    if (min_eval > 0) {
        metrics.condition_number = max_eval / min_eval;
    } else {
        metrics.condition_number = std::numeric_limits<double>::infinity();
    }
    
    // Numerical rank (count eigenvalues > tolerance)
    metrics.rank = 0;
    for (double eval : metrics.eigenvalues) {
        if (std::abs(eval) > 1e-10) {
            metrics.rank++;
        }
    }
    
    // Check positive definiteness
    metrics.is_positive_definite = true;
    for (double eval : metrics.eigenvalues) {
        if (eval <= 0) {
            metrics.is_positive_definite = false;
            break;
        }
    }
    
    // HNF curvature invariant: κ_f^{curv} = (1/2) ||D²f||_op
    metrics.kappa_curv = 0.5 * metrics.spectral_norm;
    
    return metrics;
}

double ExactHessianComputer::compute_spectral_norm_stochastic(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters,
    int num_iterations,
    int num_samples) {
    
    // Power iteration for dominant eigenvalue of H
    // Uses Hessian-vector products to avoid forming H explicitly
    
    int64_t total_params = 0;
    for (const auto& p : parameters) {
        if (p.defined() && p.requires_grad()) {
            total_params += p.numel();
        }
    }
    
    if (total_params == 0) {
        return 0.0;
    }
    
    double max_eigenvalue = 0.0;
    
    for (int sample = 0; sample < num_samples; ++sample) {
        // Initialize random vector
        std::vector<torch::Tensor> v;
        for (const auto& p : parameters) {
            if (p.defined() && p.requires_grad()) {
                v.push_back(torch::randn_like(p));
            }
        }
        
        // Normalize
        double norm = 0.0;
        for (const auto& vi : v) {
            norm += vi.pow(2).sum().item<double>();
        }
        norm = std::sqrt(norm);
        for (auto& vi : v) {
            vi = vi / norm;
        }
        
        // Power iteration
        for (int iter = 0; iter < num_iterations; ++iter) {
            // Compute gradient
            auto grads = torch::autograd::grad({loss}, parameters,
                                               {}, /*retain_graph=*/true,
                                               /*create_graph=*/true);
            
            // Compute grad · v
            torch::Tensor gv = torch::zeros({1}, loss.options());
            for (size_t i = 0; i < grads.size(); ++i) {
                if (grads[i].defined() && i < v.size() && v[i].defined()) {
                    gv = gv + (grads[i] * v[i]).sum();
                }
            }
            
            // Compute H·v = ∇(grad · v)
            auto hv = torch::autograd::grad({gv}, parameters,
                                            {}, /*retain_graph=*/true,
                                            /*create_graph=*/false,
                                            /*allow_unused=*/true);
            
            // Normalize H·v to get next v
            v.clear();
            double hv_norm = 0.0;
            for (const auto& hvi : hv) {
                if (hvi.defined()) {
                    v.push_back(hvi.detach());
                    hv_norm += hvi.pow(2).sum().item<double>();
                }
            }
            hv_norm = std::sqrt(hv_norm);
            
            if (hv_norm < 1e-12) {
                break;  // Converged to zero
            }
            
            for (auto& vi : v) {
                vi = vi / hv_norm;
            }
            
            // Estimate eigenvalue
            if (iter == num_iterations - 1) {
                max_eigenvalue = std::max(max_eigenvalue, hv_norm);
            }
        }
    }
    
    return max_eigenvalue;
}

double ExactHessianComputer::verify_hessian_finite_diff(
    std::function<torch::Tensor(const std::vector<torch::Tensor>&)> loss_fn,
    const std::vector<torch::Tensor>& parameters,
    double step_size) {
    
    // Compute Hessian via autograd
    torch::Tensor loss = loss_fn(parameters);
    Eigen::MatrixXd H_auto = compute_hessian_matrix(loss, parameters);
    
    int64_t n = H_auto.rows();
    if (n == 0 || n > 100) {
        // Too large for finite differences
        return 0.0;
    }
    
    // Compute Hessian via finite differences
    Eigen::MatrixXd H_fd = Eigen::MatrixXd::Zero(n, n);
    
    // Flatten parameters
    std::vector<torch::Tensor> params_copy;
    std::vector<std::vector<int64_t>> shapes;
    for (const auto& p : parameters) {
        if (p.defined() && p.requires_grad()) {
            params_copy.push_back(p.detach().clone());
            shapes.push_back(p.sizes().vec());
        }
    }
    
    auto get_param_value = [&](int64_t idx) -> double {
        int64_t offset = 0;
        for (size_t pi = 0; pi < params_copy.size(); ++pi) {
            int64_t numel = params_copy[pi].numel();
            if (idx < offset + numel) {
                auto flat = params_copy[pi].flatten();
                return flat[idx - offset].item<double>();
            }
            offset += numel;
        }
        return 0.0;
    };
    
    auto set_param_value = [&](int64_t idx, double value) {
        int64_t offset = 0;
        for (size_t pi = 0; pi < params_copy.size(); ++pi) {
            int64_t numel = params_copy[pi].numel();
            if (idx < offset + numel) {
                auto flat = params_copy[pi].flatten();
                flat[idx - offset] = value;
                params_copy[pi] = flat.reshape(shapes[pi]);
                return;
            }
            offset += numel;
        }
    };
    
    // Central finite difference for Hessian
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = i; j < n; ++j) {
            // Save original values
            double val_i = get_param_value(i);
            double val_j = get_param_value(j);
            
            // f(x + h_i + h_j)
            set_param_value(i, val_i + step_size);
            set_param_value(j, val_j + step_size);
            double f_pp = loss_fn(params_copy).item<double>();
            
            // f(x + h_i - h_j)
            set_param_value(j, val_j - step_size);
            double f_pm = loss_fn(params_copy).item<double>();
            
            // f(x - h_i + h_j)
            set_param_value(i, val_i - step_size);
            set_param_value(j, val_j + step_size);
            double f_mp = loss_fn(params_copy).item<double>();
            
            // f(x - h_i - h_j)
            set_param_value(j, val_j - step_size);
            double f_mm = loss_fn(params_copy).item<double>();
            
            // H[i,j] ≈ (f++ - f+- - f-+ + f--) / (4h²)
            H_fd(i, j) = (f_pp - f_pm - f_mp + f_mm) / (4.0 * step_size * step_size);
            H_fd(j, i) = H_fd(i, j);  // Symmetric
            
            // Restore
            set_param_value(i, val_i);
            set_param_value(j, val_j);
        }
    }
    
    // Compute relative error
    double max_error = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        for (int64_t j = 0; j < n; ++j) {
            double error = std::abs(H_auto(i,j) - H_fd(i,j));
            double denom = std::max(std::abs(H_auto(i,j)), std::abs(H_fd(i,j)));
            if (denom > 1e-10) {
                max_error = std::max(max_error, error / denom);
            }
        }
    }
    
    return max_error;
}

// ============================================================================
// CompositionalCurvatureValidator Implementation
// ============================================================================

std::string CompositionalCurvatureValidator::CompositionMetrics::to_string() const {
    std::ostringstream oss;
    oss << "Compositional Curvature Metrics:\n";
    oss << "  Layer f: κ_f = " << kappa_f << ", L_f = " << L_f << "\n";
    oss << "  Layer g: κ_g = " << kappa_g << ", L_g = " << L_g << "\n";
    oss << "  Composition:\n";
    oss << "    Actual:     κ_{g∘f} = " << kappa_composed_actual << "\n";
    oss << "    Bound:      κ_g·L_f² + L_g·κ_f = " << kappa_composed_bound << "\n";
    oss << "    Tightness:  " << (bound_tightness * 100.0) << "%\n";
    oss << "    Slack:      " << bound_slack << "\n";
    oss << "  Bound " << (bound_satisfied ? "SATISFIED ✓" : "VIOLATED ✗") << "\n";
    return oss.str();
}

CompositionalCurvatureValidator::CompositionMetrics 
CompositionalCurvatureValidator::validate_composition(
    std::function<torch::Tensor(torch::Tensor)> layer_f,
    std::function<torch::Tensor(torch::Tensor)> layer_g,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    torch::Tensor input,
    const std::vector<torch::Tensor>& params_f,
    const std::vector<torch::Tensor>& params_g) {
    
    CompositionMetrics metrics;
    
    // Compute output of first layer
    torch::Tensor intermediate = layer_f(input);
    
    // Compute loss through f only
    torch::Tensor loss_f = loss_fn(intermediate);
    
    // Compute curvature of f
    auto hessian_metrics_f = ExactHessianComputer::compute_metrics(loss_f, params_f);
    metrics.kappa_f = hessian_metrics_f.kappa_curv;
    
    // Compute curvature of g
    torch::Tensor output = layer_g(intermediate.detach());
    torch::Tensor loss_g = loss_fn(output);
    auto hessian_metrics_g = ExactHessianComputer::compute_metrics(loss_g, params_g);
    metrics.kappa_g = hessian_metrics_g.kappa_curv;
    
    // Compute Lipschitz constants (use spectral norm for linear layers)
    // For simplicity, use empirical estimation
    std::vector<torch::Tensor> samples = {input};
    metrics.L_f = estimate_lipschitz_constant(layer_f, samples, true);
    
    std::vector<torch::Tensor> intermediate_samples = {intermediate};
    metrics.L_g = estimate_lipschitz_constant(layer_g, intermediate_samples, true);
    
    // Compute composed curvature
    torch::Tensor composed_output = layer_g(layer_f(input));
    torch::Tensor loss_composed = loss_fn(composed_output);
    
    // Combine parameter lists
    std::vector<torch::Tensor> all_params;
    all_params.insert(all_params.end(), params_f.begin(), params_f.end());
    all_params.insert(all_params.end(), params_g.begin(), params_g.end());
    
    auto hessian_metrics_composed = ExactHessianComputer::compute_metrics(
        loss_composed, all_params);
    metrics.kappa_composed_actual = hessian_metrics_composed.kappa_curv;
    
    // Theoretical bound from Lemma 4.2
    // κ_{g∘f} ≤ κ_g · L_f² + L_g · κ_f
    metrics.kappa_composed_bound = metrics.kappa_g * metrics.L_f * metrics.L_f 
                                  + metrics.L_g * metrics.kappa_f;
    
    // Compute tightness
    if (metrics.kappa_composed_bound > 0) {
        metrics.bound_tightness = metrics.kappa_composed_actual / metrics.kappa_composed_bound;
    } else {
        metrics.bound_tightness = 0.0;
    }
    
    metrics.bound_slack = metrics.kappa_composed_bound - metrics.kappa_composed_actual;
    metrics.bound_satisfied = (metrics.kappa_composed_actual <= metrics.kappa_composed_bound * 1.01);  // 1% tolerance
    
    return metrics;
}

double CompositionalCurvatureValidator::estimate_lipschitz_constant(
    std::function<torch::Tensor(torch::Tensor)> layer,
    const std::vector<torch::Tensor>& input_samples,
    bool use_spectral_norm) {
    
    if (input_samples.empty()) {
        return 1.0;  // Default
    }
    
    // Empirical estimation: L ≈ max ||f(x) - f(y)|| / ||x - y||
    double max_ratio = 0.0;
    
    for (size_t i = 0; i < input_samples.size(); ++i) {
        for (size_t j = i + 1; j < input_samples.size(); ++j) {
            torch::Tensor x = input_samples[i];
            torch::Tensor y = input_samples[j];
            
            torch::Tensor fx = layer(x);
            torch::Tensor fy = layer(y);
            
            double output_diff = (fx - fy).norm().item<double>();
            double input_diff = (x - y).norm().item<double>();
            
            if (input_diff > 1e-10) {
                double ratio = output_diff / input_diff;
                max_ratio = std::max(max_ratio, ratio);
            }
        }
    }
    
    // If no pairs, use gradient-based estimate
    if (max_ratio == 0.0 && !input_samples.empty()) {
        torch::Tensor x = input_samples[0];
        x.requires_grad_(true);
        torch::Tensor fx = layer(x);
        
        // Compute Jacobian norm (Frobenius norm approximation)
        auto grad_outputs = torch::ones_like(fx);
        auto grads = torch::autograd::grad({fx}, {x}, {grad_outputs},
                                           /*retain_graph=*/false,
                                           /*create_graph=*/false);
        if (!grads.empty() && grads[0].defined()) {
            max_ratio = grads[0].norm().item<double>();
        }
    }
    
    return std::max(max_ratio, 1e-6);  // Ensure non-zero
}

std::vector<CompositionalCurvatureValidator::CompositionMetrics>
CompositionalCurvatureValidator::validate_deep_composition(
    const std::vector<std::function<torch::Tensor(torch::Tensor)>>& layers,
    std::function<torch::Tensor(torch::Tensor)> loss_fn,
    torch::Tensor input,
    const std::vector<std::vector<torch::Tensor>>& all_params) {
    
    std::vector<CompositionMetrics> all_metrics;
    
    if (layers.size() < 2 || layers.size() != all_params.size()) {
        return all_metrics;
    }
    
    // Validate each consecutive pair
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        auto composition_metrics = validate_composition(
            layers[i],
            layers[i+1],
            loss_fn,
            input,
            all_params[i],
            all_params[i+1]
        );
        
        all_metrics.push_back(composition_metrics);
        
        // Update input for next layer
        input = layers[i](input).detach();
    }
    
    return all_metrics;
}

} // namespace profiler
} // namespace hnf
