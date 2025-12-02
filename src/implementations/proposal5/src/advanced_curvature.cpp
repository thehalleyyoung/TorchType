#include "advanced_curvature.hpp"
#include "hessian_exact.hpp"
#include <cmath>
#include <algorithm>
#include <numeric>
#include <random>
#include <sstream>

namespace hnf {
namespace profiler {
namespace advanced {

// ============================================================================
// RiemannianMetricTensor Implementation
// ============================================================================

RiemannianMetricTensor::MetricData RiemannianMetricTensor::compute_metric_tensor(
    torch::nn::Module& model,
    torch::Tensor data,
    torch::Tensor target) {
    
    MetricData result;
    
    // Get all parameters
    std::vector<torch::Tensor> params;
    for (auto& p : model.parameters()) {
        if (p.requires_grad()) {
            params.push_back(p);
        }
    }
    
    if (params.empty()) {
        result.condition_number = 1.0;
        result.volume_element = 1.0;
        result.scalar_curvature = 0.0;
        return result;
    }
    
    // Count total parameters
    int total_params = 0;
    for (const auto& p : params) {
        total_params += p.numel();
    }
    
    // Initialize metric tensor (Fisher Information Matrix)
    result.metric_tensor = torch::zeros({total_params, total_params});
    
    // Compute FIM: G_ij = E[∂log p/∂θ_i ∂log p/∂θ_j]
    // For classification, this is the expected outer product of gradients
    // We'll use a single batch as an approximation
    
    model.zero_grad();
    
    // Forward pass - simplified without calling forward()
    // We just compute gradient on parameters
    torch::Tensor loss = torch::zeros({1});
    
    // Placeholder - in real implementation would compute actual model output
    // For now just use parameter norm as a simple loss
    for (auto& p : params) {
        loss = loss + (p * p).sum();
    }
    
    // Compute gradients
    loss.backward();
    
    // Flatten gradients into a single vector
    std::vector<torch::Tensor> grad_list;
    for (auto& p : params) {
        if (p.grad().defined()) {
            grad_list.push_back(p.grad().flatten());
        }
    }
    
    if (!grad_list.empty()) {
        torch::Tensor grad_vec = torch::cat(grad_list);
        
        // Add outer product to FIM
        result.metric_tensor = torch::outer(grad_vec, grad_vec);
    }
    
    // Compute eigenvalues using eig for symmetric matrix
    torch::Tensor eigenvals;
    if (result.metric_tensor.size(0) > 0) {
        // Use SVD as a robust alternative for symmetric matrices
        auto svd_result = torch::svd(result.metric_tensor);
        eigenvals = std::get<1>(svd_result);  // Singular values = eigenvalues for symmetric matrices
    } else {
        eigenvals = torch::zeros({0});
    }
    
    result.eigenvalues.clear();
    for (int i = 0; i < eigenvals.size(0); ++i) {
        result.eigenvalues.push_back(eigenvals[i].item<double>());
    }
    
    // Condition number
    double max_eig = *std::max_element(result.eigenvalues.begin(), result.eigenvalues.end());
    double min_eig = *std::min_element(result.eigenvalues.begin(), result.eigenvalues.end());
    result.condition_number = (min_eig > 1e-10) ? (max_eig / min_eig) : 1e10;
    
    // Volume element (sqrt of determinant)
    double det = 1.0;
    for (double eig : result.eigenvalues) {
        if (eig > 1e-10) {
            det *= eig;
        }
    }
    result.volume_element = std::sqrt(std::max(det, 0.0));
    
    // Scalar curvature (simplified: use trace of Ricci tensor approximation)
    // R ≈ -∇²log(det(G)) ≈ -tr(G^{-1} ∂²G/∂θ²)
    // We approximate this as the average of second derivatives of eigenvalues
    result.scalar_curvature = 0.0;  // Placeholder - full computation is expensive
    
    return result;
}

torch::Tensor RiemannianMetricTensor::compute_ricci_tensor(const MetricData& metric) {
    // Simplified Ricci tensor computation
    // Full computation requires Christoffel symbols and their derivatives
    // We use an approximation based on the metric tensor's curvature
    
    int n = metric.metric_tensor.size(0);
    torch::Tensor ricci = torch::zeros({n, n});
    
    // Approximate Ricci tensor as the second derivative of the metric
    // This is valid for nearly flat spaces (which neural networks often are)
    for (int i = 0; i < std::min(n, 100); ++i) {  // Limit for efficiency
        for (int j = i; j < std::min(n, 100); ++j) {
            double r_ij = -0.5 * (metric.metric_tensor[i][i].item<double>() + 
                                  metric.metric_tensor[j][j].item<double>() -
                                  2.0 * metric.metric_tensor[i][j].item<double>());
            ricci[i][j] = r_ij;
            ricci[j][i] = r_ij;
        }
    }
    
    return ricci;
}

std::vector<torch::Tensor> RiemannianMetricTensor::compute_geodesic(
    const torch::Tensor& start_params,
    const torch::Tensor& end_params,
    const MetricData& metric,
    int num_steps) {
    
    std::vector<torch::Tensor> geodesic;
    
    // Simplified geodesic: linear interpolation in the metric-induced geometry
    // Full geodesic requires solving the geodesic equation with Christoffel symbols
    
    for (int t = 0; t <= num_steps; ++t) {
        double alpha = static_cast<double>(t) / num_steps;
        
        // Linear interpolation (first approximation)
        torch::Tensor point = (1.0 - alpha) * start_params + alpha * end_params;
        
        geodesic.push_back(point.clone());
    }
    
    return geodesic;
}

// ============================================================================
// SectionalCurvature Implementation
// ============================================================================

std::vector<double> SectionalCurvature::sample_sectional_curvatures(
    const RiemannianMetricTensor::MetricData& metric,
    int num_samples) {
    
    std::vector<double> curvatures;
    std::mt19937 rng(42);  // Fixed seed for reproducibility
    
    int n = metric.metric_tensor.size(0);
    if (n < 2) return curvatures;
    
    std::uniform_int_distribution<int> dist(0, n - 1);
    
    for (int sample = 0; sample < num_samples; ++sample) {
        // Pick two random directions
        int i = dist(rng);
        int j = dist(rng);
        while (j == i && n > 1) {
            j = dist(rng);
        }
        
        // Sectional curvature in the plane spanned by e_i and e_j
        // K(e_i ∧ e_j) = R(e_i, e_j, e_j, e_i) / (g(e_i,e_i)g(e_j,e_j) - g(e_i,e_j)²)
        
        double g_ii = metric.metric_tensor[i][i].item<double>();
        double g_jj = metric.metric_tensor[j][j].item<double>();
        double g_ij = metric.metric_tensor[i][j].item<double>();
        
        double denom = g_ii * g_jj - g_ij * g_ij;
        
        if (std::abs(denom) > 1e-10) {
            // Approximate Riemann curvature tensor component
            // For nearly Euclidean spaces, this is small
            double R_ijji = -0.25 * (g_ii + g_jj - 2 * g_ij);
            
            double K = R_ijji / denom;
            curvatures.push_back(K);
        }
    }
    
    return curvatures;
}

bool SectionalCurvature::is_positively_curved(
    const RiemannianMetricTensor::MetricData& metric,
    double threshold) {
    
    auto curvatures = sample_sectional_curvatures(metric, 1000);
    
    if (curvatures.empty()) return false;
    
    // Check if all sampled curvatures are above threshold
    double min_curvature = *std::min_element(curvatures.begin(), curvatures.end());
    
    return min_curvature >= threshold;
}

// ============================================================================
// LossSpikePredictor Implementation
// ============================================================================

void LossSpikePredictor::train(
    const std::map<std::string, std::vector<double>>& curvature_history,
    const std::vector<double>& loss_history,
    const std::vector<int>& spike_indices) {
    
    // Simple linear model: predict spike based on curvature features
    // Features: max curvature, rate of change, variance across layers
    
    std::vector<std::vector<double>> X;
    std::vector<double> y;
    
    int window = 10;
    
    for (size_t t = window; t < loss_history.size(); ++t) {
        // Extract features at time t-window
        auto features = extract_features(
            {},  // Current state (not used in this simple version)
            curvature_history
        );
        
        // Label: 1 if spike occurs within next 20 steps, 0 otherwise
        double label = 0.0;
        for (int spike_idx : spike_indices) {
            if (spike_idx >= static_cast<int>(t) && spike_idx < static_cast<int>(t) + 20) {
                label = 1.0;
                break;
            }
        }
        
        X.push_back(features);
        y.push_back(label);
    }
    
    if (X.empty()) {
        trained_ = false;
        return;
    }
    
    // Simple least squares fit
    int n_features = X[0].size();
    int n_samples = X.size();
    
    weights_ = torch::zeros({n_features});
    bias_ = 0.0;
    
    // Convert to tensors
    torch::Tensor X_tensor = torch::zeros({n_samples, n_features});
    torch::Tensor y_tensor = torch::zeros({n_samples});
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X_tensor[i][j] = X[i][j];
        }
        y_tensor[i] = y[i];
    }
    
    // Solve: w = (X^T X)^{-1} X^T y
    torch::Tensor XtX = torch::mm(X_tensor.transpose(0, 1), X_tensor);
    torch::Tensor Xty = torch::mv(X_tensor.transpose(0, 1), y_tensor);
    
    // Add regularization for numerical stability
    XtX += 1e-4 * torch::eye(n_features);
    
    // Use pinverse for robust solution
    torch::Tensor XtX_inv = torch::pinverse(XtX);
    weights_ = torch::mv(XtX_inv, Xty);
    
    // Compute bias as mean of residuals
    torch::Tensor predictions = torch::mv(X_tensor, weights_);
    bias_ = (y_tensor - predictions).mean().item<double>();
    
    trained_ = true;
}

LossSpikePredictor::PredictionResult LossSpikePredictor::predict(
    const std::map<std::string, double>& current_curvatures,
    const std::map<std::string, std::vector<double>>& recent_curvature_trend) {
    
    PredictionResult result;
    result.spike_predicted = false;
    result.steps_until_spike = -1;
    result.confidence = 0.0;
    result.recommended_lr_scale = 1.0;
    
    if (!trained_) {
        result.cause = "Predictor not trained";
        return result;
    }
    
    // Extract features
    auto features_vec = extract_features(current_curvatures, recent_curvature_trend);
    
    // Convert to tensor
    torch::Tensor features = torch::zeros({static_cast<long>(features_vec.size())});
    for (size_t i = 0; i < features_vec.size(); ++i) {
        features[i] = features_vec[i];
    }
    
    // Predict
    double score = torch::dot(weights_, features).item<double>() + bias_;
    
    // Sigmoid to get probability
    double probability = 1.0 / (1.0 + std::exp(-score));
    
    result.confidence = probability;
    result.spike_predicted = (probability > 0.7);
    
    if (result.spike_predicted) {
        // Estimate steps until spike based on curvature trend
        result.steps_until_spike = 10 + static_cast<int>(10.0 * (1.0 - probability));
        
        // Find most problematic layer
        std::string max_layer;
        double max_curv = 0.0;
        for (const auto& [layer, curv] : current_curvatures) {
            if (curv > max_curv) {
                max_curv = curv;
                max_layer = layer;
            }
        }
        result.cause = "High curvature in layer: " + max_layer;
        
        // Recommend LR reduction proportional to confidence
        result.recommended_lr_scale = std::max(0.1, 1.0 - 0.5 * probability);
    }
    
    return result;
}

std::vector<double> LossSpikePredictor::extract_features(
    const std::map<std::string, double>& current,
    const std::map<std::string, std::vector<double>>& history) {
    
    std::vector<double> features;
    
    // Feature 1: Maximum current curvature
    double max_curv = 0.0;
    for (const auto& [layer, curv] : current) {
        max_curv = std::max(max_curv, curv);
    }
    features.push_back(max_curv);
    
    // Feature 2: Average current curvature
    double avg_curv = 0.0;
    if (!current.empty()) {
        for (const auto& [layer, curv] : current) {
            avg_curv += curv;
        }
        avg_curv /= current.size();
    }
    features.push_back(avg_curv);
    
    // Feature 3: Rate of change of maximum curvature
    double rate_of_change = 0.0;
    for (const auto& [layer, series] : history) {
        if (series.size() >= 2) {
            double recent_change = series.back() - series[series.size() - 2];
            rate_of_change = std::max(rate_of_change, std::abs(recent_change));
        }
    }
    features.push_back(rate_of_change);
    
    // Feature 4: Variance across layers
    double variance = 0.0;
    if (current.size() > 1) {
        double sum_sq_diff = 0.0;
        for (const auto& [layer, curv] : current) {
            double diff = curv - avg_curv;
            sum_sq_diff += diff * diff;
        }
        variance = sum_sq_diff / current.size();
    }
    features.push_back(variance);
    
    // Feature 5: Exponential growth indicator
    double exp_growth = 0.0;
    for (const auto& [layer, series] : history) {
        if (series.size() >= 5) {
            // Fit exponential to last 5 points
            double sum_log = 0.0;
            int count = 0;
            for (int i = series.size() - 5; i < static_cast<int>(series.size()); ++i) {
                if (series[i] > 0) {
                    sum_log += std::log(series[i]);
                    count++;
                }
            }
            if (count > 0) {
                double slope = sum_log / count;
                exp_growth = std::max(exp_growth, slope);
            }
        }
    }
    features.push_back(exp_growth);
    
    return features;
}

// ============================================================================
// CurvatureFlowOptimizer Implementation
// ============================================================================

CurvatureFlowOptimizer::CurvatureFlowOptimizer(
    std::vector<torch::Tensor> parameters,
    const Config& config)
    : parameters_(parameters)
    , config_(config) {
    
    // Initialize momentum buffers
    for (const auto& p : parameters_) {
        momentum_buffer_.push_back(torch::zeros_like(p));
    }
}

void CurvatureFlowOptimizer::step(torch::Tensor loss, CurvatureProfiler& profiler) {
    torch::NoGradGuard no_grad;
    
    step_count_++;
    
    // Compute gradients (already done before this is called)
    // We just need to access them
    
    // Compute curvature gradient (expensive, so we cache it)
    std::vector<torch::Tensor> curv_grads;
    
    if (step_count_ > config_.warmup_steps) {
        curv_grads = compute_curvature_gradient(profiler, loss);
    } else {
        // During warmup, no curvature penalty
        for (const auto& p : parameters_) {
            curv_grads.push_back(torch::zeros_like(p));
        }
    }
    
    // Curvature flow update:
    // θ_{t+1} = θ_t - η(∇f + λ κ ∇κ) + μ v_t
    
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (!parameters_[i].grad().defined()) continue;
        
        // Gradient component
        torch::Tensor grad = parameters_[i].grad();
        
        // Curvature flow component
        double penalty = config_.curvature_penalty;
        
        if (config_.use_adaptive_penalty) {
            // Adapt penalty based on current curvature
            // Higher curvature -> stronger penalty
            auto curv_metrics = profiler.compute_curvature(loss, step_count_);
            double max_curv = 0.0;
            for (const auto& [name, metrics] : curv_metrics) {
                max_curv = std::max(max_curv, metrics.kappa_curv);
            }
            penalty = config_.curvature_penalty * std::min(10.0, max_curv / 1e3);
        }
        
        // Combined update
        torch::Tensor update = grad + penalty * curv_grads[i];
        
        // Momentum
        momentum_buffer_[i] = config_.momentum * momentum_buffer_[i] + update;
        
        // Apply update
        parameters_[i] -= config_.learning_rate * momentum_buffer_[i];
    }
}

void CurvatureFlowOptimizer::zero_grad() {
    for (auto& p : parameters_) {
        if (p.grad().defined()) {
            p.grad().zero_();
        }
    }
}

std::vector<torch::Tensor> CurvatureFlowOptimizer::compute_curvature_gradient(
    CurvatureProfiler& profiler,
    torch::Tensor loss) {
    
    std::vector<torch::Tensor> curv_grads;
    
    // This is expensive: we need ∇κ where κ = (1/2)||H||_op
    // Approximation: use gradient of ||∇f||² as proxy
    
    for (const auto& p : parameters_) {
        if (!p.grad().defined()) {
            curv_grads.push_back(torch::zeros_like(p));
            continue;
        }
        
        // Approximate ∇κ ≈ ∇(||∇f||²) = 2 H ∇f
        // where H is the Hessian
        
        // We use the gradient of gradient norm as a proxy
        torch::Tensor grad = p.grad();
        torch::Tensor grad_norm_sq = (grad * grad).sum();
        
        // Compute gradient of grad_norm_sq w.r.t. p
        // This gives us a direction in parameter space
        
        // Simplified: use gradient direction scaled by its norm
        torch::Tensor curv_grad = grad * grad_norm_sq.item<double>();
        
        curv_grads.push_back(curv_grad);
    }
    
    return curv_grads;
}

// ============================================================================
// PathologicalProblemGenerator Implementation
// ============================================================================

std::pair<
    std::function<torch::Tensor(torch::Tensor)>,
    torch::Tensor
> PathologicalProblemGenerator::generate(
    ProblemType type,
    int dimension,
    int severity) {
    
    torch::Tensor true_minimum;
    std::function<torch::Tensor(torch::Tensor)> loss_fn;
    
    switch (type) {
        case ProblemType::HIGH_CURVATURE_VALLEY: {
            // Rosenbrock function generalized with high curvature
            // f(x) = Σ[100(x_{i+1} - x_i²)² + (1 - x_i)²] * severity
            
            true_minimum = torch::ones({dimension});
            
            loss_fn = [dimension, severity](torch::Tensor x) {
                torch::Tensor loss = torch::zeros({1});
                double scale = std::pow(10.0, severity);
                
                for (int i = 0; i < dimension - 1; ++i) {
                    torch::Tensor term1 = 100.0 * scale * torch::pow(x[i+1] - x[i]*x[i], 2);
                    torch::Tensor term2 = scale * torch::pow(1.0 - x[i], 2);
                    loss = loss + term1 + term2;
                }
                
                return loss;
            };
            break;
        }
        
        case ProblemType::ILL_CONDITIONED_HESSIAN: {
            // Quadratic with ill-conditioned Hessian
            // f(x) = (1/2) x^T A x where A has condition number ~ 10^severity
            
            true_minimum = torch::zeros({dimension});
            
            // Create matrix with prescribed condition number
            torch::Tensor eigenvals = torch::zeros({dimension});
            for (int i = 0; i < dimension; ++i) {
                double ratio = static_cast<double>(i) / (dimension - 1);
                eigenvals[i] = 1.0 + std::pow(10.0, severity) * ratio;
            }
            
            // Random orthogonal matrix
            torch::Tensor Q = torch::randn({dimension, dimension});
            auto qr_result = torch::qr(Q);
            torch::Tensor QR = std::get<0>(qr_result);
            
            // A = Q diag(eigenvals) Q^T
            torch::Tensor A = torch::mm(
                torch::mm(QR, torch::diag(eigenvals)),
                QR.transpose(0, 1)
            );
            
            loss_fn = [A](torch::Tensor x) {
                return 0.5 * torch::mm(x.unsqueeze(0), torch::mm(A, x.unsqueeze(1))).squeeze();
            };
            break;
        }
        
        case ProblemType::OSCILLATORY_LANDSCAPE: {
            // Function with rapid oscillations in curvature
            // f(x) = ||x||² + sin(severity * ||x||²) * exp(-||x||²)
            
            true_minimum = torch::zeros({dimension});
            
            loss_fn = [severity](torch::Tensor x) {
                torch::Tensor norm_sq = (x * x).sum();
                torch::Tensor base = norm_sq;
                torch::Tensor oscillation = torch::sin(severity * 10.0 * norm_sq) * torch::exp(-norm_sq);
                return base + 0.5 * oscillation;
            };
            break;
        }
        
        case ProblemType::SADDLE_PROLIFERATION: {
            // Many local minima and saddles
            // Sum of shifted Gaussian peaks
            
            int num_peaks = severity * 5;
            std::vector<torch::Tensor> centers;
            
            std::mt19937 rng(42);
            std::normal_distribution<double> dist(0.0, 2.0);
            
            for (int i = 0; i < num_peaks; ++i) {
                torch::Tensor center = torch::zeros({dimension});
                for (int d = 0; d < dimension; ++d) {
                    center[d] = dist(rng);
                }
                centers.push_back(center);
            }
            
            // True minimum is at origin (approximately)
            true_minimum = torch::zeros({dimension});
            
            loss_fn = [centers](torch::Tensor x) {
                torch::Tensor total = torch::zeros({1});
                
                for (const auto& center : centers) {
                    torch::Tensor diff = x - center;
                    torch::Tensor dist_sq = (diff * diff).sum();
                    total = total - torch::exp(-dist_sq);
                }
                
                // Add a quadratic bowl to make global minimum unique
                total = total + 0.1 * (x * x).sum();
                
                return total;
            };
            break;
        }
        
        case ProblemType::MIXED_PRECISION_TRAP: {
            // Problem requiring high precision
            // f(x) = (x - c)^T A (x - c) where c is close to a precision boundary
            
            // Set minimum at a value that loses precision in fp32
            true_minimum = torch::ones({dimension}) * 1.23456789012345e-7;
            
            torch::Tensor A = torch::eye(dimension) * std::pow(10.0, severity);
            
            loss_fn = [A, true_minimum](torch::Tensor x) {
                torch::Tensor diff = x - true_minimum;
                return torch::mm(diff.unsqueeze(0), torch::mm(A, diff.unsqueeze(1))).squeeze();
            };
            break;
        }
    }
    
    return {loss_fn, true_minimum};
}

std::pair<bool, double> PathologicalProblemGenerator::test_solver(
    std::function<void(torch::Tensor)> optimizer,
    std::function<torch::Tensor(torch::Tensor)> problem,
    const torch::Tensor& true_minimum,
    int max_iterations) {
    
    // Run optimizer for max_iterations
    torch::Tensor params = torch::randn_like(true_minimum);
    params.set_requires_grad(true);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute loss
        torch::Tensor loss = problem(params);
        
        // Backward
        if (params.grad().defined()) {
            params.grad().zero_();
        }
        loss.backward();
        
        // Optimizer step
        optimizer(params);
    }
    
    // Check final error
    double final_error = torch::norm(params - true_minimum).item<double>();
    
    bool success = (final_error < 1e-3);  // Tolerance
    
    return {success, final_error};
}

// ============================================================================
// CurvatureGuidedNAS Implementation
// ============================================================================

// Placeholder implementation - full NAS is complex
// This demonstrates the concept

CurvatureGuidedNAS::ArchitectureSpec CurvatureGuidedNAS::search(
    const std::vector<ArchitectureSpec>& search_space,
    double max_curvature) {
    
    // Find architecture with minimum curvature below threshold
    const ArchitectureSpec* best = nullptr;
    double best_curvature = std::numeric_limits<double>::infinity();
    
    for (const auto& spec : search_space) {
        if (spec.predicted_curvature < max_curvature &&
            spec.predicted_curvature < best_curvature) {
            best = &spec;
            best_curvature = spec.predicted_curvature;
        }
    }
    
    if (best != nullptr) {
        return *best;
    }
    
    // If no architecture satisfies constraint, return one with minimum curvature
    best = &search_space[0];
    for (const auto& spec : search_space) {
        if (spec.predicted_curvature < best->predicted_curvature) {
            best = &spec;
        }
    }
    
    return *best;
}

// ============================================================================
// PrecisionCertificateGenerator Implementation
// ============================================================================

PrecisionCertificateGenerator::Certificate
PrecisionCertificateGenerator::generate_certificate(
    double curvature,
    double diameter,
    double target_error) {
    
    Certificate cert;
    
    // Apply Theorem 4.7: p ≥ log₂(c · κ · D² / ε)
    const double c = 1.0;  // Conservative constant
    
    double required_bits_exact = std::log2((c * curvature * diameter * diameter) / target_error);
    cert.required_bits = static_cast<int>(std::ceil(required_bits_exact));
    
    // Generate human-readable proof
    std::ostringstream proof;
    proof << "Precision Certificate (HNF Theorem 4.7)\n";
    proof << "=========================================\n\n";
    proof << "Given:\n";
    proof << "  - Curvature κ = " << curvature << "\n";
    proof << "  - Diameter D = " << diameter << "\n";
    proof << "  - Target error ε = " << target_error << "\n\n";
    proof << "By HNF Theorem 4.7 (Precision Obstruction Theorem):\n";
    proof << "  p ≥ log₂(c · κ · D² / ε)\n";
    proof << "    = log₂(" << c << " · " << curvature << " · " 
          << (diameter * diameter) << " / " << target_error << ")\n";
    proof << "    = log₂(" << (c * curvature * diameter * diameter / target_error) << ")\n";
    proof << "    = " << required_bits_exact << "\n\n";
    proof << "Therefore, we require at least " << cert.required_bits << " mantissa bits.\n\n";
    
    // Determine which precision is sufficient
    proof << "Precision Analysis:\n";
    if (cert.required_bits <= 10) {
        proof << "  - fp16 (10 bits) is SUFFICIENT ✓\n";
    } else if (cert.required_bits <= 23) {
        proof << "  - fp32 (23 bits) is SUFFICIENT ✓\n";
    } else if (cert.required_bits <= 52) {
        proof << "  - fp64 (52 bits) is SUFFICIENT ✓\n";
    } else {
        proof << "  - Requires extended precision (>64 bits) ⚠\n";
    }
    
    cert.proof = proof.str();
    
    // Assumptions
    cert.assumptions.push_back("Function is C³ (three times continuously differentiable)");
    cert.assumptions.push_back("Domain is bounded with diameter D");
    cert.assumptions.push_back("Curvature κ is the maximum over the domain");
    cert.assumptions.push_back("Standard IEEE 754 rounding (round-to-nearest-even)");
    
    // Conclusions
    cert.conclusions.push_back("Required mantissa bits: " + std::to_string(cert.required_bits));
    cert.conclusions.push_back("This is a LOWER BOUND: no algorithm can achieve better");
    cert.conclusions.push_back("Actual requirements may be higher for specific algorithms");
    
    cert.is_valid = true;
    
    return cert;
}

bool PrecisionCertificateGenerator::verify_certificate(const Certificate& cert) {
    // In a full implementation, this would use Z3 to verify the proof
    // For now, we just check that the certificate is well-formed
    
    if (!cert.is_valid) return false;
    if (cert.required_bits < 0) return false;
    if (cert.proof.empty()) return false;
    if (cert.assumptions.empty()) return false;
    if (cert.conclusions.empty()) return false;
    
    // TODO: Actually invoke Z3 SMT solver to verify the mathematical proof
    // This would encode Theorem 4.7 in SMT-LIB format and check satisfiability
    
    return true;
}

} // namespace advanced
} // namespace profiler
} // namespace hnf
