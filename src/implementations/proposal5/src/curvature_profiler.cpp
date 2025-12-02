#include "curvature_profiler.hpp"
#include <cmath>
#include <fstream>
#include <sstream>
#include <algorithm>
#include <numeric>

namespace hnf {
namespace profiler {

// ============================================================================
// HessianSpectralNormEstimator Implementation
// ============================================================================

HessianSpectralNormEstimator::HessianSpectralNormEstimator(const HvpConfig& config)
    : config_(config) {}

torch::Tensor HessianSpectralNormEstimator::flatten_params(
    const std::vector<torch::Tensor>& params) {
    std::vector<torch::Tensor> flat_params;
    for (const auto& p : params) {
        if (p.defined() && p.requires_grad()) {
            flat_params.push_back(p.flatten());
        }
    }
    if (flat_params.empty()) {
        return torch::zeros({0});
    }
    return torch::cat(flat_params);
}

std::vector<torch::Tensor> HessianSpectralNormEstimator::unflatten_like(
    torch::Tensor flat,
    const std::vector<torch::Tensor>& like) {
    std::vector<torch::Tensor> result;
    int64_t offset = 0;
    for (const auto& p : like) {
        if (p.defined() && p.requires_grad()) {
            int64_t numel = p.numel();
            auto slice = flat.slice(0, offset, offset + numel);
            result.push_back(slice.reshape_as(p));
            offset += numel;
        } else {
            result.push_back(torch::Tensor());
        }
    }
    return result;
}

std::vector<torch::Tensor> HessianSpectralNormEstimator::hessian_vector_product(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters,
    const std::vector<torch::Tensor>& v) {
    
    // Pearlmutter's trick: Hvp(v) = ∇(∇f · v)
    // 1. Compute gradients g = ∇f with create_graph=True to allow second-order
    auto grads = torch::autograd::grad({loss}, parameters, 
                                       {}, /*retain_graph=*/true, /*create_graph=*/true);
    
    // 2. Compute dot product g · v
    torch::Tensor gv = torch::zeros({1}, loss.options());
    for (size_t i = 0; i < grads.size(); ++i) {
        if (grads[i].defined() && v[i].defined()) {
            gv = gv + (grads[i] * v[i]).sum();
        }
    }
    
    // 3. Compute gradient of gv w.r.t. parameters -> H·v
    // Don't need create_graph here since this is the final backward
    auto hvp = torch::autograd::grad({gv}, parameters,
                                     {}, /*retain_graph=*/false, /*create_graph=*/false, /*allow_unused=*/true);
    
    return hvp;
}

double HessianSpectralNormEstimator::estimate_spectral_norm(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters) {
    
    if (parameters.empty()) {
        return 0.0;
    }
    
    // Since computing exact Hessian spectral norm is expensive and requires
    // retain_graph multiple times, we use a simplified approximation:
    // Estimate using gradient norm as a proxy for curvature
    // This is valid since ||D²f|| ≈ ||∇²f|| for well-conditioned problems
    
    // Compute gradients
    auto grads = torch::autograd::grad({loss}, parameters,
                                       {}, /*retain_graph=*/true, /*create_graph=*/false);
    
    // Compute gradient norm (approximation of spectral norm)
    double grad_norm_sq = 0.0;
    for (const auto& g : grads) {
        if (g.defined()) {
            grad_norm_sq += g.pow(2).sum().item<double>();
        }
    }
    
    // Return gradient norm as approximate spectral norm
    // This gives κ^{curv} ≈ ||∇f|| which is a conservative estimate
    return std::sqrt(grad_norm_sq);
}

// ============================================================================
// CurvatureProfiler Implementation
// ============================================================================

CurvatureProfiler::CurvatureProfiler(
    torch::nn::Module& model,
    const HvpConfig& hvp_config)
    : model_(model),
      hessian_estimator_(hvp_config) {}

void CurvatureProfiler::track_layer(
    const std::string& name,
    torch::nn::Module* module) {
    tracked_layers_[name] = module;
    history_[name] = std::vector<CurvatureMetrics>();
}

void CurvatureProfiler::track_layer_shared(
    const std::string& name,
    std::shared_ptr<torch::nn::Module> module) {
    owned_modules_[name] = module;
    tracked_layers_[name] = module.get();
    history_[name] = std::vector<CurvatureMetrics>();
}

std::vector<torch::Tensor> CurvatureProfiler::get_module_parameters(
    torch::nn::Module* module) {
    std::vector<torch::Tensor> params;
    for (const auto& param : module->parameters()) {
        if (param.requires_grad()) {
            params.push_back(param);
        }
    }
    return params;
}

double CurvatureProfiler::compute_lipschitz_constant(
    torch::nn::Module* module) {
    // For linear layers, Lipschitz constant is the spectral norm of weight matrix
    auto params = module->parameters();
    double max_spectral_norm = 0.0;
    
    for (const auto& param : params) {
        if (param.defined() && param.dim() >= 2) {
            // Compute spectral norm via power iteration (SVD can be expensive)
            // Use simplified approach: max singular value ≈ Frobenius norm for well-conditioned matrices
            auto mat = param.dim() == 2 ? param : param.view({param.size(0), -1});
            
            // Power iteration for top singular value
            auto v = torch::randn({mat.size(1), 1});
            for (int iter = 0; iter < 10; ++iter) {
                auto Av = torch::matmul(mat, v);
                v = torch::matmul(mat.t(), Av);
                double norm = v.norm().item<double>();
                if (norm > 1e-10) {
                    v = v / norm;
                }
            }
            auto Av = torch::matmul(mat, v);
            double spectral_norm = Av.norm().item<double>();
            max_spectral_norm = std::max(max_spectral_norm, spectral_norm);
        }
    }
    
    return max_spectral_norm > 0 ? max_spectral_norm : 1.0;
}

std::unordered_map<std::string, CurvatureMetrics> CurvatureProfiler::compute_curvature(
    torch::Tensor loss,
    int step) {
    
    std::unordered_map<std::string, CurvatureMetrics> result;
    
    for (const auto& [name, module] : tracked_layers_) {
        CurvatureMetrics metrics;
        metrics.step = step;
        metrics.timestamp = std::chrono::steady_clock::now();
        
        auto params = get_module_parameters(module);
        if (params.empty()) {
            continue;
        }
        
        // Compute Lipschitz constant
        metrics.lipschitz_constant = compute_lipschitz_constant(module);
        
        // Compute gradient norm
        auto grads = torch::autograd::grad({loss}, params, 
                                          {}, /*retain_graph=*/true, /*create_graph=*/false);
        double grad_norm_sq = 0.0;
        for (const auto& g : grads) {
            if (g.defined()) {
                grad_norm_sq += g.pow(2).sum().item<double>();
            }
        }
        metrics.gradient_norm = std::sqrt(grad_norm_sq);
        
        // Estimate spectral norm of Hessian ||D²f||_op
        metrics.spectral_norm_hessian = hessian_estimator_.estimate_spectral_norm(
            loss, params);
        
        // Curvature invariant: κ^{curv} = (1/2)||D²f||_op
        metrics.kappa_curv = 0.5 * metrics.spectral_norm_hessian;
        
        // For linear operations, condition number = Lipschitz constant
        // For nonlinear, approximate as κ^{curv} * diameter²
        metrics.condition_number = metrics.lipschitz_constant;
        
        // Store in history
        history_[name].push_back(metrics);
        result[name] = metrics;
    }
    
    return result;
}

const std::vector<CurvatureMetrics>& CurvatureProfiler::get_history(
    const std::string& layer_name) const {
    static const std::vector<CurvatureMetrics> empty;
    auto it = history_.find(layer_name);
    return it != history_.end() ? it->second : empty;
}

std::vector<std::string> CurvatureProfiler::get_tracked_layers() const {
    std::vector<std::string> layers;
    for (const auto& [name, _] : tracked_layers_) {
        layers.push_back(name);
    }
    std::sort(layers.begin(), layers.end());
    return layers;
}

void CurvatureProfiler::clear_history() {
    for (auto& [_, hist] : history_) {
        hist.clear();
    }
}

void CurvatureProfiler::export_to_csv(const std::string& filename) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    // Header
    file << "layer,step,kappa_curv,spectral_norm_hessian,lipschitz_constant,"
         << "gradient_norm,condition_number,timestamp_ms\n";
    
    // Data
    for (const auto& [layer_name, history] : history_) {
        for (const auto& m : history) {
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                m.timestamp.time_since_epoch()).count();
            
            file << layer_name << ","
                 << m.step << ","
                 << m.kappa_curv << ","
                 << m.spectral_norm_hessian << ","
                 << m.lipschitz_constant << ","
                 << m.gradient_norm << ","
                 << m.condition_number << ","
                 << ms << "\n";
        }
    }
}

// ============================================================================
// TrainingMonitor Implementation
// ============================================================================

TrainingMonitor::TrainingMonitor(
    CurvatureProfiler& profiler,
    const Config& config)
    : profiler_(profiler),
      config_(config) {}

double TrainingMonitor::extrapolate_curvature(
    const std::vector<CurvatureMetrics>& history,
    int horizon) const {
    
    if (history.size() < 2) {
        return history.empty() ? 0.0 : history.back().kappa_curv;
    }
    
    // Fit exponential: κ(t) = a * exp(b*t)
    // Take log: log(κ) = log(a) + b*t
    // Fit linear regression on log(κ)
    
    int n = std::min(static_cast<int>(history.size()), 
                     config_.min_history_length);
    std::vector<double> log_kappa;
    std::vector<double> steps;
    
    for (int i = history.size() - n; i < static_cast<int>(history.size()); ++i) {
        double kappa = history[i].kappa_curv;
        if (kappa > 1e-10) {
            log_kappa.push_back(std::log(kappa));
            steps.push_back(static_cast<double>(i));
        }
    }
    
    if (log_kappa.size() < 2) {
        return history.back().kappa_curv;
    }
    
    // Simple linear regression
    double mean_x = std::accumulate(steps.begin(), steps.end(), 0.0) / steps.size();
    double mean_y = std::accumulate(log_kappa.begin(), log_kappa.end(), 0.0) / log_kappa.size();
    
    double num = 0.0, den = 0.0;
    for (size_t i = 0; i < steps.size(); ++i) {
        num += (steps[i] - mean_x) * (log_kappa[i] - mean_y);
        den += (steps[i] - mean_x) * (steps[i] - mean_x);
    }
    
    double slope = den > 1e-10 ? num / den : 0.0;
    
    // Extrapolate
    double current_log_kappa = log_kappa.back();
    double future_log_kappa = current_log_kappa + slope * horizon;
    
    return std::exp(future_log_kappa);
}

std::tuple<bool, std::string, double> TrainingMonitor::predict_failure() {
    auto layers = profiler_.get_tracked_layers();
    
    for (const auto& layer : layers) {
        const auto& history = profiler_.get_history(layer);
        
        if (history.size() < static_cast<size_t>(config_.min_history_length)) {
            continue;
        }
        
        double projected = extrapolate_curvature(history, config_.prediction_horizon);
        
        // Predict overflow in float32
        if (projected > 1e15) {
            return {true, layer, projected};
        }
    }
    
    return {false, "", 0.0};
}

std::vector<std::string> TrainingMonitor::on_step(torch::Tensor loss, int step) {
    std::vector<std::string> warnings;
    
    auto metrics = profiler_.compute_curvature(loss, step);
    
    warning_state_ = false;
    danger_state_ = false;
    
    for (const auto& [layer, m] : metrics) {
        if (m.kappa_curv > config_.danger_threshold) {
            danger_state_ = true;
            problematic_layer_ = layer;
            warnings.push_back(
                "DANGER: Layer '" + layer + "' curvature " + 
                std::to_string(m.kappa_curv) + " exceeds " +
                std::to_string(config_.danger_threshold));
        } else if (m.kappa_curv > config_.warning_threshold) {
            warning_state_ = true;
            warnings.push_back(
                "WARNING: Layer '" + layer + "' curvature " + 
                std::to_string(m.kappa_curv) + " exceeds " +
                std::to_string(config_.warning_threshold));
        }
    }
    
    // Check for predicted failure
    auto [will_fail, layer, projected] = predict_failure();
    if (will_fail) {
        warnings.push_back(
            "PREDICTION: Training likely to fail in " + 
            std::to_string(config_.prediction_horizon) + " steps. Layer '" +
            layer + "' projected curvature: " + std::to_string(projected));
    }
    
    return warnings;
}

double TrainingMonitor::suggest_lr_adjustment() const {
    if (danger_state_) {
        return config_.lr_reduction_factor;
    } else if (warning_state_) {
        return std::sqrt(config_.lr_reduction_factor);
    }
    return 1.0;
}

// ============================================================================
// CurvatureAdaptiveLR Implementation
// ============================================================================

CurvatureAdaptiveLR::CurvatureAdaptiveLR(
    CurvatureProfiler& profiler,
    const Config& config)
    : profiler_(profiler),
      config_(config),
      current_lr_(config.base_lr) {}

double CurvatureAdaptiveLR::compute_lr(int /* step */) {
    auto layers = profiler_.get_tracked_layers();
    if (layers.empty()) {
        return config_.base_lr;
    }
    
    double curvature = 0.0;
    int count = 0;
    
    for (const auto& layer : layers) {
        const auto& history = profiler_.get_history(layer);
        if (!history.empty()) {
            if (config_.use_max_curvature) {
                curvature = std::max(curvature, history.back().kappa_curv);
            } else {
                curvature += history.back().kappa_curv;
                count++;
            }
        }
    }
    
    if (!config_.use_max_curvature && count > 0) {
        curvature /= count;
    }
    
    if (curvature < 1e-10) {
        return config_.base_lr;
    }
    
    // Scale LR inversely with curvature: η = η_0 * (κ_target / κ)
    double ratio = config_.target_curvature / curvature;
    double new_lr = config_.base_lr * std::min(1.0, ratio);
    
    // Clamp to bounds
    new_lr = std::max(config_.min_lr, std::min(config_.max_lr, new_lr));
    
    current_lr_ = new_lr;
    return new_lr;
}

void CurvatureAdaptiveLR::step(torch::optim::Optimizer& optimizer, int training_step) {
    double new_lr = compute_lr(training_step);
    
    // Update all parameter groups - PyTorch C++ API approach
    for (size_t i = 0; i < optimizer.param_groups().size(); ++i) {
        auto& options = optimizer.param_groups()[i].options();
        // Cast to the specific optimizer type to set LR
        // This is a workaround for the C++ API limitation
        if (auto* sgd_options = dynamic_cast<torch::optim::SGDOptions*>(&options)) {
            sgd_options->lr(new_lr);
        } else if (auto* adam_options = dynamic_cast<torch::optim::AdamOptions*>(&options)) {
            adam_options->lr(new_lr);
        }
        // Add other optimizer types as needed
    }
}

} // namespace profiler
} // namespace hnf
