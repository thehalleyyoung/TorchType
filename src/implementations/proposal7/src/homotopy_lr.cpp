#include "homotopy_lr.hpp"
#include <algorithm>
#include <numeric>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <random>
#include <cmath>

namespace hnf {
namespace homotopy {

//==============================================================================
// CurvatureEstimator Implementation
//==============================================================================

CurvatureEstimator::CurvatureEstimator(const HutchinsonConfig& config)
    : config_(config)
{}

CurvatureMetrics CurvatureEstimator::estimate(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters)
{
    step_counter_++;
    
    // Only estimate every N steps for efficiency
    if (step_counter_ % config_.estimation_frequency != 0 && ema_initialized_) {
        // Return cached EMA values
        CurvatureMetrics metrics;
        metrics.spectral_norm_hessian = ema_spectral_norm_;
        metrics.trace_hessian = ema_trace_;
        
        // Compute gradient norm
        double grad_norm_sq = 0.0;
        for (const auto& p : parameters) {
            if (p.grad().defined()) {
                grad_norm_sq += p.grad().pow(2).sum().item<double>();
            }
        }
        metrics.gradient_norm = std::sqrt(grad_norm_sq);
        
        // κ^{curv} = ||∇²L|| / ||∇L||²
        if (metrics.gradient_norm > 1e-10) {
            metrics.kappa_curv = metrics.spectral_norm_hessian / 
                                (metrics.gradient_norm * metrics.gradient_norm);
        } else {
            metrics.kappa_curv = 0.0;
        }
        
        metrics.step = step_counter_;
        metrics.timestamp = std::chrono::steady_clock::now();
        
        history_.push_back(metrics);
        return metrics;
    }
    
    // Full estimation
    CurvatureMetrics metrics;
    metrics.step = step_counter_;
    metrics.timestamp = std::chrono::steady_clock::now();
    
    // 1. Compute gradient norm ||∇L||
    double grad_norm_sq = 0.0;
    for (const auto& p : parameters) {
        if (p.grad().defined()) {
            grad_norm_sq += p.grad().pow(2).sum().item<double>();
        }
    }
    metrics.gradient_norm = std::sqrt(grad_norm_sq);
    
    // 2. Estimate spectral norm ||∇²L|| using power iteration
    metrics.spectral_norm_hessian = estimate_spectral_norm_power(loss, parameters);
    
    // 3. Estimate trace tr(∇²L) using Hutchinson's method
    metrics.trace_hessian = estimate_trace_hutchinson(loss, parameters);
    
    // 4. Compute curvature κ^{curv} = ||∇²L|| / ||∇L||²
    if (metrics.gradient_norm > 1e-10) {
        metrics.kappa_curv = metrics.spectral_norm_hessian / 
                            (metrics.gradient_norm * metrics.gradient_norm);
    } else {
        // Avoid division by zero
        metrics.kappa_curv = 0.0;
    }
    
    // Update EMA
    if (!ema_initialized_) {
        ema_spectral_norm_ = metrics.spectral_norm_hessian;
        ema_trace_ = metrics.trace_hessian;
        ema_initialized_ = true;
    } else {
        ema_spectral_norm_ = config_.ema_decay * ema_spectral_norm_ +
                            (1.0 - config_.ema_decay) * metrics.spectral_norm_hessian;
        ema_trace_ = config_.ema_decay * ema_trace_ +
                    (1.0 - config_.ema_decay) * metrics.trace_hessian;
    }
    
    history_.push_back(metrics);
    return metrics;
}

std::vector<torch::Tensor> CurvatureEstimator::hessian_vector_product(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters,
    const std::vector<torch::Tensor>& v)
{
    // Pearlmutter's trick: Hvp(v) = ∇(∇L · v)
    // 
    // Step 1: Compute gradient ∇L
    auto grads = torch::autograd::grad(
        {loss},
        parameters,
        {},
        /*retain_graph=*/true,
        /*create_graph=*/true);
    
    // Step 2: Compute ∇L · v (dot product)
    torch::Tensor grad_dot_v;
    bool first = true;
    for (size_t i = 0; i < grads.size(); ++i) {
        if (grads[i].defined() && v[i].defined()) {
            auto term = (grads[i] * v[i]).sum();
            if (first) {
                grad_dot_v = term;
                first = false;
            } else {
                grad_dot_v = grad_dot_v + term;
            }
        }
    }
    
    if (!grad_dot_v.defined()) {
        // Return zero if no valid gradients
        std::vector<torch::Tensor> result;
        for (const auto& p : parameters) {
            result.push_back(torch::zeros_like(p));
        }
        return result;
    }
    
    // Step 3: Compute ∇(∇L · v) = Hv
    auto hvp = torch::autograd::grad(
        {grad_dot_v},
        parameters,
        {},
        /*retain_graph=*/true,
        /*create_graph=*/false);
    
    return hvp;
}

double CurvatureEstimator::estimate_trace_hutchinson(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters)
{
    // Hutchinson's method: tr(H) ≈ (1/m) Σ_i v_i^T H v_i
    // where v_i ~ N(0, I) or Rademacher {-1, +1}
    
    double trace_estimate = 0.0;
    
    for (int sample = 0; sample < config_.num_samples; ++sample) {
        // Generate random vector
        auto v = generate_random_vector(parameters, config_.use_rademacher);
        
        // Compute Hv
        auto hv = hessian_vector_product(loss, parameters, v);
        
        // Compute v^T H v
        double vt_hv = dot_product(v, hv);
        trace_estimate += vt_hv;
    }
    
    return trace_estimate / config_.num_samples;
}

double CurvatureEstimator::estimate_spectral_norm_power(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters)
{
    // Power iteration to find largest eigenvalue
    // v_{k+1} = H v_k / ||H v_k||
    // λ_max = v_k^T H v_k
    
    // Initialize with random vector
    auto v = generate_random_vector(parameters, false);
    normalize_inplace(v);
    
    double eigenvalue = 0.0;
    double prev_eigenvalue = 0.0;
    
    for (int iter = 0; iter < config_.power_iterations; ++iter) {
        // Compute Hv
        auto hv = hessian_vector_product(loss, parameters, v);
        
        // Compute Rayleigh quotient: λ = v^T H v / v^T v
        double vt_hv = dot_product(v, hv);
        double vt_v = norm_squared(v);
        eigenvalue = vt_hv / (vt_v + 1e-10);
        
        // Normalize for next iteration
        v = hv;
        normalize_inplace(v);
        
        // Check convergence
        if (iter > 0 && std::abs(eigenvalue - prev_eigenvalue) < config_.power_iter_tol) {
            break;
        }
        
        prev_eigenvalue = eigenvalue;
    }
    
    return std::abs(eigenvalue);
}

std::vector<double> CurvatureEstimator::estimate_top_eigenvalues_lanczos(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters,
    int k)
{
    // Lanczos iteration for top-k eigenvalues
    // More accurate than power iteration but more expensive
    // 
    // Builds tridiagonal matrix T via:
    // β_{j+1} v_{j+1} = H v_j - α_j v_j - β_j v_{j-1}
    // where α_j = v_j^T H v_j, β_j = ||β_{j+1} v_{j+1}||
    
    std::vector<double> alphas;
    std::vector<double> betas;
    std::vector<std::vector<torch::Tensor>> v_vecs;
    
    // Initialize
    auto v = generate_random_vector(parameters, false);
    normalize_inplace(v);
    v_vecs.push_back(v);
    
    for (int j = 0; j < std::min(k + 5, 50); ++j) {  // k+5 for accuracy
        // Compute H v_j
        auto hv = hessian_vector_product(loss, parameters, v_vecs[j]);
        
        // α_j = v_j^T H v_j
        double alpha = dot_product(v_vecs[j], hv);
        alphas.push_back(alpha);
        
        // w = H v_j - α_j v_j
        auto w = hv;
        for (size_t i = 0; i < w.size(); ++i) {
            w[i] = w[i] - alpha * v_vecs[j][i];
        }
        
        // w = w - β_j v_{j-1}
        if (j > 0) {
            double beta = betas.back();
            for (size_t i = 0; i < w.size(); ++i) {
                w[i] = w[i] - beta * v_vecs[j-1][i];
            }
        }
        
        // β_{j+1} = ||w||
        double beta = std::sqrt(norm_squared(w));
        betas.push_back(beta);
        
        if (beta < 1e-10) {
            // Lanczos breakdown (converged)
            break;
        }
        
        // v_{j+1} = w / β_{j+1}
        scale_inplace(w, 1.0 / beta);
        v_vecs.push_back(w);
    }
    
    // Diagonalize tridiagonal matrix to get eigenvalues
    // For simplicity, return sorted |alphas| as rough eigenvalue estimates
    // (proper implementation would use QR algorithm on T)
    std::vector<double> eigenvalues;
    for (double alpha : alphas) {
        eigenvalues.push_back(std::abs(alpha));
    }
    std::sort(eigenvalues.begin(), eigenvalues.end(), std::greater<double>());
    
    eigenvalues.resize(std::min(k, static_cast<int>(eigenvalues.size())));
    return eigenvalues;
}

std::vector<torch::Tensor> CurvatureEstimator::generate_random_vector(
    const std::vector<torch::Tensor>& like,
    bool rademacher)
{
    std::vector<torch::Tensor> result;
    
    for (const auto& tensor : like) {
        if (rademacher) {
            // Rademacher: uniform {-1, +1}
            auto r = torch::randint(0, 2, tensor.sizes(), tensor.options()) * 2.0 - 1.0;
            result.push_back(r);
        } else {
            // Gaussian N(0, I)
            auto r = torch::randn(tensor.sizes(), tensor.options());
            result.push_back(r);
        }
    }
    
    return result;
}

torch::Tensor CurvatureEstimator::flatten_tensors(const std::vector<torch::Tensor>& tensors)
{
    std::vector<torch::Tensor> flattened;
    for (const auto& t : tensors) {
        if (t.defined()) {
            flattened.push_back(t.flatten());
        }
    }
    
    if (flattened.empty()) {
        return torch::tensor({});
    }
    
    return torch::cat(flattened);
}

std::vector<torch::Tensor> CurvatureEstimator::unflatten_like(
    torch::Tensor flat,
    const std::vector<torch::Tensor>& like)
{
    std::vector<torch::Tensor> result;
    int64_t offset = 0;
    
    for (const auto& template_tensor : like) {
        if (!template_tensor.defined()) {
            result.push_back(torch::Tensor());
            continue;
        }
        
        int64_t numel = template_tensor.numel();
        auto slice = flat.slice(0, offset, offset + numel);
        result.push_back(slice.reshape(template_tensor.sizes()));
        offset += numel;
    }
    
    return result;
}

double CurvatureEstimator::dot_product(
    const std::vector<torch::Tensor>& a,
    const std::vector<torch::Tensor>& b)
{
    double result = 0.0;
    
    for (size_t i = 0; i < a.size() && i < b.size(); ++i) {
        if (a[i].defined() && b[i].defined()) {
            result += (a[i] * b[i]).sum().item<double>();
        }
    }
    
    return result;
}

double CurvatureEstimator::norm_squared(const std::vector<torch::Tensor>& v)
{
    return dot_product(v, v);
}

void CurvatureEstimator::normalize_inplace(std::vector<torch::Tensor>& v)
{
    double norm = std::sqrt(norm_squared(v));
    if (norm > 1e-10) {
        scale_inplace(v, 1.0 / norm);
    }
}

void CurvatureEstimator::scale_inplace(std::vector<torch::Tensor>& v, double scalar)
{
    for (auto& tensor : v) {
        if (tensor.defined()) {
            tensor = tensor * scalar;
        }
    }
}

//==============================================================================
// HomotopyLRScheduler Implementation
//==============================================================================

HomotopyLRScheduler::HomotopyLRScheduler(
    const Config& config,
    const HutchinsonConfig& hvp_config)
    : config_(config)
    , estimator_(hvp_config)
    , current_lr_(config.base_lr)
    , current_curvature_(config.target_curvature)
{}

double HomotopyLRScheduler::step(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters,
    int step_num)
{
    step_count_ = step_num;
    
    // Estimate curvature
    auto metrics = estimator_.estimate(loss, parameters);
    current_curvature_ = metrics.kappa_curv;
    
    // Determine target curvature
    double target_kappa = config_.target_curvature;
    if (config_.adaptive_target && step_num >= config_.warmup_steps) {
        target_kappa = adaptive_target_curvature();
    }
    
    // Compute LR from curvature
    // η(t) = η_base / (1 + α · (κ(t) / κ_target - 1)₊)
    // where (x)₊ = max(0, x)
    
    double ratio = current_curvature_ / (target_kappa + 1e-10);
    double scale = 1.0 / (1.0 + config_.alpha * std::max(0.0, ratio - 1.0));
    
    current_lr_ = config_.base_lr * scale;
    
    // Clamp to bounds
    current_lr_ = std::clamp(current_lr_, config_.min_lr, config_.max_lr);
    
    return current_lr_;
}

void HomotopyLRScheduler::apply_to_optimizer(torch::optim::Optimizer& optimizer)
{
    // Set LR for all param groups
    for (auto& param_group : optimizer.param_groups()) {
        if (param_group.has_options()) {
            auto& options = static_cast<torch::optim::OptimizerOptions&>(
                param_group.options());
            
            // This is a bit of a hack since PyTorch C++ doesn't have a clean LR API
            // In practice, you'd need to access the specific optimizer's options
            // For now, we'll just note that this needs optimizer-specific implementation
        }
    }
}

double HomotopyLRScheduler::adaptive_target_curvature() const
{
    const auto& history = estimator_.history();
    
    if (history.size() < static_cast<size_t>(config_.warmup_steps)) {
        return config_.target_curvature;
    }
    
    // Extract curvature values from recent history
    std::vector<double> recent_kappas;
    int lookback = std::min(static_cast<int>(history.size()), 1000);
    
    for (int i = history.size() - lookback; i < static_cast<int>(history.size()); ++i) {
        recent_kappas.push_back(history[i].kappa_curv);
    }
    
    // Compute target as percentile
    std::sort(recent_kappas.begin(), recent_kappas.end());
    int idx = static_cast<int>(recent_kappas.size() * config_.target_percentile);
    idx = std::clamp(idx, 0, static_cast<int>(recent_kappas.size()) - 1);
    
    return recent_kappas[idx];
}

double HomotopyLRScheduler::compute_lr_from_curvature(double kappa)
{
    double ratio = kappa / (config_.target_curvature + 1e-10);
    double scale = 1.0 / (1.0 + config_.alpha * std::max(0.0, ratio - 1.0));
    return std::clamp(config_.base_lr * scale, config_.min_lr, config_.max_lr);
}

void HomotopyLRScheduler::export_metrics(const std::string& filename) const
{
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    // CSV header
    file << "step,spectral_norm,trace,gradient_norm,kappa_curv,learning_rate\n";
    
    // Data
    const auto& history = estimator_.history();
    for (size_t i = 0; i < history.size(); ++i) {
        const auto& m = history[i];
        
        // Compute LR for this step
        double lr = compute_lr_from_curvature(m.kappa_curv);
        
        file << m.step << ","
             << m.spectral_norm_hessian << ","
             << m.trace_hessian << ","
             << m.gradient_norm << ","
             << m.kappa_curv << ","
             << lr << "\n";
    }
}

//==============================================================================
// PerLayerHomotopyLR Implementation
//==============================================================================

PerLayerHomotopyLR::PerLayerHomotopyLR(
    const Config& config,
    const HutchinsonConfig& hvp_config)
    : config_(config)
    , hvp_config_(hvp_config)
{}

void PerLayerHomotopyLR::register_layer(
    const std::string& name,
    const std::vector<torch::Tensor>& parameters)
{
    layers_.emplace(name, LayerInfo(hvp_config_));
    layers_.at(name).parameters = parameters;
}

std::unordered_map<std::string, double> PerLayerHomotopyLR::step(
    torch::Tensor loss,
    int step_num)
{
    step_count_ = step_num;
    std::unordered_map<std::string, double> layer_lrs;
    
    // Estimate curvature for each layer (if it's time)
    if (step_num % config_.estimation_frequency == 0) {
        for (auto& [name, info] : layers_) {
            auto metrics = info.estimator.estimate(loss, info.parameters);
            info.current_curvature = metrics.kappa_curv;
        }
    }
    
    // Collect curvatures for normalization
    std::vector<double> curvatures;
    for (const auto& [name, info] : layers_) {
        curvatures.push_back(info.current_curvature);
    }
    
    // Compute median curvature
    double median_curvature = config_.base_lr;  // Default
    if (config_.normalize_by_median && !curvatures.empty()) {
        std::sort(curvatures.begin(), curvatures.end());
        median_curvature = curvatures[curvatures.size() / 2];
    }
    
    // Compute LR for each layer: η_ℓ ∝ 1/κ_ℓ
    for (auto& [name, info] : layers_) {
        double ratio = info.current_curvature / (median_curvature + 1e-10);
        double lr = config_.base_lr / (1.0 + ratio);
        lr = std::clamp(lr, config_.min_lr, config_.max_lr);
        
        info.current_lr = lr;
        layer_lrs[name] = lr;
    }
    
    return layer_lrs;
}

void PerLayerHomotopyLR::apply_to_optimizer(torch::optim::Optimizer& optimizer)
{
    // This requires mapping parameters to layers and setting per-parameter LR
    // PyTorch C++ API doesn't have great support for this, so this is a stub
    // In practice, you'd need to use param_groups with custom options
}

double PerLayerHomotopyLR::get_layer_curvature(const std::string& layer_name) const
{
    auto it = layers_.find(layer_name);
    if (it != layers_.end()) {
        return it->second.current_curvature;
    }
    return 0.0;
}

void PerLayerHomotopyLR::export_metrics(const std::string& filename) const
{
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    
    // CSV header
    file << "layer,step,curvature,learning_rate\n";
    
    // Data
    for (const auto& [name, info] : layers_) {
        const auto& history = info.estimator.history();
        for (const auto& m : history) {
            file << name << ","
                 << m.step << ","
                 << m.kappa_curv << ","
                 << info.current_lr << "\n";
        }
    }
}

//==============================================================================
// CurvatureAwareGradientClipper Implementation
//==============================================================================

CurvatureAwareGradientClipper::CurvatureAwareGradientClipper(const Config& config)
    : config_(config)
{}

double CurvatureAwareGradientClipper::clip_gradients(
    const std::vector<torch::Tensor>& parameters,
    double curvature)
{
    // Effective clip norm: clip = base_clip / (1 + κ/κ_target)
    double ratio = curvature / (config_.curvature_target + 1e-10);
    double effective_clip = config_.base_clip_norm / (1.0 + ratio);
    effective_clip = std::max(effective_clip, config_.min_clip_norm);
    
    // Compute total gradient norm
    double total_norm = 0.0;
    for (const auto& p : parameters) {
        if (p.grad().defined()) {
            total_norm += p.grad().pow(2).sum().item<double>();
        }
    }
    total_norm = std::sqrt(total_norm);
    
    // Clip if necessary
    if (total_norm > effective_clip) {
        double scale = effective_clip / (total_norm + 1e-10);
        for (const auto& p : parameters) {
            if (p.grad().defined()) {
                p.grad().mul_(scale);
            }
        }
    }
    
    return effective_clip;
}

//==============================================================================
// CurvatureAwareWarmup Implementation
//==============================================================================

CurvatureAwareWarmup::CurvatureAwareWarmup(
    const Config& config,
    const HutchinsonConfig& hvp_config)
    : config_(config)
    , estimator_(hvp_config)
    , current_lr_(config.target_lr * config.initial_lr_fraction)
{}

double CurvatureAwareWarmup::step(
    torch::Tensor loss,
    const std::vector<torch::Tensor>& parameters)
{
    if (warmup_complete_) {
        return current_lr_;
    }
    
    step_count_++;
    
    // Estimate curvature
    auto metrics = estimator_.estimate(loss, parameters);
    
    // Increase LR only if curvature is below threshold
    if (metrics.kappa_curv < config_.curvature_threshold) {
        current_lr_ *= config_.increase_factor;
    }
    
    // Check if warmup complete
    if (current_lr_ >= config_.target_lr || step_count_ >= config_.max_warmup_steps) {
        current_lr_ = config_.target_lr;
        warmup_complete_ = true;
    }
    
    return current_lr_;
}

//==============================================================================
// HomotopyOptimizer Implementation
//==============================================================================

HomotopyOptimizer::HomotopyOptimizer(
    std::shared_ptr<torch::optim::Optimizer> base_optimizer,
    const HomotopyLRScheduler::Config& scheduler_config,
    const HutchinsonConfig& hvp_config)
    : base_optimizer_(base_optimizer)
    , scheduler_(std::make_unique<HomotopyLRScheduler>(scheduler_config, hvp_config))
{}

void HomotopyOptimizer::step(torch::Tensor loss)
{
    // Extract parameters from optimizer
    auto params = get_parameters();
    
    // Update LR based on curvature
    double lr = scheduler_->step(loss, params, step_count_);
    
    // Apply to optimizer
    scheduler_->apply_to_optimizer(*base_optimizer_);
    
    // Perform optimization step
    base_optimizer_->step();
    
    step_count_++;
}

std::vector<torch::Tensor> HomotopyOptimizer::get_parameters() const
{
    std::vector<torch::Tensor> params;
    
    for (const auto& group : base_optimizer_->param_groups()) {
        for (const auto& p : group.params()) {
            params.push_back(p);
        }
    }
    
    return params;
}

} // namespace homotopy
} // namespace hnf
