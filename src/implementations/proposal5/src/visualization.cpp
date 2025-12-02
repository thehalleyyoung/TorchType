#include "visualization.hpp"
#include <algorithm>
#include <numeric>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <sys/stat.h>

namespace hnf {
namespace profiler {

// ============================================================================
// CurvatureVisualizer Implementation
// ============================================================================

CurvatureVisualizer::CurvatureVisualizer(const CurvatureProfiler& profiler)
    : profiler_(profiler) {}

char CurvatureVisualizer::curvature_to_symbol(double kappa) const {
    if (kappa < 1e3) return '.';        // Low
    if (kappa < 1e6) return 'o';        // Medium-low
    if (kappa < 1e9) return 'O';        // Medium-high
    if (kappa < 1e12) return '@';       // High
    return '!';                         // Danger
}

std::string CurvatureVisualizer::generate_heatmap(
    const std::vector<std::string>& layers_to_plot) const {
    
    auto layers = layers_to_plot.empty() ? 
                  profiler_.get_tracked_layers() : layers_to_plot;
    
    if (layers.empty()) {
        return "No layers tracked.\n";
    }
    
    // Find the maximum history length
    size_t max_steps = 0;
    for (const auto& layer : layers) {
        const auto& hist = profiler_.get_history(layer);
        max_steps = std::max(max_steps, hist.size());
    }
    
    if (max_steps == 0) {
        return "No data collected yet.\n";
    }
    
    // Determine step sampling (show at most 60 columns)
    size_t stride = std::max(size_t(1), max_steps / 60);
    
    std::ostringstream oss;
    oss << "\nCurvature Heatmap (κ^{curv} over training steps)\n";
    oss << "Legend: . = low (<1e3), o = med-low (<1e6), O = med-high (<1e9), "
        << "@ = high (<1e12), ! = danger (≥1e12)\n\n";
    
    // Find max layer name length for alignment
    size_t max_name_len = 0;
    for (const auto& layer : layers) {
        max_name_len = std::max(max_name_len, layer.length());
    }
    
    // Header with step numbers
    oss << std::string(max_name_len + 3, ' ') << "│";
    for (size_t i = 0; i < max_steps; i += stride) {
        if (i % (stride * 10) == 0) {
            oss << std::setw(5) << i;
        }
    }
    oss << "\n";
    oss << std::string(max_name_len + 3, '-') << "+" 
        << std::string(max_steps / stride, '-') << "\n";
    
    // Heatmap rows
    for (const auto& layer : layers) {
        const auto& hist = profiler_.get_history(layer);
        
        oss << std::setw(max_name_len) << std::left << layer << " │ ";
        
        for (size_t i = 0; i < max_steps; i += stride) {
            if (i < hist.size()) {
                oss << curvature_to_symbol(hist[i].kappa_curv);
            } else {
                oss << ' ';
            }
        }
        oss << "\n";
    }
    
    return oss.str();
}

void CurvatureVisualizer::generate_matplotlib_script(
    const std::string& output_script,
    const std::string& data_file) const {
    
    // Export data first
    profiler_.export_to_csv(data_file);
    
    // Generate Python plotting script
    std::ofstream script(output_script);
    if (!script.is_open()) {
        throw std::runtime_error("Failed to open script file: " + output_script);
    }
    
    script << R"(#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load data
df = pd.read_csv(')" << data_file << R"(')

# Get unique layers
layers = df['layer'].unique()

# Create subplots
fig, axes = plt.subplots(len(layers), 1, figsize=(12, 4*len(layers)), sharex=True)
if len(layers) == 1:
    axes = [axes]

for ax, layer in zip(axes, layers):
    layer_data = df[df['layer'] == layer]
    
    # Plot curvature
    ax.semilogy(layer_data['step'], layer_data['kappa_curv'], 
                label=f'{layer}', linewidth=2)
    
    # Add threshold lines
    ax.axhline(1e6, color='orange', linestyle='--', alpha=0.5, label='Warning')
    ax.axhline(1e9, color='red', linestyle='--', alpha=0.5, label='Danger')
    
    ax.set_ylabel('Curvature κ')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

axes[-1].set_xlabel('Training Step')
plt.suptitle('Curvature Evolution During Training')
plt.tight_layout()
plt.savefig('curvature_timeseries.png', dpi=150)
print('Saved curvature_timeseries.png')

# Create correlation plot if we have gradient norms
fig2, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

# Total curvature
total_curv = df.groupby('step')['kappa_curv'].sum()
ax1.semilogy(total_curv.index, total_curv.values, linewidth=2)
ax1.set_ylabel('Total Curvature')
ax1.grid(True, alpha=0.3)
ax1.set_title('Total Curvature Across All Layers')

# Gradient norm
total_grad = df.groupby('step')['gradient_norm'].sum()
ax2.semilogy(total_grad.index, total_grad.values, linewidth=2, color='green')
ax2.set_ylabel('Total Gradient Norm')
ax2.set_xlabel('Training Step')
ax2.grid(True, alpha=0.3)
ax2.set_title('Total Gradient Norm')

plt.tight_layout()
plt.savefig('curvature_gradient_correlation.png', dpi=150)
print('Saved curvature_gradient_correlation.png')

plt.show()
)";
    
    script.close();
    
    // Make executable on Unix systems
#if defined(__unix__) || defined(__APPLE__)
    chmod(output_script.c_str(), 0755);
#endif
}

void CurvatureVisualizer::export_for_plotting(
    const std::string& filename,
    const std::string& format) const {
    
    if (format == "csv") {
        profiler_.export_to_csv(filename);
    } else if (format == "json") {
        // Simple JSON export
        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Failed to open file: " + filename);
        }
        
        file << "{\n  \"layers\": {\n";
        auto layers = profiler_.get_tracked_layers();
        for (size_t i = 0; i < layers.size(); ++i) {
            const auto& layer = layers[i];
            const auto& hist = profiler_.get_history(layer);
            
            file << "    \"" << layer << "\": [\n";
            for (size_t j = 0; j < hist.size(); ++j) {
                const auto& m = hist[j];
                file << "      {\"step\": " << m.step 
                     << ", \"kappa_curv\": " << m.kappa_curv
                     << ", \"gradient_norm\": " << m.gradient_norm
                     << ", \"lipschitz\": " << m.lipschitz_constant
                     << "}";
                if (j < hist.size() - 1) file << ",";
                file << "\n";
            }
            file << "    ]";
            if (i < layers.size() - 1) file << ",";
            file << "\n";
        }
        file << "  }\n}\n";
    } else {
        throw std::runtime_error("Unsupported format: " + format);
    }
}

std::vector<int> CurvatureVisualizer::find_spikes(
    const std::vector<double>& series,
    double threshold) const {
    
    std::vector<int> spikes;
    if (series.size() < 2) return spikes;
    
    for (size_t i = 1; i < series.size(); ++i) {
        if (series[i-1] > 1e-10 && series[i] / series[i-1] > threshold) {
            spikes.push_back(static_cast<int>(i));
        }
    }
    
    return spikes;
}

std::vector<std::tuple<int, int, std::string>> 
CurvatureVisualizer::correlate_with_loss_spikes(
    const std::vector<double>& loss_history,
    double spike_threshold,
    int max_lag) const {
    
    std::vector<std::tuple<int, int, std::string>> correlations;
    
    // Find loss spikes
    auto loss_spikes = find_spikes(loss_history, spike_threshold);
    
    // For each layer, find curvature spikes
    for (const auto& layer : profiler_.get_tracked_layers()) {
        const auto& hist = profiler_.get_history(layer);
        
        std::vector<double> kappa_series;
        for (const auto& m : hist) {
            kappa_series.push_back(m.kappa_curv);
        }
        
        auto curv_spikes = find_spikes(kappa_series, spike_threshold);
        
        // Find correlations
        for (int loss_spike : loss_spikes) {
            for (int curv_spike : curv_spikes) {
                int lag = loss_spike - curv_spike;
                if (lag >= 0 && lag <= max_lag) {
                    correlations.push_back({curv_spike, loss_spike, layer});
                }
            }
        }
    }
    
    return correlations;
}

std::string CurvatureVisualizer::generate_summary_report() const {
    std::ostringstream oss;
    
    oss << "\n=== Curvature Profiling Summary ===\n\n";
    
    auto layers = profiler_.get_tracked_layers();
    if (layers.empty()) {
        oss << "No layers tracked.\n";
        return oss.str();
    }
    
    for (const auto& layer : layers) {
        const auto& hist = profiler_.get_history(layer);
        if (hist.empty()) continue;
        
        oss << "Layer: " << layer << "\n";
        oss << "  Steps tracked: " << hist.size() << "\n";
        
        // Compute statistics
        double max_kappa = 0.0, min_kappa = 1e100, avg_kappa = 0.0;
        double max_grad = 0.0, avg_grad = 0.0;
        
        for (const auto& m : hist) {
            max_kappa = std::max(max_kappa, m.kappa_curv);
            min_kappa = std::min(min_kappa, m.kappa_curv);
            avg_kappa += m.kappa_curv;
            max_grad = std::max(max_grad, m.gradient_norm);
            avg_grad += m.gradient_norm;
        }
        avg_kappa /= hist.size();
        avg_grad /= hist.size();
        
        oss << "  Curvature (κ^{curv}):\n";
        oss << "    Min: " << std::scientific << min_kappa << "\n";
        oss << "    Max: " << max_kappa << "\n";
        oss << "    Avg: " << avg_kappa << "\n";
        oss << "  Gradient norm:\n";
        oss << "    Max: " << max_grad << "\n";
        oss << "    Avg: " << avg_grad << "\n";
        
        // Precision requirement estimate
        double diameter = 1.0;  // Assume unit diameter for now
        double target_eps = 1e-6;
        double req_bits = hist.back().required_mantissa_bits(diameter, target_eps);
        oss << "  Estimated precision req: " << std::fixed << std::setprecision(1) 
            << req_bits << " bits (D=1, ε=1e-6)\n";
        
        oss << "\n";
    }
    
    return oss.str();
}

// ============================================================================
// RealTimeDashboard Implementation
// ============================================================================

RealTimeDashboard::RealTimeDashboard(
    const CurvatureProfiler& profiler,
    const TrainingMonitor& monitor)
    : profiler_(profiler),
      monitor_(monitor) {}

void RealTimeDashboard::update(int step, double loss) {
    if (compact_mode_) {
        // One-line update
        std::cout << clear_line();
        std::cout << "Step " << step << " | Loss: " << std::scientific << loss;
        
        // Show max curvature
        double max_kappa = 0.0;
        std::string max_layer;
        for (const auto& layer : profiler_.get_tracked_layers()) {
            const auto& hist = profiler_.get_history(layer);
            if (!hist.empty() && hist.back().kappa_curv > max_kappa) {
                max_kappa = hist.back().kappa_curv;
                max_layer = layer;
            }
        }
        
        std::cout << " | Max κ: " << max_kappa << " (" << max_layer << ")";
        
        if (monitor_.is_danger_state()) {
            std::cout << " " << color_red() << "[DANGER]" << color_reset();
        } else if (monitor_.is_warning_state()) {
            std::cout << " " << color_yellow() << "[WARNING]" << color_reset();
        } else {
            std::cout << " " << color_green() << "[OK]" << color_reset();
        }
        
        std::cout << std::flush;
    } else {
        // Detailed update every N steps
        if (step % 10 == 0 || step != last_step_ + 1) {
            std::cout << "\n--- Step " << step << " ---\n";
            std::cout << "Loss: " << loss << "\n";
            
            for (const auto& layer : profiler_.get_tracked_layers()) {
                const auto& hist = profiler_.get_history(layer);
                if (!hist.empty()) {
                    const auto& m = hist.back();
                    std::cout << "  " << layer << ": κ=" << m.kappa_curv
                              << ", ||∇||=" << m.gradient_norm;
                    
                    if (m.kappa_curv > 1e9) {
                        std::cout << " " << color_red() << "<!>" << color_reset();
                    } else if (m.kappa_curv > 1e6) {
                        std::cout << " " << color_yellow() << "<?>" << color_reset();
                    }
                    std::cout << "\n";
                }
            }
        }
    }
    
    last_step_ = step;
}

void RealTimeDashboard::reset() {
    std::cout << "\033[2J\033[H";  // Clear screen and move to top
    last_step_ = -1;
}

} // namespace profiler
} // namespace hnf
