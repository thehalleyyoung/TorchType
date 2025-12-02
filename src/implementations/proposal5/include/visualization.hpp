#pragma once

#include "curvature_profiler.hpp"
#include <string>
#include <vector>
#include <fstream>

namespace hnf {
namespace profiler {

/**
 * @brief Visualization utilities for curvature metrics
 * 
 * Generates plots and heatmaps to visualize training dynamics:
 * - Curvature heatmaps over time
 * - Time series plots
 * - Correlation with loss spikes
 */
class CurvatureVisualizer {
public:
    explicit CurvatureVisualizer(const CurvatureProfiler& profiler);
    
    /**
     * @brief Generate ASCII heatmap of curvature over time
     * @param layers Layers to include (empty = all)
     * @return ASCII art heatmap string
     */
    std::string generate_heatmap(
        const std::vector<std::string>& layers = {}) const;
    
    /**
     * @brief Generate matplotlib script for time series plots
     * @param output_script Path to write Python script
     * @param data_file Path to write data CSV
     */
    void generate_matplotlib_script(
        const std::string& output_script,
        const std::string& data_file) const;
    
    /**
     * @brief Export data in format for external plotting tools
     * @param filename Output file path
     * @param format "csv" or "json"
     */
    void export_for_plotting(
        const std::string& filename,
        const std::string& format = "csv") const;
    
    /**
     * @brief Generate summary statistics report
     */
    std::string generate_summary_report() const;
    
    /**
     * @brief Identify curvature spikes that preceded loss spikes
     * @param loss_history History of loss values
     * @param spike_threshold Relative increase to count as spike
     * @param max_lag Maximum lag to consider (steps)
     * @return Vector of (curv_spike_step, loss_spike_step, layer_name)
     */
    std::vector<std::tuple<int, int, std::string>> correlate_with_loss_spikes(
        const std::vector<double>& loss_history,
        double spike_threshold = 1.5,
        int max_lag = 100) const;
    
private:
    const CurvatureProfiler& profiler_;
    
    // Helper: classify curvature magnitude
    char curvature_to_symbol(double kappa) const;
    
    // Helper: detect spikes in a time series
    std::vector<int> find_spikes(
        const std::vector<double>& series,
        double threshold) const;
};

/**
 * @brief Real-time dashboard for training monitoring
 * 
 * Provides live updates during training with ANSI color codes.
 */
class RealTimeDashboard {
public:
    explicit RealTimeDashboard(
        const CurvatureProfiler& profiler,
        const TrainingMonitor& monitor);
    
    /**
     * @brief Display current status (call each training step)
     * @param step Current training step
     * @param loss Current loss value
     */
    void update(int step, double loss);
    
    /**
     * @brief Set display mode
     */
    void set_compact_mode(bool compact) { compact_mode_ = compact; }
    
    /**
     * @brief Clear screen and reset
     */
    void reset();
    
private:
    const CurvatureProfiler& profiler_;
    const TrainingMonitor& monitor_;
    bool compact_mode_ = false;
    int last_step_ = -1;
    
    // ANSI color codes
    std::string color_green() const { return "\033[32m"; }
    std::string color_yellow() const { return "\033[33m"; }
    std::string color_red() const { return "\033[31m"; }
    std::string color_reset() const { return "\033[0m"; }
    std::string clear_line() const { return "\033[2K\r"; }
};

} // namespace profiler
} // namespace hnf
