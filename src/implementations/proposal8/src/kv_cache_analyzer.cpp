#include "kv_cache_analyzer.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <iomanip>
#include <algorithm>
#include <numeric>

namespace hnf {
namespace kv_cache {

KVCacheAnalyzer::KVCacheAnalyzer(const KVCacheConfig& config)
    : config_(config) {
    
    curvature_analyzer_ = std::make_unique<CurvatureAnalyzer>(config);
    precision_mapper_ = std::make_unique<PrecisionMapper>(config);
}

PrecisionAnalysisResult KVCacheAnalyzer::analyze(
    const std::vector<torch::Tensor>& calibration_data,
    std::function<std::pair<torch::Tensor, std::vector<torch::Tensor>>(const torch::Tensor&)> forward_fn
) {
    PrecisionAnalysisResult result;
    result.num_layers = config_.num_layers;
    result.max_seq_length = config_.max_seq_length;
    result.hidden_dim = config_.num_heads * config_.head_dim;
    
    // Clear previous data
    layer_attention_patterns_.clear();
    layer_position_curvatures_.clear();
    
    std::cout << "Running calibration on " << calibration_data.size() << " samples...\n";
    
    // Run forward passes and collect attention patterns
    for (size_t sample_idx = 0; sample_idx < calibration_data.size(); ++sample_idx) {
        const auto& input = calibration_data[sample_idx];
        
        std::cout << "  Sample " << (sample_idx + 1) << "/" << calibration_data.size() << "...\n";
        
        // Run forward pass
        auto [output, attention_weights_per_layer] = forward_fn(input);
        
        // Analyze each layer
        for (size_t layer_idx = 0; layer_idx < attention_weights_per_layer.size(); ++layer_idx) {
            const auto& attention_weights = attention_weights_per_layer[layer_idx];
            
            // Analyze attention pattern
            auto pattern = curvature_analyzer_->analyze_attention_pattern(
                attention_weights, layer_idx
            );
            layer_attention_patterns_[layer_idx].push_back(pattern);
            
            // For now, we'll use dummy keys/values since they're not provided
            // In practice, these would be hooked from the model
            auto seq_len = attention_weights.size(2);
            auto keys = torch::randn({1, seq_len, config_.num_heads, config_.head_dim});
            auto values = torch::randn({1, seq_len, config_.num_heads, config_.head_dim});
            auto queries = torch::randn({1, seq_len, config_.num_heads, config_.head_dim});
            
            // Compute curvatures
            std::vector<PositionCurvature> curvatures;
            
            switch (config_.curvature_method) {
                case KVCacheConfig::CurvatureMethod::ATTENTION_BASED:
                    curvatures = curvature_analyzer_->compute_position_curvatures(
                        attention_weights, keys, values, queries, layer_idx
                    );
                    break;
                    
                case KVCacheConfig::CurvatureMethod::HYBRID:
                    curvatures = curvature_analyzer_->compute_hybrid_curvature(
                        attention_weights, keys, values, queries, output, layer_idx
                    );
                    break;
                    
                default:
                    curvatures = curvature_analyzer_->compute_position_curvatures(
                        attention_weights, keys, values, queries, layer_idx
                    );
            }
            
            layer_position_curvatures_[layer_idx].push_back(curvatures);
        }
    }
    
    std::cout << "Aggregating curvatures and computing precision maps...\n";
    
    // Aggregate curvatures across samples and compute precision maps
    for (int64_t layer_idx = 0; layer_idx < config_.num_layers; ++layer_idx) {
        if (layer_position_curvatures_.count(layer_idx) == 0) {
            continue;
        }
        
        // Aggregate curvatures
        auto aggregated = aggregate_curvatures(layer_position_curvatures_[layer_idx]);
        result.layer_curvatures[layer_idx] = aggregated;
        
        // Map to precision
        auto precision_map = precision_mapper_->map_curvatures_to_precisions(
            aggregated, layer_idx
        );
        
        result.layer_maps.push_back(precision_map);
    }
    
    // Apply optimization if needed
    if (config_.memory_budget_gb > 0) {
        std::cout << "Optimizing for memory budget: " << config_.memory_budget_gb << " GB...\n";
        precision_mapper_->optimize_for_memory_budget(
            result.layer_maps, config_.memory_budget_gb
        );
    } else if (config_.quality_threshold > 0 && config_.quality_threshold < 1.0) {
        std::cout << "Optimizing for quality threshold: " << config_.quality_threshold << "...\n";
        precision_mapper_->optimize_for_quality_threshold(
            result.layer_maps, config_.quality_threshold
        );
    }
    
    // Compute global statistics
    compute_global_statistics(result);
    
    // Generate recommendations
    result.recommendations = generate_recommendations(result);
    
    std::cout << "Analysis complete!\n";
    
    return result;
}

PrecisionAnalysisResult KVCacheAnalyzer::analyze_with_hooks(
    torch::nn::Module& model,
    const std::vector<torch::Tensor>& calibration_data
) {
    // This would require hooking into the model's attention layers
    // For now, we'll throw an error indicating it's not implemented
    throw std::runtime_error("analyze_with_hooks not yet implemented - use analyze() instead");
}

std::shared_ptr<AdaptivePrecisionKVCache> KVCacheAnalyzer::create_adaptive_cache(
    const PrecisionAnalysisResult& analysis
) {
    return std::make_shared<AdaptivePrecisionKVCache>(config_, analysis.layer_maps);
}

LayerPrecisionMap KVCacheAnalyzer::analyze_layer(
    int64_t layer_idx,
    const std::vector<torch::Tensor>& attention_weights_samples,
    const std::vector<torch::Tensor>& keys_samples,
    const std::vector<torch::Tensor>& values_samples,
    const std::vector<torch::Tensor>& queries_samples
) {
    std::vector<std::vector<PositionCurvature>> all_curvatures;
    
    for (size_t i = 0; i < attention_weights_samples.size(); ++i) {
        auto curvatures = curvature_analyzer_->compute_position_curvatures(
            attention_weights_samples[i],
            keys_samples[i],
            values_samples[i],
            queries_samples[i],
            layer_idx
        );
        all_curvatures.push_back(curvatures);
    }
    
    auto aggregated = aggregate_curvatures(all_curvatures);
    return precision_mapper_->map_curvatures_to_precisions(aggregated, layer_idx);
}

std::vector<std::string> KVCacheAnalyzer::generate_recommendations(
    const PrecisionAnalysisResult& analysis
) const {
    std::vector<std::string> recommendations;
    
    // Overall compression
    if (analysis.overall_compression_ratio > 3.0) {
        recommendations.push_back(
            "Excellent compression ratio of " + 
            std::to_string(analysis.overall_compression_ratio) + 
            "x. Consider using this for production."
        );
    } else if (analysis.overall_compression_ratio > 2.0) {
        recommendations.push_back(
            "Good compression ratio of " + 
            std::to_string(analysis.overall_compression_ratio) + 
            "x. Memory savings are significant."
        );
    } else {
        recommendations.push_back(
            "Moderate compression ratio of " + 
            std::to_string(analysis.overall_compression_ratio) + 
            "x. Consider adjusting quality threshold."
        );
    }
    
    // Layer-specific recommendations
    for (const auto& layer_map : analysis.layer_maps) {
        if (layer_map.compression_ratio() < 1.5) {
            recommendations.push_back(
                "Layer " + std::to_string(layer_map.layer_idx) + 
                " has high precision requirements. This may be a critical layer."
            );
        }
        
        if (layer_map.max_curvature > 10.0) {
            recommendations.push_back(
                "Layer " + std::to_string(layer_map.layer_idx) + 
                " has very high curvature (max=" + std::to_string(layer_map.max_curvature) + 
                "). Recommend keeping FP16 for high-attention positions."
            );
        }
    }
    
    // Quality preservation
    if (analysis.quality_preserved < 0.95) {
        recommendations.push_back(
            "⚠️ Quality preservation is below 95%. Consider increasing precision or adjusting thresholds."
        );
    }
    
    return recommendations;
}

void KVCacheAnalyzer::print_analysis_report(
    const PrecisionAnalysisResult& analysis,
    std::ostream& os
) const {
    os << "╔════════════════════════════════════════════════════════════════╗\n";
    os << "║          KV-CACHE PRECISION ANALYSIS REPORT                    ║\n";
    os << "╠════════════════════════════════════════════════════════════════╣\n";
    os << "║ Configuration:                                                 ║\n";
    os << "║   Layers: " << std::setw(3) << analysis.num_layers << "                                                   ║\n";
    os << "║   Max Sequence Length: " << std::setw(5) << analysis.max_seq_length << "                                ║\n";
    os << "║   Hidden Dimension: " << std::setw(5) << analysis.hidden_dim << "                                   ║\n";
    os << "╠════════════════════════════════════════════════════════════════╣\n";
    os << "║ Memory Analysis:                                               ║\n";
    os << "║   Uniform FP16: " << std::fixed << std::setprecision(2) << std::setw(6) 
       << analysis.total_memory_fp16_gb << " GB                                  ║\n";
    os << "║   Adaptive Precision: " << std::setw(6) 
       << analysis.total_memory_adaptive_gb << " GB                             ║\n";
    os << "║   Compression Ratio: " << std::setw(6) 
       << analysis.overall_compression_ratio << "x                                 ║\n";
    os << "║   Memory Saved: " << std::setw(6) 
       << (analysis.total_memory_fp16_gb - analysis.total_memory_adaptive_gb) << " GB                                   ║\n";
    os << "╠════════════════════════════════════════════════════════════════╣\n";
    os << "║ Quality:                                                       ║\n";
    os << "║   Estimated Preservation: " << std::setw(5) 
       << (analysis.quality_preserved * 100.0) << "%                            ║\n";
    os << "╠════════════════════════════════════════════════════════════════╣\n";
    os << "║ Per-Layer Breakdown:                                           ║\n";
    os << "╠════════════════════════════════════════════════════════════════╣\n";
    os << "║ Layer │ Avg Curv │ Max Curv │ Compression │ Memory (MB)       ║\n";
    os << "╠═══════╪══════════╪══════════╪═════════════╪═══════════════════╣\n";
    
    for (const auto& layer_map : analysis.layer_maps) {
        os << "║  " << std::setw(4) << layer_map.layer_idx << " │ "
           << std::setw(8) << std::setprecision(3) << layer_map.avg_curvature << " │ "
           << std::setw(8) << std::setprecision(3) << layer_map.max_curvature << " │ "
           << std::setw(10) << std::setprecision(2) << layer_map.compression_ratio() << "x │ "
           << std::setw(12) << std::setprecision(1) << (layer_map.memory_bytes_adaptive / (1024.0 * 1024.0)) << " MB   ║\n";
    }
    
    os << "╠════════════════════════════════════════════════════════════════╣\n";
    os << "║ Recommendations:                                               ║\n";
    os << "╠════════════════════════════════════════════════════════════════╣\n";
    
    for (const auto& rec : analysis.recommendations) {
        // Wrap recommendations to fit in box
        std::istringstream iss(rec);
        std::string word;
        std::string line = "║ ";
        
        while (iss >> word) {
            if (line.length() + word.length() + 1 > 64) {
                // Pad and output current line
                while (line.length() < 64) line += " ";
                line += "║";
                os << line << "\n";
                line = "║ " + word + " ";
            } else {
                line += word + " ";
            }
        }
        
        // Output remaining line
        if (line.length() > 2) {
            while (line.length() < 64) line += " ";
            line += "║";
            os << line << "\n";
        }
    }
    
    os << "╚════════════════════════════════════════════════════════════════╝\n";
}

void KVCacheAnalyzer::save_analysis(
    const PrecisionAnalysisResult& analysis,
    const std::string& filepath
) const {
    std::ofstream ofs(filepath);
    if (!ofs.is_open()) {
        throw std::runtime_error("Could not open file for writing: " + filepath);
    }
    
    print_analysis_report(analysis, ofs);
    
    ofs << "\nDetailed layer data:\n";
    for (size_t i = 0; i < analysis.layer_maps.size(); ++i) {
        const auto& map = analysis.layer_maps[i];
        ofs << "Layer " << map.layer_idx << " precision distribution:\n";
        
        std::map<PrecisionLevel, int> precision_counts;
        for (auto prec : map.position_precisions) {
            precision_counts[prec]++;
        }
        
        for (const auto& [prec, count] : precision_counts) {
            ofs << "  " << precision_to_string(prec) << ": " << count << " positions\n";
        }
    }
    
    ofs.close();
}

PrecisionAnalysisResult KVCacheAnalyzer::load_analysis(const std::string& filepath) {
    // Simplified loading - in practice would parse the saved format
    throw std::runtime_error("load_analysis not yet implemented");
}

std::vector<PositionCurvature> KVCacheAnalyzer::aggregate_curvatures(
    const std::vector<std::vector<PositionCurvature>>& samples
) {
    if (samples.empty()) {
        return {};
    }
    
    size_t num_positions = samples[0].size();
    std::vector<PositionCurvature> aggregated(num_positions);
    
    for (size_t pos = 0; pos < num_positions; ++pos) {
        aggregated[pos].position = pos;
        aggregated[pos].layer_idx = samples[0][pos].layer_idx;
        
        // Average curvature scores across samples
        double sum_curvature = 0.0;
        double sum_attention = 0.0;
        double sum_gradient = 0.0;
        double sum_hessian = 0.0;
        
        for (const auto& sample : samples) {
            if (pos < sample.size()) {
                sum_curvature += sample[pos].curvature_score;
                sum_attention += sample[pos].attention_weight;
                sum_gradient += sample[pos].gradient_norm;
                sum_hessian += sample[pos].hessian_trace;
            }
        }
        
        double n = static_cast<double>(samples.size());
        aggregated[pos].curvature_score = sum_curvature / n;
        aggregated[pos].attention_weight = sum_attention / n;
        aggregated[pos].gradient_norm = sum_gradient / n;
        aggregated[pos].hessian_trace = sum_hessian / n;
    }
    
    return aggregated;
}

void KVCacheAnalyzer::compute_global_statistics(PrecisionAnalysisResult& result) {
    double total_fp16_bytes = 0.0;
    double total_adaptive_bytes = 0.0;
    
    for (const auto& map : result.layer_maps) {
        total_fp16_bytes += map.memory_bytes_fp16;
        total_adaptive_bytes += map.memory_bytes_adaptive;
    }
    
    result.total_memory_fp16_gb = total_fp16_bytes / (1024.0 * 1024.0 * 1024.0);
    result.total_memory_adaptive_gb = total_adaptive_bytes / (1024.0 * 1024.0 * 1024.0);
    result.overall_compression_ratio = total_fp16_bytes / (total_adaptive_bytes + 1e-10);
    
    // Estimate quality preservation
    std::vector<std::vector<PositionCurvature>> curvature_vectors;
    for (const auto& [layer_idx, curvatures] : result.layer_curvatures) {
        curvature_vectors.push_back(curvatures);
    }
    
    result.quality_preserved = precision_mapper_->estimate_quality_preservation(
        result.layer_maps, curvature_vectors
    );
}

bool KVCacheAnalyzer::validate_analysis(const PrecisionAnalysisResult& result) const {
    // Check that all layers have precision maps
    if (result.layer_maps.size() != static_cast<size_t>(result.num_layers)) {
        return false;
    }
    
    // Check memory calculations make sense
    if (result.total_memory_adaptive_gb > result.total_memory_fp16_gb) {
        return false;
    }
    
    return true;
}

// DynamicPrecisionAdjuster implementation

DynamicPrecisionAdjuster::DynamicPrecisionAdjuster(
    const KVCacheConfig& config,
    std::shared_ptr<AdaptivePrecisionKVCache> cache
) : config_(config), cache_(cache) {}

void DynamicPrecisionAdjuster::update_importance(
    int64_t layer_idx,
    const torch::Tensor& attention_weights
) {
    // attention_weights: [batch, num_heads, seq_len, seq_len]
    // Compute average attention TO each position
    auto importance = attention_weights.mean(c10::IntArrayRef{0, 1, 2});  // [seq_len]
    
    if (position_importance_.count(layer_idx) == 0) {
        position_importance_[layer_idx] = importance;
    } else {
        // Exponential moving average
        position_importance_[layer_idx] = 
            ema_alpha_ * importance + 
            (1.0 - ema_alpha_) * position_importance_[layer_idx];
    }
}

PrecisionLevel DynamicPrecisionAdjuster::get_precision(int64_t layer_idx, int64_t position) {
    if (position_importance_.count(layer_idx) == 0) {
        return PrecisionLevel::FP16;
    }
    
    auto importance = position_importance_[layer_idx][position].item<double>();
    return importance_to_precision(importance);
}

bool DynamicPrecisionAdjuster::adjust_precisions(int64_t layer_idx) {
    if (position_importance_.count(layer_idx) == 0) {
        return false;
    }
    
    bool changed = false;
    auto importance = position_importance_[layer_idx];
    auto seq_len = cache_->get_seq_length(layer_idx);
    
    for (int64_t pos = 0; pos < seq_len; ++pos) {
        auto imp = importance[pos].item<double>();
        auto new_precision = importance_to_precision(imp);
        
        // Only change if significantly different
        // This avoids constant requantization
        // In practice, would check current precision first
        
        // For now, we skip the actual adjustment
        // to avoid too frequent requantization
    }
    
    return changed;
}

PrecisionLevel DynamicPrecisionAdjuster::importance_to_precision(double importance) {
    if (importance > upgrade_threshold_) {
        return PrecisionLevel::FP16;
    } else if (importance > downgrade_threshold_) {
        return PrecisionLevel::INT8;
    } else {
        return PrecisionLevel::INT4;
    }
}

} // namespace kv_cache
} // namespace hnf
