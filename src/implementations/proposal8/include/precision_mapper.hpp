#pragma once

#include "kv_cache_types.hpp"
#include "curvature_analyzer.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace hnf {
namespace kv_cache {

/**
 * PrecisionMapper: Maps curvature scores to precision requirements
 * 
 * Implements HNF Theorem 5.7 to determine minimum precision per position:
 * p_min = log_2(c * κ * D^2 / ε) + safety_margin
 * 
 * Also considers:
 * - Memory budget constraints
 * - Quality threshold requirements
 * - Layer-specific characteristics
 */
class PrecisionMapper {
public:
    explicit PrecisionMapper(const KVCacheConfig& config);
    
    /**
     * Map curvatures to precision levels
     * 
     * @param curvatures: Position-wise curvature scores
     * @param layer_idx: Which layer
     * @return: LayerPrecisionMap with precision assignments
     */
    LayerPrecisionMap map_curvatures_to_precisions(
        const std::vector<PositionCurvature>& curvatures,
        int64_t layer_idx
    );
    
    /**
     * Optimize precision map to meet memory budget
     * Uses dynamic programming to maximize quality under memory constraint
     */
    void optimize_for_memory_budget(
        std::vector<LayerPrecisionMap>& layer_maps,
        double memory_budget_gb
    );
    
    /**
     * Optimize precision map to meet quality threshold
     * Minimizes memory while maintaining quality
     */
    void optimize_for_quality_threshold(
        std::vector<LayerPrecisionMap>& layer_maps,
        double quality_threshold
    );
    
    /**
     * Compute required precision for a given curvature
     * Based on HNF Theorem 5.7
     */
    PrecisionLevel compute_required_precision(
        double curvature,
        double diameter,
        double target_epsilon
    ) const;
    
    /**
     * Estimate quality preservation from precision map
     * Returns value in [0, 1]
     */
    double estimate_quality_preservation(
        const std::vector<LayerPrecisionMap>& layer_maps,
        const std::vector<std::vector<PositionCurvature>>& curvatures
    ) const;
    
    /**
     * Compute memory usage for a precision map
     */
    double compute_memory_usage_gb(
        const std::vector<LayerPrecisionMap>& layer_maps
    ) const;
    
private:
    KVCacheConfig config_;
    
    // HNF constant from Theorem 5.7
    double hnf_constant_c_ = 4.0;  // Empirically determined
    
    // Map continuous curvature score to discrete precision level
    PrecisionLevel score_to_precision(double curvature_score) const;
    
    // Greedy algorithm for precision downgrading under memory constraint
    void greedy_precision_downgrade(
        std::vector<LayerPrecisionMap>& layer_maps,
        double target_memory_gb
    );
    
    // Dynamic programming for optimal precision allocation
    void dp_precision_allocation(
        std::vector<LayerPrecisionMap>& layer_maps,
        double memory_budget_gb
    );
};

} // namespace kv_cache
} // namespace hnf
