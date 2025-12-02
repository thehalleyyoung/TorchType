#pragma once

#include "kv_cache_types.hpp"
#include <torch/torch.h>
#include <vector>
#include <map>

namespace hnf {
namespace kv_cache {

/**
 * MixedPrecisionBuffer: Stores KV cache entries at different precisions
 * 
 * Supports:
 * - FP32, FP16, BF16, INT8, INT4 storage
 * - Per-position precision control
 * - Efficient quantization/dequantization
 * - Memory-efficient storage
 */
class MixedPrecisionBuffer {
public:
    MixedPrecisionBuffer(
        int64_t max_length,
        int64_t dim,
        const std::vector<PrecisionLevel>& position_precisions
    );
    
    /**
     * Write a tensor to the buffer at specified position and precision
     */
    void write(
        int64_t position,
        const torch::Tensor& tensor,
        PrecisionLevel precision
    );
    
    /**
     * Read tensor from buffer (automatically dequantizes)
     */
    torch::Tensor read(int64_t position);
    
    /**
     * Read multiple positions
     */
    torch::Tensor read_batch(const std::vector<int64_t>& positions);
    
    /**
     * Update precision for a position (requantizes if needed)
     */
    void update_precision(int64_t position, PrecisionLevel new_precision);
    
    /**
     * Get current memory usage in bytes
     */
    int64_t memory_usage_bytes() const;
    
    /**
     * Get precision at a position
     */
    PrecisionLevel get_precision(int64_t position) const;
    
private:
    int64_t max_length_;
    int64_t dim_;
    std::vector<PrecisionLevel> position_precisions_;
    
    // Separate storage for each precision level
    std::map<int64_t, torch::Tensor> fp32_storage_;
    std::map<int64_t, torch::Tensor> fp16_storage_;
    std::map<int64_t, torch::Tensor> int8_storage_;
    std::map<int64_t, torch::Tensor> int4_storage_;
    
    // Quantization parameters for each position
    std::map<int64_t, QuantizationParams> quant_params_;
    
    // Quantization helpers
    void quantize_to_int8(
        const torch::Tensor& tensor,
        torch::Tensor& quantized,
        QuantizationParams& params
    );
    
    void quantize_to_int4(
        const torch::Tensor& tensor,
        torch::Tensor& quantized,
        QuantizationParams& params
    );
    
    torch::Tensor dequantize_int8(
        const torch::Tensor& quantized,
        const QuantizationParams& params
    );
    
    torch::Tensor dequantize_int4(
        const torch::Tensor& quantized,
        const QuantizationParams& params
    );
};

/**
 * AdaptivePrecisionKVCache: Complete KV cache with adaptive precision
 * 
 * Manages key and value caches for all layers with position-specific precision
 */
class AdaptivePrecisionKVCache {
public:
    AdaptivePrecisionKVCache(
        const KVCacheConfig& config,
        const std::vector<LayerPrecisionMap>& precision_maps
    );
    
    /**
     * Update cache with new key-value pair
     */
    void update(
        int64_t layer_idx,
        int64_t position,
        const torch::Tensor& key,
        const torch::Tensor& value
    );
    
    /**
     * Get cached keys and values for a layer
     */
    std::pair<torch::Tensor, torch::Tensor> get(
        int64_t layer_idx,
        const std::vector<int64_t>& positions
    );
    
    /**
     * Get all cached keys and values for a layer
     */
    std::pair<torch::Tensor, torch::Tensor> get_all(int64_t layer_idx);
    
    /**
     * Update precision map (requantizes as needed)
     */
    void update_precision_map(
        int64_t layer_idx,
        const LayerPrecisionMap& new_map
    );
    
    /**
     * Clear cache for a layer
     */
    void clear_layer(int64_t layer_idx);
    
    /**
     * Clear entire cache
     */
    void clear_all();
    
    /**
     * Get current sequence length for a layer
     */
    int64_t get_seq_length(int64_t layer_idx) const;
    
    /**
     * Get total memory usage in bytes
     */
    int64_t total_memory_usage_bytes() const;
    
    /**
     * Get memory usage in GB
     */
    double total_memory_usage_gb() const {
        return static_cast<double>(total_memory_usage_bytes()) / (1024.0 * 1024.0 * 1024.0);
    }
    
    /**
     * Get compression ratio vs uniform FP16
     */
    double compression_ratio() const;
    
private:
    KVCacheConfig config_;
    std::vector<LayerPrecisionMap> precision_maps_;
    
    // Buffers for each layer (keys and values)
    std::vector<std::shared_ptr<MixedPrecisionBuffer>> key_buffers_;
    std::vector<std::shared_ptr<MixedPrecisionBuffer>> value_buffers_;
    
    // Current sequence lengths per layer
    std::vector<int64_t> seq_lengths_;
};

} // namespace kv_cache
} // namespace hnf
