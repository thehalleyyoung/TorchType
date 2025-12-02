#include "mixed_precision_buffer.hpp"
#include <stdexcept>
#include <cmath>

namespace hnf {
namespace kv_cache {

// MixedPrecisionBuffer implementation

MixedPrecisionBuffer::MixedPrecisionBuffer(
    int64_t max_length,
    int64_t dim,
    const std::vector<PrecisionLevel>& position_precisions
) : max_length_(max_length), dim_(dim), position_precisions_(position_precisions) {
    
    if (position_precisions.size() > static_cast<size_t>(max_length)) {
        position_precisions_.resize(max_length);
    }
}

void MixedPrecisionBuffer::write(
    int64_t position,
    const torch::Tensor& tensor,
    PrecisionLevel precision
) {
    if (position >= max_length_) {
        throw std::out_of_range("Position exceeds buffer size");
    }
    
    // Ensure tensor is contiguous and on CPU for quantization
    auto tensor_cpu = tensor.contiguous().cpu();
    
    switch (precision) {
        case PrecisionLevel::FP32:
            fp32_storage_[position] = tensor_cpu.to(torch::kFloat32);
            break;
            
        case PrecisionLevel::FP16:
            fp16_storage_[position] = tensor_cpu.to(torch::kFloat16);
            break;
            
        case PrecisionLevel::INT8: {
            torch::Tensor quantized;
            QuantizationParams params;
            quantize_to_int8(tensor_cpu, quantized, params);
            int8_storage_[position] = quantized;
            quant_params_[position] = params;
            break;
        }
        
        case PrecisionLevel::INT4: {
            torch::Tensor quantized;
            QuantizationParams params;
            quantize_to_int4(tensor_cpu, quantized, params);
            int4_storage_[position] = quantized;
            quant_params_[position] = params;
            break;
        }
    }
    
    // Update position precision
    if (position < static_cast<int64_t>(position_precisions_.size())) {
        position_precisions_[position] = precision;
    }
}

torch::Tensor MixedPrecisionBuffer::read(int64_t position) {
    if (position >= max_length_) {
        throw std::out_of_range("Position exceeds buffer size");
    }
    
    // Find which storage contains this position
    if (fp32_storage_.count(position)) {
        return fp32_storage_[position];
    } else if (fp16_storage_.count(position)) {
        return fp16_storage_[position].to(torch::kFloat32);
    } else if (int8_storage_.count(position)) {
        return dequantize_int8(int8_storage_[position], quant_params_[position]);
    } else if (int4_storage_.count(position)) {
        return dequantize_int4(int4_storage_[position], quant_params_[position]);
    }
    
    // Position not found - return zeros
    return torch::zeros({dim_});
}

torch::Tensor MixedPrecisionBuffer::read_batch(const std::vector<int64_t>& positions) {
    std::vector<torch::Tensor> tensors;
    tensors.reserve(positions.size());
    
    for (auto pos : positions) {
        tensors.push_back(read(pos));
    }
    
    return torch::stack(tensors);
}

void MixedPrecisionBuffer::update_precision(int64_t position, PrecisionLevel new_precision) {
    // Read current value, then rewrite at new precision
    auto current_value = read(position);
    write(position, current_value, new_precision);
}

int64_t MixedPrecisionBuffer::memory_usage_bytes() const {
    int64_t total = 0;
    
    for (const auto& pair : fp32_storage_) {
        total += pair.second.numel() * 4;  // 4 bytes per float32
    }
    
    for (const auto& pair : fp16_storage_) {
        total += pair.second.numel() * 2;  // 2 bytes per float16
    }
    
    for (const auto& pair : int8_storage_) {
        total += pair.second.numel() * 1;  // 1 byte per int8
        total += 8;  // Scale and zero_point storage
    }
    
    for (const auto& pair : int4_storage_) {
        total += (pair.second.numel() + 1) / 2;  // 0.5 bytes per int4 (packed)
        total += 8;  // Scale and zero_point storage
    }
    
    return total;
}

PrecisionLevel MixedPrecisionBuffer::get_precision(int64_t position) const {
    if (position < static_cast<int64_t>(position_precisions_.size())) {
        return position_precisions_[position];
    }
    return PrecisionLevel::FP16;  // Default
}

void MixedPrecisionBuffer::quantize_to_int8(
    const torch::Tensor& tensor,
    torch::Tensor& quantized,
    QuantizationParams& params
) {
    params.precision = PrecisionLevel::INT8;
    
    // Compute scale and zero point for symmetric quantization
    auto min_val = tensor.min().item<float>();
    auto max_val = tensor.max().item<float>();
    
    auto abs_max = std::max(std::abs(min_val), std::abs(max_val));
    
    params.scale = torch::full({1}, abs_max / 127.0);
    params.zero_point = torch::zeros({1});
    params.clip_min = -127.0;
    params.clip_max = 127.0;
    
    // Quantize
    quantized = torch::clamp(
        torch::round(tensor / params.scale),
        params.clip_min,
        params.clip_max
    ).to(torch::kInt8);
}

void MixedPrecisionBuffer::quantize_to_int4(
    const torch::Tensor& tensor,
    torch::Tensor& quantized,
    QuantizationParams& params
) {
    params.precision = PrecisionLevel::INT4;
    
    // Compute scale for 4-bit range [-8, 7]
    auto min_val = tensor.min().item<float>();
    auto max_val = tensor.max().item<float>();
    
    auto abs_max = std::max(std::abs(min_val), std::abs(max_val));
    
    params.scale = torch::full({1}, abs_max / 7.0);
    params.zero_point = torch::zeros({1});
    params.clip_min = -8.0;
    params.clip_max = 7.0;
    
    // Quantize to int4 range
    quantized = torch::clamp(
        torch::round(tensor / params.scale),
        params.clip_min,
        params.clip_max
    ).to(torch::kInt8);  // Store as int8, but only use 4 bits
}

torch::Tensor MixedPrecisionBuffer::dequantize_int8(
    const torch::Tensor& quantized,
    const QuantizationParams& params
) {
    return quantized.to(torch::kFloat32) * params.scale;
}

torch::Tensor MixedPrecisionBuffer::dequantize_int4(
    const torch::Tensor& quantized,
    const QuantizationParams& params
) {
    return quantized.to(torch::kFloat32) * params.scale;
}

// AdaptivePrecisionKVCache implementation

AdaptivePrecisionKVCache::AdaptivePrecisionKVCache(
    const KVCacheConfig& config,
    const std::vector<LayerPrecisionMap>& precision_maps
) : config_(config), precision_maps_(precision_maps) {
    
    // Initialize buffers for each layer
    int64_t total_dim = config.num_heads * config.head_dim;
    
    for (int64_t layer = 0; layer < config.num_layers; ++layer) {
        LayerPrecisionMap map;
        if (layer < static_cast<int64_t>(precision_maps.size())) {
            map = precision_maps[layer];
        } else {
            // Default: all FP16
            map.layer_idx = layer;
            map.num_positions = config.max_seq_length;
            map.position_precisions.resize(config.max_seq_length, PrecisionLevel::FP16);
        }
        
        // Create buffers
        auto key_buffer = std::make_shared<MixedPrecisionBuffer>(
            config.max_seq_length,
            total_dim,
            map.position_precisions
        );
        
        auto value_buffer = std::make_shared<MixedPrecisionBuffer>(
            config.max_seq_length,
            total_dim,
            map.position_precisions
        );
        
        key_buffers_.push_back(key_buffer);
        value_buffers_.push_back(value_buffer);
        seq_lengths_.push_back(0);
    }
}

void AdaptivePrecisionKVCache::update(
    int64_t layer_idx,
    int64_t position,
    const torch::Tensor& key,
    const torch::Tensor& value
) {
    if (layer_idx >= static_cast<int64_t>(key_buffers_.size())) {
        throw std::out_of_range("Layer index out of range");
    }
    
    // Get precision for this position
    PrecisionLevel precision = PrecisionLevel::FP16;
    if (layer_idx < static_cast<int64_t>(precision_maps_.size()) &&
        position < static_cast<int64_t>(precision_maps_[layer_idx].position_precisions.size())) {
        precision = precision_maps_[layer_idx].position_precisions[position];
    }
    
    // Write to buffers
    key_buffers_[layer_idx]->write(position, key, precision);
    value_buffers_[layer_idx]->write(position, value, precision);
    
    // Update sequence length
    seq_lengths_[layer_idx] = std::max(seq_lengths_[layer_idx], position + 1);
}

std::pair<torch::Tensor, torch::Tensor> AdaptivePrecisionKVCache::get(
    int64_t layer_idx,
    const std::vector<int64_t>& positions
) {
    if (layer_idx >= static_cast<int64_t>(key_buffers_.size())) {
        throw std::out_of_range("Layer index out of range");
    }
    
    auto keys = key_buffers_[layer_idx]->read_batch(positions);
    auto values = value_buffers_[layer_idx]->read_batch(positions);
    
    return {keys, values};
}

std::pair<torch::Tensor, torch::Tensor> AdaptivePrecisionKVCache::get_all(int64_t layer_idx) {
    if (layer_idx >= static_cast<int64_t>(key_buffers_.size())) {
        throw std::out_of_range("Layer index out of range");
    }
    
    auto seq_len = seq_lengths_[layer_idx];
    std::vector<int64_t> positions(seq_len);
    std::iota(positions.begin(), positions.end(), 0);
    
    return get(layer_idx, positions);
}

void AdaptivePrecisionKVCache::update_precision_map(
    int64_t layer_idx,
    const LayerPrecisionMap& new_map
) {
    if (layer_idx >= static_cast<int64_t>(precision_maps_.size())) {
        throw std::out_of_range("Layer index out of range");
    }
    
    precision_maps_[layer_idx] = new_map;
    
    // Update buffers with new precisions
    for (size_t pos = 0; pos < new_map.position_precisions.size(); ++pos) {
        key_buffers_[layer_idx]->update_precision(pos, new_map.position_precisions[pos]);
        value_buffers_[layer_idx]->update_precision(pos, new_map.position_precisions[pos]);
    }
}

void AdaptivePrecisionKVCache::clear_layer(int64_t layer_idx) {
    if (layer_idx >= static_cast<int64_t>(key_buffers_.size())) {
        return;
    }
    
    seq_lengths_[layer_idx] = 0;
    // Buffers will be overwritten on next write
}

void AdaptivePrecisionKVCache::clear_all() {
    for (size_t i = 0; i < seq_lengths_.size(); ++i) {
        seq_lengths_[i] = 0;
    }
}

int64_t AdaptivePrecisionKVCache::get_seq_length(int64_t layer_idx) const {
    if (layer_idx >= static_cast<int64_t>(seq_lengths_.size())) {
        return 0;
    }
    return seq_lengths_[layer_idx];
}

int64_t AdaptivePrecisionKVCache::total_memory_usage_bytes() const {
    int64_t total = 0;
    
    for (const auto& buffer : key_buffers_) {
        total += buffer->memory_usage_bytes();
    }
    
    for (const auto& buffer : value_buffers_) {
        total += buffer->memory_usage_bytes();
    }
    
    return total;
}

double AdaptivePrecisionKVCache::compression_ratio() const {
    // Compute what uniform FP16 would use
    int64_t total_positions = 0;
    for (auto seq_len : seq_lengths_) {
        total_positions += seq_len;
    }
    
    int64_t dim = config_.num_heads * config_.head_dim;
    int64_t uniform_fp16_bytes = total_positions * dim * 2 * 2;  // keys + values, 2 bytes each
    
    int64_t actual_bytes = total_memory_usage_bytes();
    
    return static_cast<double>(uniform_fp16_bytes) / static_cast<double>(actual_bytes + 1);
}

} // namespace kv_cache
} // namespace hnf
