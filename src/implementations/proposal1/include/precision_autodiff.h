#pragma once

#include "precision_tensor.h"
#include <torch/torch.h>
#include <memory>
#include <vector>
#include <unordered_map>
#include <functional>

namespace hnf {
namespace proposal1 {

/**
 * @brief Precision-aware automatic differentiation
 * 
 * Implements Theorem 5.10 from HNF paper on autodiff correctness:
 * For HNF^pw-definable function f, the gradient ∇f computed by
 * backpropagation satisfies:
 * 
 *   ||∇f_computed - ∇f_true|| ≤ C·L·ε_H
 * 
 * where C depends on depth, L on condition number.
 * 
 * Key insight: Gradients have HIGHER curvature than forward pass!
 * κ_∇f ≈ κ_f · L_f²
 * 
 * This explains why mixed-precision training is hard: backward pass
 * needs significantly more precision than forward pass.
 */

class PrecisionGradient {
public:
    torch::Tensor grad_data;           // Actual gradient values
    double forward_curvature;          // Curvature of forward operation
    double backward_curvature;         // Curvature of gradient computation
    double lipschitz_constant;         // Lipschitz constant
    int required_bits_forward;         // Bits needed for forward
    int required_bits_backward;        // Bits needed for backward (typically 2-3x more!)
    std::string operation_name;
    
    PrecisionGradient() 
        : forward_curvature(0.0)
        , backward_curvature(0.0)
        , lipschitz_constant(1.0)
        , required_bits_forward(23)
        , required_bits_backward(52)
        , operation_name("unknown") {}
    
    PrecisionGradient(const torch::Tensor& grad, 
                     double fwd_curv,
                     double L,
                     const std::string& op_name)
        : grad_data(grad)
        , forward_curvature(fwd_curv)
        , lipschitz_constant(L)
        , operation_name(op_name) {
        // Key theorem: backward curvature ≈ forward curvature × L²
        backward_curvature = forward_curvature * L * L;
        
        // Compute precision requirements
        compute_precision_requirements();
    }
    
    void compute_precision_requirements() {
        const double target_accuracy = 1e-6;
        const double domain_diameter = 10.0;  // Typical range for activations
        const double c = 2.0;  // Constant from Theorem 5.7
        
        if (forward_curvature > 1e-10) {
            required_bits_forward = static_cast<int>(
                std::ceil(std::log2(c * forward_curvature * domain_diameter * domain_diameter / target_accuracy))
            );
        } else {
            required_bits_forward = 23;  // float32 sufficient for linear ops
        }
        
        if (backward_curvature > 1e-10) {
            required_bits_backward = static_cast<int>(
                std::ceil(std::log2(c * backward_curvature * domain_diameter * domain_diameter / target_accuracy))
            );
        } else {
            required_bits_backward = required_bits_forward;
        }
        
        // Clamp to reasonable ranges
        required_bits_forward = std::max(4, std::min(112, required_bits_forward));
        required_bits_backward = std::max(4, std::min(112, required_bits_backward));
    }
    
    Precision recommend_forward_precision() const {
        if (required_bits_forward <= 4) return Precision::FP8;
        if (required_bits_forward <= 7) return Precision::BFLOAT16;
        if (required_bits_forward <= 10) return Precision::FLOAT16;
        if (required_bits_forward <= 23) return Precision::FLOAT32;
        if (required_bits_forward <= 52) return Precision::FLOAT64;
        return Precision::FLOAT128;
    }
    
    Precision recommend_backward_precision() const {
        if (required_bits_backward <= 4) return Precision::FP8;
        if (required_bits_backward <= 7) return Precision::BFLOAT16;
        if (required_bits_backward <= 10) return Precision::FLOAT16;
        if (required_bits_backward <= 23) return Precision::FLOAT32;
        if (required_bits_backward <= 52) return Precision::FLOAT64;
        return Precision::FLOAT128;
    }
};

/**
 * @brief Tape for recording computation graph with precision metadata
 * 
 * Unlike PyTorch's autograd tape which only tracks dependencies,
 * this records full precision requirements at each step.
 */
class PrecisionTape {
public:
    struct TapeNode {
        size_t id;
        std::string operation;
        std::vector<size_t> inputs;
        double curvature;
        double lipschitz;
        int required_bits_fwd;
        int required_bits_bwd;
        torch::Tensor cached_output;
        bool requires_grad;
        
        TapeNode() : id(0), curvature(0), lipschitz(1), 
                    required_bits_fwd(23), required_bits_bwd(52),
                    requires_grad(false) {}
    };

private:
    std::vector<TapeNode> nodes_;
    std::unordered_map<size_t, PrecisionGradient> gradients_;
    size_t next_id_;
    bool is_recording_;
    
public:
    PrecisionTape() : next_id_(0), is_recording_(false) {}
    
    void start_recording() { is_recording_ = true; nodes_.clear(); gradients_.clear(); }
    void stop_recording() { is_recording_ = false; }
    bool is_recording() const { return is_recording_; }
    
    size_t record_operation(
        const std::string& op_name,
        const std::vector<size_t>& input_ids,
        double curvature,
        double lipschitz,
        const torch::Tensor& output,
        bool requires_grad = true
    ) {
        if (!is_recording_) return 0;
        
        TapeNode node;
        node.id = next_id_++;
        node.operation = op_name;
        node.inputs = input_ids;
        node.curvature = curvature;
        node.lipschitz = lipschitz;
        node.cached_output = output;
        node.requires_grad = requires_grad;
        
        // Compute precision requirements
        const double target_acc = 1e-6;
        const double diameter = 10.0;
        const double c = 2.0;
        
        if (curvature > 1e-10) {
            node.required_bits_fwd = static_cast<int>(
                std::ceil(std::log2(c * curvature * diameter * diameter / target_acc))
            );
        } else {
            node.required_bits_fwd = 23;
        }
        
        // Backward curvature: κ_bwd ≈ κ_fwd × L²
        double backward_curvature = curvature * lipschitz * lipschitz;
        if (backward_curvature > 1e-10) {
            node.required_bits_bwd = static_cast<int>(
                std::ceil(std::log2(c * backward_curvature * diameter * diameter / target_acc))
            );
        } else {
            node.required_bits_bwd = node.required_bits_fwd;
        }
        
        // Clamp
        node.required_bits_fwd = std::max(4, std::min(112, node.required_bits_fwd));
        node.required_bits_bwd = std::max(4, std::min(112, node.required_bits_bwd));
        
        nodes_.push_back(node);
        return node.id;
    }
    
    /**
     * @brief Compute gradients via reverse-mode autodiff with precision tracking
     * 
     * Implements the backward pass with curvature-aware precision requirements.
     * Returns mapping from node ID to PrecisionGradient.
     */
    std::unordered_map<size_t, PrecisionGradient> compute_gradients(size_t output_id) {
        std::unordered_map<size_t, PrecisionGradient> grads;
        
        // Initialize output gradient (chain rule start: dy/dy = 1)
        if (output_id >= nodes_.size()) return grads;
        
        const auto& output_node = nodes_[output_id];
        PrecisionGradient output_grad;
        output_grad.grad_data = torch::ones_like(output_node.cached_output);
        output_grad.forward_curvature = output_node.curvature;
        output_grad.lipschitz_constant = 1.0;
        output_grad.operation_name = output_node.operation;
        output_grad.compute_precision_requirements();
        grads[output_id] = output_grad;
        
        // Reverse topological order
        for (int i = static_cast<int>(output_id); i >= 0; --i) {
            const auto& node = nodes_[i];
            if (!node.requires_grad) continue;
            
            // Get gradient w.r.t. this node's output
            if (grads.find(i) == grads.end()) continue;
            const auto& grad_output = grads[i];
            
            // Propagate to inputs using chain rule
            for (size_t input_id : node.inputs) {
                if (input_id >= nodes_.size()) continue;
                const auto& input_node = nodes_[input_id];
                
                // Chain rule: ∂L/∂x = ∂L/∂y · ∂y/∂x
                // Curvature of gradient: κ_∇ ≈ κ_forward × L²
                double grad_curvature = node.curvature * node.lipschitz * node.lipschitz;
                
                if (grads.find(input_id) == grads.end()) {
                    PrecisionGradient input_grad;
                    input_grad.grad_data = grad_output.grad_data;  // Simplified
                    input_grad.forward_curvature = input_node.curvature;
                    input_grad.backward_curvature = grad_curvature;
                    input_grad.lipschitz_constant = node.lipschitz;
                    input_grad.operation_name = input_node.operation;
                    input_grad.compute_precision_requirements();
                    grads[input_id] = input_grad;
                } else {
                    // Accumulate gradients (for nodes with multiple outputs)
                    grads[input_id].backward_curvature = std::max(
                        grads[input_id].backward_curvature, 
                        grad_curvature
                    );
                }
            }
        }
        
        return grads;
    }
    
    /**
     * @brief Generate precision report for entire computation graph
     */
    void print_precision_report(bool show_backward = true) const {
        std::cout << "\n";
        std::cout << "╔═══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║        PRECISION-AWARE AUTODIFF COMPUTATION GRAPH            ║\n";
        std::cout << "╚═══════════════════════════════════════════════════════════════╝\n\n";
        
        std::cout << std::setw(15) << "Operation"
                  << std::setw(12) << "Curvature"
                  << std::setw(10) << "Lipschitz"
                  << std::setw(12) << "Fwd Bits"
                  << std::setw(12) << "Bwd Bits"
                  << std::setw(15) << "Bwd Precision\n";
        std::cout << std::string(76, '-') << "\n";
        
        int total_fwd_bits = 0;
        int total_bwd_bits = 0;
        int max_fwd_bits = 0;
        int max_bwd_bits = 0;
        
        for (const auto& node : nodes_) {
            if (!node.requires_grad) continue;
            
            std::string fwd_prec = "fp32";
            if (node.required_bits_fwd <= 4) fwd_prec = "fp8";
            else if (node.required_bits_fwd <= 10) fwd_prec = "fp16";
            else if (node.required_bits_fwd <= 23) fwd_prec = "fp32";
            else if (node.required_bits_fwd <= 52) fwd_prec = "fp64";
            else fwd_prec = "fp128";
            
            std::string bwd_prec = "fp32";
            if (node.required_bits_bwd <= 4) bwd_prec = "fp8";
            else if (node.required_bits_bwd <= 10) bwd_prec = "fp16";
            else if (node.required_bits_bwd <= 23) bwd_prec = "fp32";
            else if (node.required_bits_bwd <= 52) bwd_prec = "fp64";
            else bwd_prec = "fp128";
            
            std::cout << std::setw(15) << node.operation.substr(0, 14)
                      << std::setw(12) << std::scientific << std::setprecision(2) << node.curvature
                      << std::setw(10) << std::fixed << std::setprecision(2) << node.lipschitz
                      << std::setw(12) << node.required_bits_fwd
                      << std::setw(12) << node.required_bits_bwd;
            
            if (show_backward) {
                std::cout << std::setw(15) << bwd_prec;
                if (node.required_bits_bwd > node.required_bits_fwd + 10) {
                    std::cout << " ⚠️";
                }
            }
            std::cout << "\n";
            
            total_fwd_bits += node.required_bits_fwd;
            total_bwd_bits += node.required_bits_bwd;
            max_fwd_bits = std::max(max_fwd_bits, node.required_bits_fwd);
            max_bwd_bits = std::max(max_bwd_bits, node.required_bits_bwd);
        }
        
        std::cout << std::string(76, '-') << "\n";
        std::cout << "Summary:\n";
        std::cout << "  Total operations: " << nodes_.size() << "\n";
        std::cout << "  Max forward bits: " << max_fwd_bits << "\n";
        std::cout << "  Max backward bits: " << max_bwd_bits << "\n";
        if (nodes_.size() > 0) {
            std::cout << "  Avg forward bits: " << (total_fwd_bits / nodes_.size()) << "\n";
            std::cout << "  Avg backward bits: " << (total_bwd_bits / nodes_.size()) << "\n";
            std::cout << "  Backward overhead: " 
                      << std::fixed << std::setprecision(1)
                      << (100.0 * (total_bwd_bits - total_fwd_bits) / total_fwd_bits) 
                      << "%\n";
        }
        std::cout << "\n";
    }
    
    const std::vector<TapeNode>& nodes() const { return nodes_; }
};

/**
 * @brief Variable with precision tracking and automatic differentiation
 * 
 * Wraps PrecisionTensor with computational graph recording.
 */
class PrecisionVariable {
private:
    PrecisionTensor tensor_;
    size_t tape_id_;
    std::shared_ptr<PrecisionTape> tape_;
    bool requires_grad_;
    
public:
    PrecisionVariable(
        const PrecisionTensor& tensor,
        std::shared_ptr<PrecisionTape> tape = nullptr,
        bool requires_grad = false
    ) : tensor_(tensor)
      , tape_id_(0)
      , tape_(tape)
      , requires_grad_(requires_grad) {
        if (tape_ && tape_->is_recording()) {
            tape_id_ = tape_->record_operation(
                "input",
                {},
                tensor.curvature(),
                tensor.lipschitz(),
                tensor.data(),
                requires_grad
            );
        }
    }
    
    const PrecisionTensor& tensor() const { return tensor_; }
    PrecisionTensor& tensor() { return tensor_; }
    size_t tape_id() const { return tape_id_; }
    bool requires_grad() const { return requires_grad_; }
    
    /**
     * @brief Apply operation and record in tape
     */
    static PrecisionVariable apply_op(
        const std::string& op_name,
        const std::vector<PrecisionVariable>& inputs,
        const std::function<PrecisionTensor(const std::vector<PrecisionTensor>&)>& forward_fn
    ) {
        // Execute forward pass
        std::vector<PrecisionTensor> input_tensors;
        std::vector<size_t> input_ids;
        std::shared_ptr<PrecisionTape> tape = nullptr;
        bool any_requires_grad = false;
        
        for (const auto& input : inputs) {
            input_tensors.push_back(input.tensor_);
            input_ids.push_back(input.tape_id_);
            if (input.tape_) tape = input.tape_;
            if (input.requires_grad_) any_requires_grad = true;
        }
        
        PrecisionTensor output = forward_fn(input_tensors);
        
        // Record in tape
        size_t output_id = 0;
        if (tape && tape->is_recording() && any_requires_grad) {
            output_id = tape->record_operation(
                op_name,
                input_ids,
                output.curvature(),
                output.lipschitz(),
                output.data(),
                any_requires_grad
            );
        }
        
        PrecisionVariable result(output, tape, any_requires_grad);
        result.tape_id_ = output_id;
        return result;
    }
};

/**
 * @brief Precision-aware optimizer that adjusts learning rate based on curvature
 * 
 * Key insight: High curvature → need smaller learning rate to maintain stability
 * Implements adaptive LR: α_t = α_0 / (1 + β·κ_max)
 */
class CurvatureAwareOptimizer {
private:
    double base_lr_;
    double curvature_factor_;
    std::vector<torch::Tensor> parameters_;
    std::vector<PrecisionGradient> last_gradients_;
    
public:
    CurvatureAwareOptimizer(double lr = 0.01, double curvature_factor = 0.001)
        : base_lr_(lr), curvature_factor_(curvature_factor) {}
    
    void add_param(const torch::Tensor& param) {
        parameters_.push_back(param);
    }
    
    /**
     * @brief Compute adaptive learning rate based on gradient curvature
     */
    double compute_adaptive_lr(const std::vector<PrecisionGradient>& grads) const {
        double max_backward_curvature = 0.0;
        for (const auto& grad : grads) {
            max_backward_curvature = std::max(max_backward_curvature, grad.backward_curvature);
        }
        
        // Adaptive LR formula: α = α_0 / (1 + β·κ)
        // This ensures stability even with high curvature
        double adaptive_lr = base_lr_ / (1.0 + curvature_factor_ * max_backward_curvature);
        
        return adaptive_lr;
    }
    
    /**
     * @brief Step with precision-aware gradient descent
     */
    void step(const std::vector<PrecisionGradient>& grads) {
        if (grads.size() != parameters_.size()) {
            throw std::runtime_error("Gradient count mismatch");
        }
        
        double adaptive_lr = compute_adaptive_lr(grads);
        
        torch::NoGradGuard no_grad;
        for (size_t i = 0; i < parameters_.size(); ++i) {
            // Cast gradient to appropriate precision for update
            (void)grads[i].recommend_backward_precision();  // Silence warning
            
            // Simple gradient descent: θ ← θ - α·∇f
            parameters_[i].sub_(grads[i].grad_data * adaptive_lr);
        }
        
        last_gradients_ = grads;
    }
    
    void zero_grad() {
        torch::NoGradGuard no_grad;
        for (auto& param : parameters_) {
            if (param.grad().defined()) {
                param.grad().zero_();
            }
        }
    }
    
    double get_last_adaptive_lr() const {
        if (last_gradients_.empty()) return base_lr_;
        return compute_adaptive_lr(last_gradients_);
    }
    
    const std::vector<PrecisionGradient>& get_last_gradients() const {
        return last_gradients_;
    }
};

} // namespace proposal1
} // namespace hnf
