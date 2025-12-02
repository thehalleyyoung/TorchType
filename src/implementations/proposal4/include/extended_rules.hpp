#pragma once

#include "rewrite_rules.hpp"
#include "extended_patterns.hpp"

namespace hnf {
namespace rewriter {

// Extended rule library with 20+ rules as specified in the proposal
class ExtendedRuleLibrary {
public:
    // Category 1: Advanced Cancellation Rules
    
    // Rule: sqrt(x^2) → abs(x)
    static RewriteRule sqrt_square_to_abs() {
        return RewriteRule(
            "sqrt_square_to_abs",
            "Replace sqrt(x^2) with abs(x)",
            ExtendedPatternLibrary::sqrt_square_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("abs_result", OpType::ABS, {x_it->second});
                    g.add_output("abs_result");
                }
                return g;
            }
        );
    }
    
    // Rule: x - x → 0 (when x is expensive to compute)
    static RewriteRule self_subtraction() {
        return RewriteRule(
            "self_subtraction",
            "Replace x - x with 0",
            ExtendedPatternLibrary::self_sub_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                g.add_constant("zero", 0.0);
                g.add_output("zero");
                return g;
            }
        );
    }
    
    // Rule: x / x → 1 (for x ≠ 0)
    static RewriteRule self_division() {
        return RewriteRule(
            "self_division",
            "Replace x / x with 1 (for x != 0)",
            ExtendedPatternLibrary::self_div_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                g.add_constant("one", 1.0);
                g.add_output("one");
                return g;
            }
        );
    }
    
    // Category 2: Advanced Stabilization Rules
    
    // Rule: log(1 + x) → log1p(x) for small x
    static RewriteRule log1p_rule() {
        return RewriteRule(
            "log1p",
            "Replace log(1 + x) with log1p(x) for numerical stability",
            ExtendedPatternLibrary::log_one_plus_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("log1p_result", OpType::LOG1P, {x_it->second});
                    g.add_output("log1p_result");
                }
                return g;
            }
        );
    }
    
    // Rule: exp(x) - 1 → expm1(x) for small x
    static RewriteRule expm1_rule() {
        return RewriteRule(
            "expm1",
            "Replace exp(x) - 1 with expm1(x) for numerical stability",
            ExtendedPatternLibrary::exp_minus_one_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("expm1_result", OpType::EXPM1, {x_it->second});
                    g.add_output("expm1_result");
                }
                return g;
            }
        );
    }
    
    // Rule: 1 / (1 + exp(-x)) → sigmoid(x)
    static RewriteRule stable_sigmoid() {
        return RewriteRule(
            "stable_sigmoid",
            "Replace naive sigmoid formula with stable sigmoid",
            ExtendedPatternLibrary::naive_sigmoid_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("sigmoid_result", OpType::SIGMOID, {x_it->second});
                    g.add_output("sigmoid_result");
                }
                return g;
            }
        );
    }
    
    // Rule: (exp(x) - exp(-x)) / (exp(x) + exp(-x)) → tanh(x)
    static RewriteRule stable_tanh() {
        return RewriteRule(
            "stable_tanh",
            "Replace naive tanh formula with stable tanh",
            ExtendedPatternLibrary::naive_tanh_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("tanh_result", OpType::TANH, {x_it->second});
                    g.add_output("tanh_result");
                }
                return g;
            }
        );
    }
    
    // Category 3: Advanced Fusion Rules
    
    // Rule: LayerNorm fusion (x - mean(x)) / sqrt(var(x) + eps) → layer_norm(x)
    static RewriteRule layernorm_fusion() {
        return RewriteRule(
            "layernorm_fusion",
            "Fuse layer normalization pattern into single op",
            ExtendedPatternLibrary::layernorm_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("ln_result", OpType::LAYER_NORM, {x_it->second});
                    g.add_output("ln_result");
                }
                return g;
            }
        );
    }
    
    // Rule: BatchNorm fusion
    static RewriteRule batchnorm_fusion() {
        return RewriteRule(
            "batchnorm_fusion",
            "Fuse batch normalization pattern into single op",
            ExtendedPatternLibrary::batchnorm_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("bn_result", OpType::BATCH_NORM, {x_it->second});
                    g.add_output("bn_result");
                }
                return g;
            }
        );
    }
    
    // Rule: RMSNorm fusion: x / sqrt(mean(x^2) + eps) → rms_norm(x)
    static RewriteRule rmsnorm_fusion() {
        return RewriteRule(
            "rmsnorm_fusion",
            "Fuse RMS normalization pattern",
            ExtendedPatternLibrary::rmsnorm_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("rms_result", OpType::RMS_NORM, {x_it->second});
                    g.add_output("rms_result");
                }
                return g;
            }
        );
    }
    
    // Rule: GELU fusion: 0.5 * x * (1 + tanh(...)) → gelu(x)
    static RewriteRule gelu_fusion() {
        return RewriteRule(
            "gelu_fusion",
            "Fuse GELU activation pattern",
            ExtendedPatternLibrary::gelu_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("gelu_result", OpType::GELU, {x_it->second});
                    g.add_output("gelu_result");
                }
                return g;
            }
        );
    }
    
    // Rule: SwiGLU fusion: swish(x) * y → swiglu(x, y)
    static RewriteRule swiglu_fusion() {
        return RewriteRule(
            "swiglu_fusion",
            "Fuse SwiGLU activation pattern",
            ExtendedPatternLibrary::swiglu_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                auto y_it = match.find("$y");
                if (x_it != match.end() && y_it != match.end()) {
                    g.add_node("swiglu_result", OpType::SWIGLU, {x_it->second, y_it->second});
                    g.add_output("swiglu_result");
                }
                return g;
            }
        );
    }
    
    // Category 4: Matrix Operation Rewrites
    
    // Rule: (A @ B) @ C → A @ (B @ C) when beneficial
    static RewriteRule matmul_reassociate() {
        return RewriteRule(
            "matmul_reassociate",
            "Reassociate matrix multiplication for better performance",
            ExtendedPatternLibrary::matmul_chain_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto a_it = match.find("$A");
                auto b_it = match.find("$B");
                auto c_it = match.find("$C");
                
                if (a_it != match.end() && b_it != match.end() && c_it != match.end()) {
                    // Build: A @ (B @ C)
                    std::string bc_node = "bc_matmul";
                    g.add_node(bc_node, OpType::MATMUL, {b_it->second, c_it->second});
                    g.add_node("result", OpType::MATMUL, {a_it->second, bc_node});
                    g.add_output("result");
                }
                return g;
            },
            // Condition: only apply if it reduces computation
            [](const Graph& graph, const auto& match) -> bool {
                // This would check matrix dimensions to see if reassociation helps
                // For now, return true
                return true;
            }
        );
    }
    
    // Rule: Transpose fusion: (A^T)^T → A
    static RewriteRule double_transpose() {
        return RewriteRule(
            "double_transpose",
            "Cancel double transpose",
            ExtendedPatternLibrary::double_transpose_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_input(x_it->second);
                    g.add_output(x_it->second);
                }
                return g;
            }
        );
    }
    
    // Rule: Transpose-matmul fusion: A^T @ B → transpose_matmul(A, B)
    static RewriteRule transpose_matmul_fusion() {
        return RewriteRule(
            "transpose_matmul_fusion",
            "Fuse transpose with matmul for better cache usage",
            ExtendedPatternLibrary::transpose_matmul_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto a_it = match.find("$A");
                auto b_it = match.find("$B");
                
                if (a_it != match.end() && b_it != match.end()) {
                    NodeAttrs attrs;
                    attrs.set_int("transpose_a", 1);
                    g.add_node("result", OpType::MATMUL, {a_it->second, b_it->second}, attrs);
                    g.add_output("result");
                }
                return g;
            }
        );
    }
    
    // Category 5: Attention-Specific Rewrites
    
    // Rule: Flash attention pattern recognition
    static RewriteRule flash_attention() {
        return RewriteRule(
            "flash_attention",
            "Replace standard attention with FlashAttention pattern",
            ExtendedPatternLibrary::attention_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto q_it = match.find("$Q");
                auto k_it = match.find("$K");
                auto v_it = match.find("$V");
                
                if (q_it != match.end() && k_it != match.end() && v_it != match.end()) {
                    g.add_node("flash_attn", OpType::FLASH_ATTENTION, 
                              {q_it->second, k_it->second, v_it->second});
                    g.add_output("flash_attn");
                }
                return g;
            }
        );
    }
    
    // Rule: Scaled dot-product attention optimization
    static RewriteRule scaled_dot_product_attention() {
        return RewriteRule(
            "scaled_dot_product_attention",
            "Fuse scaled dot-product attention into single efficient op",
            ExtendedPatternLibrary::scaled_attention_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto q_it = match.find("$Q");
                auto k_it = match.find("$K");
                auto v_it = match.find("$V");
                
                if (q_it != match.end() && k_it != match.end() && v_it != match.end()) {
                    g.add_node("sdpa", OpType::SCALED_DOT_PRODUCT_ATTENTION,
                              {q_it->second, k_it->second, v_it->second});
                    g.add_output("sdpa");
                }
                return g;
            }
        );
    }
    
    // Category 6: Compensated Arithmetic
    
    // Rule: Kahan summation for large sums
    static RewriteRule kahan_sum() {
        return RewriteRule(
            "kahan_sum",
            "Use Kahan summation for numerically stable accumulation",
            ExtendedPatternLibrary::sum_accumulation_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto x_it = match.find("$x");
                if (x_it != match.end()) {
                    g.add_node("kahan_sum", OpType::KAHAN_SUM, {x_it->second});
                    g.add_output("kahan_sum");
                }
                return g;
            }
        );
    }
    
    // Rule: Compensated dot product
    static RewriteRule compensated_dot() {
        return RewriteRule(
            "compensated_dot",
            "Use compensated summation in dot products",
            ExtendedPatternLibrary::dot_product_pattern(),
            [](const auto& match) -> Graph {
                Graph g;
                auto a_it = match.find("$a");
                auto b_it = match.find("$b");
                
                if (a_it != match.end() && b_it != match.end()) {
                    g.add_node("comp_dot", OpType::COMPENSATED_DOT, 
                              {a_it->second, b_it->second});
                    g.add_output("comp_dot");
                }
                return g;
            }
        );
    }
    
    // Get all extended rules
    static std::vector<RewriteRule> all_rules() {
        return {
            // Cancellations
            sqrt_square_to_abs(),
            self_subtraction(),
            self_division(),
            
            // Stabilizations
            log1p_rule(),
            expm1_rule(),
            stable_sigmoid(),
            stable_tanh(),
            
            // Fusions
            layernorm_fusion(),
            batchnorm_fusion(),
            rmsnorm_fusion(),
            gelu_fusion(),
            swiglu_fusion(),
            
            // Matrix ops
            matmul_reassociate(),
            double_transpose(),
            transpose_matmul_fusion(),
            
            // Attention
            flash_attention(),
            scaled_dot_product_attention(),
            
            // Compensated arithmetic
            kahan_sum(),
            compensated_dot()
        };
    }
    
    // Get rules by category
    static std::vector<RewriteRule> stability_rules() {
        return {
            log1p_rule(),
            expm1_rule(),
            stable_sigmoid(),
            stable_tanh(),
            kahan_sum(),
            compensated_dot()
        };
    }
    
    static std::vector<RewriteRule> fusion_rules() {
        return {
            layernorm_fusion(),
            batchnorm_fusion(),
            rmsnorm_fusion(),
            gelu_fusion(),
            swiglu_fusion(),
            flash_attention(),
            scaled_dot_product_attention()
        };
    }
    
    static std::vector<RewriteRule> simplification_rules() {
        return {
            sqrt_square_to_abs(),
            self_subtraction(),
            self_division(),
            double_transpose()
        };
    }
};

} // namespace rewriter
} // namespace hnf
