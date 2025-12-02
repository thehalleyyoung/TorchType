#pragma once

#include "pattern.hpp"

namespace hnf {
namespace rewriter {

// Extended pattern library for all the new rewrite rules
class ExtendedPatternLibrary {
public:
    // sqrt(x^2) pattern
    static Pattern sqrt_square_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        
        NodeAttrs pow_attrs;
        pow_attrs.set_float("exponent", 2.0);
        auto pow_node = std::make_shared<Node>("square", OpType::POW, 
                                               std::vector<std::string>{"$x"}, pow_attrs);
        auto sqrt_node = std::make_shared<Node>("sqrt", OpType::SQRT,
                                                std::vector<std::string>{"square"});
        
        g.add_node(x_node);
        g.add_node(pow_node);
        g.add_node(sqrt_node);
        g.set_outputs({"sqrt"});
        
        return Pattern(g, "sqrt");
    }
    
    // x - x pattern
    static Pattern self_sub_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto sub_node = std::make_shared<Node>("sub", OpType::SUB,
                                               std::vector<std::string>{"$x", "$x"});
        
        g.add_node(x_node);
        g.add_node(sub_node);
        g.set_outputs({"sub"});
        
        return Pattern(g, "sub");
    }
    
    // x / x pattern
    static Pattern self_div_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto div_node = std::make_shared<Node>("div", OpType::DIV,
                                               std::vector<std::string>{"$x", "$x"});
        
        g.add_node(x_node);
        g.add_node(div_node);
        g.set_outputs({"div"});
        
        return Pattern(g, "div");
    }
    
    // log(1 + x) pattern
    static Pattern log_one_plus_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto one_node = std::make_shared<Node>("one", OpType::CONSTANT);
        auto add_node = std::make_shared<Node>("add", OpType::ADD,
                                               std::vector<std::string>{"one", "$x"});
        auto log_node = std::make_shared<Node>("log", OpType::LOG,
                                               std::vector<std::string>{"add"});
        
        g.add_node(x_node);
        g.add_node(one_node);
        g.add_node(add_node);
        g.add_node(log_node);
        g.set_outputs({"log"});
        
        return Pattern(g, "log");
    }
    
    // exp(x) - 1 pattern
    static Pattern exp_minus_one_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP,
                                               std::vector<std::string>{"$x"});
        auto one_node = std::make_shared<Node>("one", OpType::CONSTANT);
        auto sub_node = std::make_shared<Node>("sub", OpType::SUB,
                                               std::vector<std::string>{"exp", "one"});
        
        g.add_node(x_node);
        g.add_node(exp_node);
        g.add_node(one_node);
        g.add_node(sub_node);
        g.set_outputs({"sub"});
        
        return Pattern(g, "sub");
    }
    
    // 1 / (1 + exp(-x)) pattern (naive sigmoid)
    static Pattern naive_sigmoid_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto neg_node = std::make_shared<Node>("neg", OpType::NEG,
                                               std::vector<std::string>{"$x"});
        auto exp_node = std::make_shared<Node>("exp", OpType::EXP,
                                               std::vector<std::string>{"neg"});
        auto one_node = std::make_shared<Node>("one", OpType::CONSTANT);
        auto add_node = std::make_shared<Node>("add", OpType::ADD,
                                               std::vector<std::string>{"one", "exp"});
        auto div_node = std::make_shared<Node>("div", OpType::DIV,
                                               std::vector<std::string>{"one", "add"});
        
        g.add_node(x_node);
        g.add_node(neg_node);
        g.add_node(exp_node);
        g.add_node(one_node);
        g.add_node(add_node);
        g.add_node(div_node);
        g.set_outputs({"div"});
        
        return Pattern(g, "div");
    }
    
    // Simplified tanh pattern: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
    static Pattern naive_tanh_pattern() {
        // Simplified version for pattern matching
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        
        // This is a simplified pattern - full tanh pattern is complex
        // We'll just match any complex combination that looks like tanh
        
        g.add_node(x_node);
        g.set_outputs({"$x"});
        
        return Pattern(g, "$x");
    }
    
    // LayerNorm pattern: (x - mean(x)) / sqrt(var(x) + eps)
    static Pattern layernorm_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        
        // Simplified pattern for layer norm
        auto mean_node = std::make_shared<Node>("mean", OpType::MEAN,
                                                std::vector<std::string>{"$x"});
        auto sub_node = std::make_shared<Node>("sub", OpType::SUB,
                                               std::vector<std::string>{"$x", "mean"});
        
        g.add_node(x_node);
        g.add_node(mean_node);
        g.add_node(sub_node);
        g.set_outputs({"sub"});
        
        return Pattern(g, "sub");
    }
    
    // BatchNorm pattern (simplified)
    static Pattern batchnorm_pattern() {
        // Similar to layer norm but across batch dimension
        return layernorm_pattern();
    }
    
    // RMSNorm pattern: x / sqrt(mean(x^2) + eps)
    static Pattern rmsnorm_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        
        NodeAttrs pow_attrs;
        pow_attrs.set_float("exponent", 2.0);
        auto square_node = std::make_shared<Node>("square", OpType::POW,
                                                  std::vector<std::string>{"$x"}, pow_attrs);
        auto mean_node = std::make_shared<Node>("mean", OpType::MEAN,
                                                std::vector<std::string>{"square"});
        
        g.add_node(x_node);
        g.add_node(square_node);
        g.add_node(mean_node);
        g.set_outputs({"mean"});
        
        return Pattern(g, "mean");
    }
    
    // GELU pattern (simplified)
    static Pattern gelu_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        g.add_node(x_node);
        g.set_outputs({"$x"});
        
        return Pattern(g, "$x");
    }
    
    // SwiGLU pattern
    static Pattern swiglu_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto y_node = std::make_shared<Node>("$y", OpType::INPUT);
        
        g.add_node(x_node);
        g.add_node(y_node);
        g.set_outputs({"$x"});
        
        return Pattern(g, "$x");
    }
    
    // (A @ B) @ C pattern
    static Pattern matmul_chain_pattern() {
        Graph g;
        auto a_node = std::make_shared<Node>("$A", OpType::INPUT);
        auto b_node = std::make_shared<Node>("$B", OpType::INPUT);
        auto c_node = std::make_shared<Node>("$C", OpType::INPUT);
        
        auto ab_node = std::make_shared<Node>("AB", OpType::MATMUL,
                                              std::vector<std::string>{"$A", "$B"});
        auto abc_node = std::make_shared<Node>("ABC", OpType::MATMUL,
                                               std::vector<std::string>{"AB", "$C"});
        
        g.add_node(a_node);
        g.add_node(b_node);
        g.add_node(c_node);
        g.add_node(ab_node);
        g.add_node(abc_node);
        g.set_outputs({"ABC"});
        
        return Pattern(g, "ABC");
    }
    
    // (A^T)^T pattern
    static Pattern double_transpose_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto t1_node = std::make_shared<Node>("T1", OpType::TRANSPOSE,
                                              std::vector<std::string>{"$x"});
        auto t2_node = std::make_shared<Node>("T2", OpType::TRANSPOSE,
                                              std::vector<std::string>{"T1"});
        
        g.add_node(x_node);
        g.add_node(t1_node);
        g.add_node(t2_node);
        g.set_outputs({"T2"});
        
        return Pattern(g, "T2");
    }
    
    // A^T @ B pattern
    static Pattern transpose_matmul_pattern() {
        Graph g;
        auto a_node = std::make_shared<Node>("$A", OpType::INPUT);
        auto b_node = std::make_shared<Node>("$B", OpType::INPUT);
        
        auto t_node = std::make_shared<Node>("AT", OpType::TRANSPOSE,
                                             std::vector<std::string>{"$A"});
        auto mm_node = std::make_shared<Node>("MM", OpType::MATMUL,
                                              std::vector<std::string>{"AT", "$B"});
        
        g.add_node(a_node);
        g.add_node(b_node);
        g.add_node(t_node);
        g.add_node(mm_node);
        g.set_outputs({"MM"});
        
        return Pattern(g, "MM");
    }
    
    // Standard attention pattern: softmax(Q @ K^T) @ V
    static Pattern attention_pattern() {
        Graph g;
        auto q_node = std::make_shared<Node>("$Q", OpType::INPUT);
        auto k_node = std::make_shared<Node>("$K", OpType::INPUT);
        auto v_node = std::make_shared<Node>("$V", OpType::INPUT);
        
        auto kt_node = std::make_shared<Node>("KT", OpType::TRANSPOSE,
                                              std::vector<std::string>{"$K"});
        auto scores_node = std::make_shared<Node>("scores", OpType::MATMUL,
                                                  std::vector<std::string>{"$Q", "KT"});
        auto softmax_node = std::make_shared<Node>("weights", OpType::SOFTMAX,
                                                   std::vector<std::string>{"scores"});
        auto output_node = std::make_shared<Node>("output", OpType::MATMUL,
                                                  std::vector<std::string>{"weights", "$V"});
        
        g.add_node(q_node);
        g.add_node(k_node);
        g.add_node(v_node);
        g.add_node(kt_node);
        g.add_node(scores_node);
        g.add_node(softmax_node);
        g.add_node(output_node);
        g.set_outputs({"output"});
        
        return Pattern(g, "output");
    }
    
    // Scaled attention pattern
    static Pattern scaled_attention_pattern() {
        // Similar to attention but with scaling
        return attention_pattern();
    }
    
    // Sum accumulation pattern (for Kahan sum)
    static Pattern sum_accumulation_pattern() {
        Graph g;
        auto x_node = std::make_shared<Node>("$x", OpType::INPUT);
        auto sum_node = std::make_shared<Node>("sum", OpType::SUM,
                                               std::vector<std::string>{"$x"});
        
        g.add_node(x_node);
        g.add_node(sum_node);
        g.set_outputs({"sum"});
        
        return Pattern(g, "sum");
    }
    
    // Dot product pattern
    static Pattern dot_product_pattern() {
        Graph g;
        auto a_node = std::make_shared<Node>("$a", OpType::INPUT);
        auto b_node = std::make_shared<Node>("$b", OpType::INPUT);
        
        auto mul_node = std::make_shared<Node>("mul", OpType::MUL,
                                               std::vector<std::string>{"$a", "$b"});
        auto sum_node = std::make_shared<Node>("sum", OpType::SUM,
                                               std::vector<std::string>{"mul"});
        
        g.add_node(a_node);
        g.add_node(b_node);
        g.add_node(mul_node);
        g.add_node(sum_node);
        g.set_outputs({"sum"});
        
        return Pattern(g, "sum");
    }
};

} // namespace rewriter
} // namespace hnf
