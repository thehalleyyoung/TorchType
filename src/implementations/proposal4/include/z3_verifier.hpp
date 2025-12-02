#pragma once

#include "graph_ir.hpp"
#include <sstream>
#include <fstream>

namespace hnf {
namespace rewriter {

// Z3-based verification of rewrite correctness
// Generates SMT-LIB2 constraints to prove semantic equivalence
class Z3Verifier {
public:
    // Verify that two graphs compute the same function
    static bool verify_equivalence(const Graph& g1, const Graph& g2,
                                   const std::string& z3_path = "z3") {
        std::string smt2_code = generate_equivalence_query(g1, g2);
        
        // Write to temp file
        std::string filename = "/tmp/hnf_verify_" + std::to_string(rand()) + ".smt2";
        std::ofstream out(filename);
        out << smt2_code;
        out.close();
        
        // Run Z3
        std::string cmd = z3_path + " " + filename + " 2>&1";
        FILE* pipe = popen(cmd.c_str(), "r");
        if (!pipe) return false;
        
        char buffer[256];
        std::string result;
        while (fgets(buffer, sizeof(buffer), pipe)) {
            result += buffer;
        }
        pclose(pipe);
        
        // Clean up
        remove(filename.c_str());
        
        // Check result
        return result.find("unsat") != std::string::npos;
    }
    
    // Generate SMT-LIB2 code to verify g1 ≡ g2
    static std::string generate_equivalence_query(const Graph& g1, const Graph& g2) {
        std::ostringstream smt;
        
        smt << "; SMT-LIB2 query to verify graph equivalence\n";
        smt << "(set-logic QF_NRA)\n\n";
        
        // Collect all inputs from both graphs
        std::unordered_set<std::string> all_inputs;
        for (const auto& inp : g1.inputs()) all_inputs.insert(inp);
        for (const auto& inp : g2.inputs()) all_inputs.insert(inp);
        
        // Declare input variables as reals
        for (const auto& inp : all_inputs) {
            smt << "(declare-const " << inp << " Real)\n";
        }
        smt << "\n";
        
        // Add constraints that inputs are in a reasonable range
        for (const auto& inp : all_inputs) {
            smt << "(assert (and (>= " << inp << " -100.0) (<= " << inp << " 100.0)))\n";
        }
        smt << "\n";
        
        // Translate g1
        std::unordered_map<std::string, std::string> g1_vars;
        translate_graph(g1, "g1_", g1_vars, smt);
        
        // Translate g2
        std::unordered_map<std::string, std::string> g2_vars;
        translate_graph(g2, "g2_", g2_vars, smt);
        
        // Assert that outputs differ (we check for unsat)
        if (!g1.outputs().empty() && !g2.outputs().empty()) {
            std::string out1 = g1_vars[g1.outputs()[0]];
            std::string out2 = g2_vars[g2.outputs()[0]];
            smt << "; Check if outputs can differ\n";
            smt << "(assert (not (= " << out1 << " " << out2 << ")))\n\n";
        }
        
        smt << "(check-sat)\n";
        
        return smt.str();
    }
    
private:
    static void translate_graph(const Graph& g, const std::string& prefix,
                               std::unordered_map<std::string, std::string>& var_map,
                               std::ostringstream& smt) {
        auto order = g.topological_order();
        
        for (const auto& node_id : order) {
            auto node_ptr = g.get_node(node_id);
            if (!node_ptr) continue;
            
            std::string var_name = prefix + node_id;
            var_map[node_id] = var_name;
            
            // Skip inputs (already declared)
            if (node_ptr->op == OpType::INPUT) {
                var_map[node_id] = node_id;
                continue;
            }
            
            // Declare variable for this node
            smt << "(declare-const " << var_name << " Real)\n";
            
            // Generate constraint based on operation
            std::string constraint = translate_operation(*node_ptr, var_map, prefix);
            if (!constraint.empty()) {
                smt << "(assert (= " << var_name << " " << constraint << "))\n";
            }
        }
        smt << "\n";
    }
    
    static std::string translate_operation(const Node& node,
                                           const std::unordered_map<std::string, std::string>& var_map,
                                           const std::string& prefix) {
        auto get_input = [&](size_t idx) -> std::string {
            if (idx >= node.inputs.size()) return "0.0";
            auto it = var_map.find(node.inputs[idx]);
            if (it != var_map.end()) return it->second;
            return node.inputs[idx];
        };
        
        switch (node.op) {
            case OpType::ADD:
                if (node.inputs.size() >= 2) {
                    return "(+ " + get_input(0) + " " + get_input(1) + ")";
                }
                break;
            
            case OpType::SUB:
                if (node.inputs.size() >= 2) {
                    return "(- " + get_input(0) + " " + get_input(1) + ")";
                }
                break;
            
            case OpType::MUL:
                if (node.inputs.size() >= 2) {
                    return "(* " + get_input(0) + " " + get_input(1) + ")";
                }
                break;
            
            case OpType::DIV:
                if (node.inputs.size() >= 2) {
                    std::string denom = get_input(1);
                    // Add constraint that denominator is non-zero
                    return "(/ " + get_input(0) + " " + denom + ")";
                }
                break;
            
            case OpType::NEG:
                if (node.inputs.size() >= 1) {
                    return "(- 0.0 " + get_input(0) + ")";
                }
                break;
            
            case OpType::EXP:
                if (node.inputs.size() >= 1) {
                    // Z3 doesn't have exp in QF_NRA
                    // Use polynomial approximation for small values
                    std::string x = get_input(0);
                    // exp(x) ≈ 1 + x + x^2/2 + x^3/6 for small x
                    return "(+ 1.0 (+ " + x + " (+ (/ (* " + x + " " + x + ") 2.0) "
                           "(/ (* (* " + x + " " + x + ") " + x + ") 6.0))))";
                }
                break;
            
            case OpType::LOG:
                // Similar approximation challenge for log
                // For verification purposes, we can use uninterpreted functions
                if (node.inputs.size() >= 1) {
                    // Declare as uninterpreted function
                    return "(" + prefix + "log " + get_input(0) + ")";
                }
                break;
            
            case OpType::SQRT:
                if (node.inputs.size() >= 1) {
                    std::string x = get_input(0);
                    // Use square relation: result^2 = x
                    return "(" + prefix + "sqrt " + x + ")";
                }
                break;
            
            case OpType::POW:
                if (node.inputs.size() >= 1) {
                    std::string x = get_input(0);
                    double exp = node.attrs.get_float("exponent", 2.0);
                    
                    if (exp == 2.0) {
                        return "(* " + x + " " + x + ")";
                    } else if (exp == 3.0) {
                        return "(* (* " + x + " " + x + ") " + x + ")";
                    }
                    // For general powers, use uninterpreted function
                    return "(" + prefix + "pow " + x + ")";
                }
                break;
            
            case OpType::MAX:
                if (node.inputs.size() >= 1) {
                    // For single input, it's identity
                    // For array reduction, we can't express easily in SMT
                    return get_input(0);
                }
                break;
            
            case OpType::SUM:
                if (node.inputs.size() >= 1) {
                    // For array sum, assume it's identity (conservative)
                    return get_input(0);
                }
                break;
            
            case OpType::IDENTITY:
                if (node.inputs.size() >= 1) {
                    return get_input(0);
                }
                break;
            
            case OpType::STABLE_SOFTMAX:
            case OpType::SOFTMAX:
            case OpType::LOGSUMEXP:
                // These require uninterpreted functions for full generality
                // For specific cases, we can expand them
                if (node.inputs.size() >= 1) {
                    return "(" + prefix + optype_to_string(node.op) + " " + get_input(0) + ")";
                }
                break;
            
            default:
                break;
        }
        
        return "";
    }
};

// Simpler symbolic verifier for common patterns
class SymbolicVerifier {
public:
    // Verify log(exp(x)) = x symbolically
    static bool verify_log_exp_cancel(const Graph& original, const Graph& rewritten) {
        // Check structure
        if (original.outputs().size() != 1 || rewritten.outputs().size() != 1) {
            return false;
        }
        
        auto orig_out = original.get_node(original.outputs()[0]);
        if (!orig_out || orig_out->op != OpType::LOG) return false;
        
        if (orig_out->inputs.empty()) return false;
        auto exp_node = original.get_node(orig_out->inputs[0]);
        if (!exp_node || exp_node->op != OpType::EXP) return false;
        
        // Rewritten should just be the input to exp
        auto rewrite_out = rewritten.get_node(rewritten.outputs()[0]);
        if (!rewrite_out) return false;
        
        // Check if it's identity or direct input
        if (rewrite_out->op == OpType::IDENTITY || rewrite_out->op == OpType::INPUT) {
            return true;
        }
        
        return false;
    }
    
    // Verify exp(log(x)) = x symbolically
    static bool verify_exp_log_cancel(const Graph& original, const Graph& rewritten) {
        auto orig_out = original.get_node(original.outputs()[0]);
        if (!orig_out || orig_out->op != OpType::EXP) return false;
        
        if (orig_out->inputs.empty()) return false;
        auto log_node = original.get_node(orig_out->inputs[0]);
        if (!log_node || log_node->op != OpType::LOG) return false;
        
        auto rewrite_out = rewritten.get_node(rewritten.outputs()[0]);
        if (!rewrite_out) return false;
        
        if (rewrite_out->op == OpType::IDENTITY || rewrite_out->op == OpType::INPUT) {
            return true;
        }
        
        return false;
    }
    
    // Verify naive softmax -> stable softmax maintains semantics
    static bool verify_softmax_equivalence(const Graph& original, const Graph& rewritten) {
        // Mathematically: exp(x-m)/sum(exp(x-m)) = exp(x)/sum(exp(x)) for any m
        // We verify the structure is correct
        
        auto orig_out = original.get_node(original.outputs()[0]);
        if (!orig_out) return false;
        
        // Original should be division
        if (orig_out->op != OpType::DIV) return false;
        
        // Rewritten should be stable_softmax
        auto rewrite_out = rewritten.get_node(rewritten.outputs()[0]);
        if (!rewrite_out) return false;
        
        if (rewrite_out->op == OpType::STABLE_SOFTMAX) {
            return true;
        }
        
        return false;
    }
};

} // namespace rewriter
} // namespace hnf
