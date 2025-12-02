#pragma once

#include <torch/torch.h>
#include <memory>
#include <cmath>
#include <limits>
#include <functional>

namespace hnf {

// Hardware models representing different floating-point precisions
enum class HardwareModel {
    FLOAT16,    // IEEE 754 binary16: 10-bit mantissa
    BFLOAT16,   // Google bfloat16: 7-bit mantissa
    FLOAT32,    // IEEE 754 binary32: 23-bit mantissa
    FLOAT64,    // IEEE 754 binary64: 52-bit mantissa
    FLOAT128    // IEEE 754 binary128: 112-bit mantissa
};

// Get mantissa precision for hardware model
inline int mantissa_precision(HardwareModel H) {
    switch(H) {
        case HardwareModel::FLOAT16: return 10;
        case HardwareModel::BFLOAT16: return 7;
        case HardwareModel::FLOAT32: return 23;
        case HardwareModel::FLOAT64: return 52;
        case HardwareModel::FLOAT128: return 112;
        default: return 23;
    }
}

// Machine epsilon for hardware model
inline double machine_epsilon(HardwareModel H) {
    return std::pow(2.0, -mantissa_precision(H));
}

// Numerical Type: A tensor with precision metadata
// Represents a point in the category NMet from the HNF paper
class NumericalType {
public:
    torch::Tensor data;              // The underlying tensor (|A| in the paper)
    double lipschitz_constant;       // L_f: Lipschitz constant
    double curvature;                // κ^curv: curvature bound from Theorem 5.7
    double domain_diameter;          // D: diameter of the domain
    int precision_bits_required;     // p_min: from precision obstruction theorem
    HardwareModel hardware;          // Current hardware representation
    
    // Error propagation functional Φ_f(ε, H)
    // This is the key structure from Definition 3.3
    std::function<double(double, HardwareModel)> error_functional;
    
    NumericalType(
        const torch::Tensor& t,
        double L = 1.0,
        double kappa = 0.0,
        HardwareModel H = HardwareModel::FLOAT32
    ) : data(t), 
        lipschitz_constant(L), 
        curvature(kappa),
        hardware(H),
        domain_diameter(0.0),
        precision_bits_required(0)
    {
        // Default error functional: Φ_f(ε, H) = L·ε + Δ(H)
        error_functional = [L, kappa](double eps, HardwareModel H) {
            return L * eps + machine_epsilon(H);
        };
        
        // Compute domain diameter from tensor statistics
        if (t.numel() > 0) {
            auto flat = t.flatten();
            double max_val = flat.max().item<double>();
            double min_val = flat.min().item<double>();
            domain_diameter = std::abs(max_val - min_val);
        }
        
        // Compute precision requirement using Theorem 5.7 (Precision Obstruction)
        // p ≥ log₂(c·κ·D²/ε) where c is a constant
        if (kappa > 0 && domain_diameter > 0) {
            double target_eps = 1e-6;  // Default target accuracy
            double c = 2.0;  // Conservative constant from the theorem
            precision_bits_required = std::ceil(
                std::log2(c * kappa * domain_diameter * domain_diameter / target_eps)
            );
        }
    }
    
    // Update precision requirement for specific target accuracy
    void update_precision_requirement(double target_eps) {
        if (curvature > 0 && domain_diameter > 0) {
            double c = 2.0;
            precision_bits_required = std::ceil(
                std::log2(c * curvature * domain_diameter * domain_diameter / target_eps)
            );
        }
    }
    
    // Check if current hardware is sufficient
    bool hardware_is_sufficient() const {
        return mantissa_precision(hardware) >= precision_bits_required;
    }
    
    // Recommend minimum hardware
    HardwareModel recommend_hardware() const {
        if (precision_bits_required <= 7) return HardwareModel::BFLOAT16;
        if (precision_bits_required <= 10) return HardwareModel::FLOAT16;
        if (precision_bits_required <= 23) return HardwareModel::FLOAT32;
        if (precision_bits_required <= 52) return HardwareModel::FLOAT64;
        return HardwareModel::FLOAT128;
    }
    
    // Compute actual error for given input error and hardware
    double propagate_error(double input_eps) const {
        return error_functional(input_eps, hardware);
    }
};

// Numerical Morphism: A function between numerical types
// Represents morphisms in NMet category
struct NumericalMorphism {
    std::string name;
    double lipschitz_constant;
    double curvature;
    
    // Error functional Φ_f as in Definition 3.3
    std::function<double(double, HardwareModel)> error_functional;
    
    // The actual computation
    std::function<torch::Tensor(const torch::Tensor&)> forward;
    
    NumericalMorphism(
        const std::string& n,
        double L,
        double kappa,
        std::function<torch::Tensor(const torch::Tensor&)> f
    ) : name(n), lipschitz_constant(L), curvature(kappa), forward(f) {
        // Construct error functional per Definition 3.3
        error_functional = [L, kappa](double eps, HardwareModel H) {
            double eps_mach = machine_epsilon(H);
            // Φ_f(ε, H) = L·ε + Δ_f(H) where Δ_f captures roundoff
            return L * eps + kappa * eps_mach;
        };
    }
    
    // Apply morphism to numerical type (composition in NMet)
    NumericalType operator()(const NumericalType& input) const {
        torch::Tensor output = forward(input.data);
        
        // Composition of Lipschitz constants: L_{g∘f} = L_g · L_f
        double composed_lipschitz = lipschitz_constant * input.lipschitz_constant;
        
        // Composition of curvature (Proposition in Section 5)
        // κ_{g∘f} ≤ κ_g·L_f² + κ_f·||Dg||
        double composed_curvature = 
            curvature * input.lipschitz_constant * input.lipschitz_constant +
            input.curvature * lipschitz_constant;
        
        // Composed error functional (Theorem 3.8: Stability Composition Theorem)
        // Φ_{g∘f}(ε, H) = Φ_g(Φ_f(ε, H), H) + L_g · Φ_f(ε, H)
        auto composed_error = [this, input](double eps, HardwareModel H) {
            double phi_f = input.error_functional(eps, H);
            double phi_g_of_phi_f = this->error_functional(phi_f, H);
            return phi_g_of_phi_f + this->lipschitz_constant * phi_f;
        };
        
        NumericalType result(output, composed_lipschitz, composed_curvature, input.hardware);
        result.error_functional = composed_error;
        
        // Propagate domain diameter (conservative upper bound)
        result.domain_diameter = lipschitz_constant * input.domain_diameter;
        
        return result;
    }
};

// Numerical Equivalence: (f, g, η, μ) from Definition 4.1
struct NumericalEquivalence {
    NumericalMorphism forward;
    NumericalMorphism backward;
    
    // Condition number of equivalence: cond_eq(f,g) = L_f · L_g
    double condition_number() const {
        return forward.lipschitz_constant * backward.lipschitz_constant;
    }
    
    // Check if this is a good equivalence (low condition number)
    bool is_well_conditioned(double threshold = 10.0) const {
        return condition_number() <= threshold;
    }
};

} // namespace hnf
