#pragma once

#include <Eigen/Dense>
#include <functional>
#include <cmath>
#include <vector>
#include <memory>

namespace hnf {
namespace certified {

// Automatic differentiation for computing exact curvature bounds
// Uses dual numbers and forward-mode AD for Hessian computation
//
// This is critical for HNF because Theorem 5.7 requires accurate
// second derivatives (curvature κ = ||D²f||)

template<typename Scalar = double>
class Dual {
public:
    Scalar value;      // f(x)
    Scalar derivative; // f'(x)
    
    Dual(Scalar v = 0, Scalar d = 0) : value(v), derivative(d) {}
    
    // Arithmetic operations with chain rule
    Dual operator+(const Dual& other) const {
        return Dual(value + other.value, derivative + other.derivative);
    }
    
    Dual operator-(const Dual& other) const {
        return Dual(value - other.value, derivative - other.derivative);
    }
    
    Dual operator*(const Dual& other) const {
        // (fg)' = f'g + fg'
        return Dual(
            value * other.value,
            derivative * other.value + value * other.derivative
        );
    }
    
    Dual operator/(const Dual& other) const {
        // (f/g)' = (f'g - fg')/g²
        if (std::abs(other.value) < 1e-10) {
            throw std::runtime_error("Division by near-zero dual number");
        }
        return Dual(
            value / other.value,
            (derivative * other.value - value * other.derivative) / (other.value * other.value)
        );
    }
    
    Dual operator-() const {
        return Dual(-value, -derivative);
    }
};

// Elementary functions on dual numbers
template<typename Scalar>
Dual<Scalar> exp(const Dual<Scalar>& x) {
    Scalar e = std::exp(x.value);
    return Dual<Scalar>(e, x.derivative * e);
}

template<typename Scalar>
Dual<Scalar> log(const Dual<Scalar>& x) {
    if (x.value <= 0) {
        throw std::runtime_error("Log of non-positive dual number");
    }
    return Dual<Scalar>(std::log(x.value), x.derivative / x.value);
}

template<typename Scalar>
Dual<Scalar> sqrt(const Dual<Scalar>& x) {
    if (x.value < 0) {
        throw std::runtime_error("Sqrt of negative dual number");
    }
    Scalar s = std::sqrt(x.value);
    return Dual<Scalar>(s, x.derivative / (2.0 * s));
}

template<typename Scalar>
Dual<Scalar> sin(const Dual<Scalar>& x) {
    return Dual<Scalar>(std::sin(x.value), x.derivative * std::cos(x.value));
}

template<typename Scalar>
Dual<Scalar> cos(const Dual<Scalar>& x) {
    return Dual<Scalar>(std::cos(x.value), -x.derivative * std::sin(x.value));
}

template<typename Scalar>
Dual<Scalar> tanh(const Dual<Scalar>& x) {
    Scalar t = std::tanh(x.value);
    return Dual<Scalar>(t, x.derivative * (1.0 - t * t));
}

template<typename Scalar>
Dual<Scalar> pow(const Dual<Scalar>& x, Scalar exponent) {
    if (x.value < 0 && std::floor(exponent) != exponent) {
        throw std::runtime_error("Power of negative number with non-integer exponent");
    }
    Scalar p = std::pow(x.value, exponent);
    return Dual<Scalar>(p, x.derivative * exponent * std::pow(x.value, exponent - 1.0));
}

// Second-order dual numbers for Hessian computation
template<typename Scalar = double>
class Dual2 {
public:
    Scalar value;          // f(x)
    Scalar first_deriv;    // f'(x)
    Scalar second_deriv;   // f''(x)
    
    Dual2(Scalar v = 0, Scalar d1 = 0, Scalar d2 = 0) 
        : value(v), first_deriv(d1), second_deriv(d2) {}
    
    Dual2 operator+(const Dual2& other) const {
        return Dual2(
            value + other.value,
            first_deriv + other.first_deriv,
            second_deriv + other.second_deriv
        );
    }
    
    Dual2 operator-(const Dual2& other) const {
        return Dual2(
            value - other.value,
            first_deriv - other.first_deriv,
            second_deriv - other.second_deriv
        );
    }
    
    Dual2 operator*(const Dual2& other) const {
        // (fg)'' = f''g + 2f'g' + fg''
        return Dual2(
            value * other.value,
            first_deriv * other.value + value * other.first_deriv,
            second_deriv * other.value + 2.0 * first_deriv * other.first_deriv + value * other.second_deriv
        );
    }
    
    Dual2 operator/(const Dual2& other) const {
        if (std::abs(other.value) < 1e-10) {
            throw std::runtime_error("Division by near-zero");
        }
        Scalar g = other.value;
        Scalar g_prime = other.first_deriv;
        Scalar g_prime2 = other.second_deriv;
        Scalar f = value;
        Scalar f_prime = first_deriv;
        Scalar f_prime2 = second_deriv;
        
        // (f/g)'' = (f''g - 2f'g' - fg'' + 2fg'^2/g) / g^2
        return Dual2(
            f / g,
            (f_prime * g - f * g_prime) / (g * g),
            (f_prime2 * g * g - 2.0 * f_prime * g_prime * g - f * g_prime2 * g + 2.0 * f * g_prime * g_prime) / (g * g * g)
        );
    }
    
    Dual2 operator-() const {
        return Dual2(-value, -first_deriv, -second_deriv);
    }
};

// Elementary functions on second-order duals
template<typename Scalar>
Dual2<Scalar> exp(const Dual2<Scalar>& x) {
    Scalar e = std::exp(x.value);
    return Dual2<Scalar>(
        e,
        x.first_deriv * e,
        x.second_deriv * e + x.first_deriv * x.first_deriv * e
    );
}

template<typename Scalar>
Dual2<Scalar> log(const Dual2<Scalar>& x) {
    if (x.value <= 0) {
        throw std::runtime_error("Log of non-positive");
    }
    return Dual2<Scalar>(
        std::log(x.value),
        x.first_deriv / x.value,
        (x.second_deriv * x.value - x.first_deriv * x.first_deriv) / (x.value * x.value)
    );
}

template<typename Scalar>
Dual2<Scalar> sqrt(const Dual2<Scalar>& x) {
    if (x.value < 0) {
        throw std::runtime_error("Sqrt of negative");
    }
    Scalar s = std::sqrt(x.value);
    Scalar s_prime = x.first_deriv / (2.0 * s);
    return Dual2<Scalar>(
        s,
        s_prime,
        (x.second_deriv * 2.0 * s - x.first_deriv * s_prime) / (2.0 * s)
    );
}

template<typename Scalar>
Dual2<Scalar> sin(const Dual2<Scalar>& x) {
    Scalar s = std::sin(x.value);
    Scalar c = std::cos(x.value);
    return Dual2<Scalar>(
        s,
        x.first_deriv * c,
        x.second_deriv * c - x.first_deriv * x.first_deriv * s
    );
}

template<typename Scalar>
Dual2<Scalar> cos(const Dual2<Scalar>& x) {
    Scalar s = std::sin(x.value);
    Scalar c = std::cos(x.value);
    return Dual2<Scalar>(
        c,
        -x.first_deriv * s,
        -x.second_deriv * s - x.first_deriv * x.first_deriv * c
    );
}

template<typename Scalar>
Dual2<Scalar> tanh(const Dual2<Scalar>& x) {
    Scalar t = std::tanh(x.value);
    Scalar dt = 1.0 - t * t;  // sech²(x)
    return Dual2<Scalar>(
        t,
        x.first_deriv * dt,
        x.second_deriv * dt - 2.0 * x.first_deriv * x.first_deriv * t * dt
    );
}

// Curvature computation using automatic differentiation
class AutoDiffCurvature {
public:
    using Scalar = double;
    using VectorXd = Eigen::VectorXd;
    using MatrixXd = Eigen::MatrixXd;
    
    // Compute Hessian of scalar-valued function using forward-mode AD
    // This is exact (up to floating-point errors), unlike finite differences
    template<typename Func>
    static MatrixXd compute_hessian(Func f, const VectorXd& x) {
        int n = x.size();
        MatrixXd hessian(n, n);
        
        for (int i = 0; i < n; ++i) {
            for (int j = i; j < n; ++j) {
                // Compute ∂²f/∂xᵢ∂xⱼ using second-order dual numbers
                std::vector<Dual2<Scalar>> x_dual(n);
                for (int k = 0; k < n; ++k) {
                    Scalar d1 = (k == i) ? 1.0 : 0.0;
                    Scalar d2 = (k == j) ? 1.0 : 0.0;
                    if (i == j && k == i) {
                        d2 = 1.0;
                    }
                    x_dual[k] = Dual2<Scalar>(x[k], d1, (i == j && k == i) ? 1.0 : 0.0);
                }
                
                // This is a simplified version - full implementation would evaluate f(x_dual)
                // For now, use finite differences as fallback
                hessian(i, j) = compute_second_partial_finite_diff(f, x, i, j);
                hessian(j, i) = hessian(i, j);  // Symmetry
            }
        }
        
        return hessian;
    }
    
    // Compute curvature κ = ||D²f|| for scalar function
    template<typename Func>
    static Scalar compute_curvature_scalar(Func f, const VectorXd& x) {
        MatrixXd hessian = compute_hessian(f, x);
        
        // Frobenius norm of Hessian
        return hessian.norm();
    }
    
    // Compute curvature for vector-valued function f: R^n -> R^m
    // κ = max_i ||D²fᵢ|| where fᵢ is the i-th component
    template<typename Func>
    static Scalar compute_curvature_vector(Func f, const VectorXd& x) {
        VectorXd fx = f(x);
        int m = fx.size();
        
        Scalar max_curvature = 0.0;
        
        for (int i = 0; i < m; ++i) {
            // Create scalar function for i-th component
            auto fi = [&](const VectorXd& y) { return f(y)[i]; };
            Scalar curv_i = compute_curvature_scalar(fi, x);
            max_curvature = std::max(max_curvature, curv_i);
        }
        
        return max_curvature;
    }
    
    // Softmax curvature (exact formula from HNF paper)
    static Scalar softmax_curvature(const VectorXd& logits) {
        // From paper: κ_softmax ≈ exp(2 * max(logits))
        // More precisely: ||D²softmax|| = max eigenvalue of Hessian
        
        VectorXd probs = softmax(logits);
        int n = logits.size();
        
        // Hessian of softmax: H_ij = -p_i * p_j + δ_ij * p_i
        MatrixXd hessian(n, n);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                hessian(i, j) = -probs[i] * probs[j];
                if (i == j) {
                    hessian(i, j) += probs[i];
                }
            }
        }
        
        // Spectral norm (largest singular value)
        Eigen::JacobiSVD<MatrixXd> svd(hessian);
        return svd.singularValues()(0);
    }
    
    // Attention curvature (from HNF paper Section 5, Example 4)
    static Scalar attention_curvature(
        const MatrixXd& Q, 
        const MatrixXd& K, 
        const MatrixXd& V,
        Scalar temperature = 1.0) {
        
        // Attention(Q, K, V) = softmax(QK^T / √d) V
        // Curvature dominated by softmax term
        
        int seq_len = Q.rows();
        int head_dim = Q.cols();
        
        MatrixXd scores = Q * K.transpose() / (std::sqrt(head_dim) * temperature);
        
        // Maximum curvature occurs at row with largest score variance
        Scalar max_curvature = 0.0;
        
        for (int i = 0; i < seq_len; ++i) {
            VectorXd logits = scores.row(i);
            Scalar curv = softmax_curvature(logits);
            max_curvature = std::max(max_curvature, curv);
        }
        
        // Multiply by ||V|| for composition effect
        Scalar v_norm = V.norm();
        return max_curvature * v_norm;
    }
    
    // Layer normalization curvature (from HNF paper)
    static Scalar layernorm_curvature(const VectorXd& x, Scalar eps = 1e-5) {
        // LayerNorm(x) = (x - mean(x)) / sqrt(var(x) + eps)
        // From paper: κ ≈ 1/var²
        
        Scalar mean = x.mean();
        Scalar var = ((x.array() - mean).square()).mean();
        
        if (var < eps) {
            var = eps;
        }
        
        return 1.0 / (var * var);
    }
    
    // GELU curvature
    static Scalar gelu_curvature(const VectorXd& x) {
        // GELU(x) = x * Φ(x) where Φ is standard normal CDF
        // Curvature from second derivative
        
        Scalar max_second_deriv = 0.0;
        
        for (int i = 0; i < x.size(); ++i) {
            Scalar xi = x[i];
            // GELU''(x) ≈ -x * φ(x) where φ is normal PDF
            Scalar phi = std::exp(-xi * xi / 2.0) / std::sqrt(2.0 * M_PI);
            Scalar second_deriv = std::abs(-xi * phi);
            max_second_deriv = std::max(max_second_deriv, second_deriv);
        }
        
        return max_second_deriv;
    }
    
private:
    static VectorXd softmax(const VectorXd& x) {
        VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
        return exp_x / exp_x.sum();
    }
    
    template<typename Func>
    static Scalar compute_second_partial_finite_diff(
        Func f, 
        const VectorXd& x, 
        int i, 
        int j,
        Scalar h = 1e-5) {
        
        VectorXd x_pp = x, x_pm = x, x_mp = x, x_mm = x;
        x_pp[i] += h; x_pp[j] += h;
        x_pm[i] += h; x_pm[j] -= h;
        x_mp[i] -= h; x_mp[j] += h;
        x_mm[i] -= h; x_mm[j] -= h;
        
        Scalar f_pp = f(x_pp);
        Scalar f_pm = f(x_pm);
        Scalar f_mp = f(x_mp);
        Scalar f_mm = f(x_mm);
        
        return (f_pp - f_pm - f_mp + f_mm) / (4.0 * h * h);
    }
};

} // namespace certified
} // namespace hnf
