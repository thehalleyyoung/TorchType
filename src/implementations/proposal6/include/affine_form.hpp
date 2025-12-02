#pragma once

#include "interval.hpp"
#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

namespace hnf {
namespace certified {

// Affine Arithmetic: More precise than interval arithmetic
// Represents x as: x = x₀ + ε₁·x₁ + ε₂·x₂ + ... + εₙ·xₙ
// where each εᵢ ∈ [-1, 1] represents independent noise symbols
//
// This dramatically reduces overestimation compared to standard intervals
// by tracking correlations between values.
//
// Reference: "Self-validated numerical methods and applications" (de Figueiredo & Stolfi)
class AffineForm {
public:
    using Scalar = double;
    
    // Constructor from constant
    explicit AffineForm(Scalar value = 0.0) 
        : central_(value), next_noise_id_(0) {}
    
    // Constructor from interval [a, b]
    // Represents as (a+b)/2 + (b-a)/2 * ε
    explicit AffineForm(const Interval& interval) 
        : central_(interval.midpoint()),
          next_noise_id_(0) {
        if (interval.width() > 0) {
            Scalar radius = interval.width() / 2.0;
            int noise_id = allocate_noise_symbol();
            deviations_[noise_id] = radius;
        }
    }
    
    // Get central value
    Scalar central() const { return central_; }
    
    // Get total deviation (radius)
    Scalar radius() const {
        Scalar sum = 0.0;
        for (const auto& pair : deviations_) {
            sum += std::abs(pair.second);
        }
        return sum;
    }
    
    // Convert to interval
    Interval to_interval() const {
        Scalar rad = radius();
        return Interval(central_ - rad, central_ + rad);
    }
    
    // Get bounds
    Scalar lower() const { return central_ - radius(); }
    Scalar upper() const { return central_ + radius(); }
    
    // Arithmetic operations
    AffineForm operator+(const AffineForm& other) const {
        AffineForm result;
        result.central_ = central_ + other.central_;
        result.deviations_ = deviations_;
        
        for (const auto& pair : other.deviations_) {
            result.deviations_[pair.first] += pair.second;
        }
        
        result.next_noise_id_ = std::max(next_noise_id_, other.next_noise_id_);
        return result;
    }
    
    AffineForm operator-(const AffineForm& other) const {
        AffineForm result;
        result.central_ = central_ - other.central_;
        result.deviations_ = deviations_;
        
        for (const auto& pair : other.deviations_) {
            result.deviations_[pair.first] -= pair.second;
        }
        
        result.next_noise_id_ = std::max(next_noise_id_, other.next_noise_id_);
        return result;
    }
    
    AffineForm operator*(const AffineForm& other) const {
        // x * y = (x₀ + Σxᵢεᵢ)(y₀ + Σyⱼεⱼ)
        //       = x₀y₀ + x₀Σyⱼεⱼ + y₀Σxᵢεᵢ + (Σxᵢεᵢ)(Σyⱼεⱼ)
        // The last term is quadratic in εs, approximate as new noise term
        
        AffineForm result;
        result.central_ = central_ * other.central_;
        
        // Linear terms
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] += pair.second * other.central_;
        }
        for (const auto& pair : other.deviations_) {
            result.deviations_[pair.first] += pair.second * central_;
        }
        
        // Quadratic error term (conservative bound)
        Scalar quadratic_error = radius() * other.radius();
        if (quadratic_error > 0) {
            int new_noise = result.allocate_noise_symbol();
            result.deviations_[new_noise] = quadratic_error;
        }
        
        result.next_noise_id_ = std::max(next_noise_id_, other.next_noise_id_) + 1;
        return result;
    }
    
    AffineForm operator/(const AffineForm& other) const {
        // x / y ≈ x / y₀ - x(y - y₀) / y₀²
        // This is a first-order approximation
        
        if (other.to_interval().contains(0.0)) {
            throw std::runtime_error("Division by affine form containing zero");
        }
        
        Scalar y0 = other.central_;
        Scalar y_rad = other.radius();
        
        // Inverse of y₀ ± yᵣₐ𝒹 is approximately 1/y₀ ∓ yᵣₐ𝒹/y₀²
        AffineForm result;
        result.central_ = central_ / y0;
        
        // Scale existing deviations
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] = pair.second / y0;
        }
        
        // Error from inverse approximation
        Scalar inv_error = y_rad / (y0 * y0);
        Scalar total_error = std::abs(central_) * inv_error + radius() * inv_error;
        
        if (total_error > 0) {
            int new_noise = result.allocate_noise_symbol();
            result.deviations_[new_noise] = total_error;
        }
        
        result.next_noise_id_ = std::max(next_noise_id_, other.next_noise_id_) + 1;
        return result;
    }
    
    AffineForm operator-() const {
        AffineForm result;
        result.central_ = -central_;
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] = -pair.second;
        }
        result.next_noise_id_ = next_noise_id_;
        return result;
    }
    
    // Non-linear functions - use Chebyshev approximations
    AffineForm exp() const {
        // exp(x) ≈ exp(x₀)(1 + (x - x₀) + (x - x₀)²/2 + ...)
        // Use first-order and bound the error
        
        Interval interval = to_interval();
        Scalar x_mid = central_;
        Scalar x_rad = radius();
        
        // First-order approximation: exp(x₀) * (1 + (x - x₀))
        AffineForm result;
        Scalar exp_mid = std::exp(x_mid);
        result.central_ = exp_mid;
        
        // Propagate deviations scaled by exp(x₀)
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] = pair.second * exp_mid;
        }
        
        // Error bound from higher-order terms
        // |exp(x) - exp(x₀)(1 + (x-x₀))| ≤ exp(x_max) * (x-x₀)²/2
        Scalar max_error = std::exp(interval.upper()) * x_rad * x_rad / 2.0;
        
        int new_noise = result.allocate_noise_symbol();
        result.deviations_[new_noise] = max_error;
        
        result.next_noise_id_ = next_noise_id_ + 1;
        return result;
    }
    
    AffineForm log() const {
        // log(x) ≈ log(x₀) + (x - x₀)/x₀ - (x - x₀)²/(2x₀²)
        
        Interval interval = to_interval();
        if (interval.lower() <= 0) {
            throw std::runtime_error("Log of non-positive affine form");
        }
        
        Scalar x_mid = central_;
        Scalar x_rad = radius();
        
        // First-order approximation
        AffineForm result;
        result.central_ = std::log(x_mid);
        
        // Propagate deviations scaled by 1/x₀
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] = pair.second / x_mid;
        }
        
        // Error from second-order term
        Scalar max_error = x_rad * x_rad / (2.0 * interval.lower() * interval.lower());
        
        int new_noise = result.allocate_noise_symbol();
        result.deviations_[new_noise] = max_error;
        
        result.next_noise_id_ = next_noise_id_ + 1;
        return result;
    }
    
    AffineForm sqrt() const {
        // sqrt(x) ≈ sqrt(x₀) + (x - x₀)/(2*sqrt(x₀))
        
        Interval interval = to_interval();
        if (interval.lower() < 0) {
            throw std::runtime_error("Sqrt of negative affine form");
        }
        
        Scalar x_mid = central_;
        Scalar x_rad = radius();
        
        if (x_mid <= 0) {
            // Degenerate case
            return AffineForm(interval.sqrt());
        }
        
        AffineForm result;
        Scalar sqrt_mid = std::sqrt(x_mid);
        result.central_ = sqrt_mid;
        
        // First derivative: 1/(2*sqrt(x))
        Scalar derivative = 1.0 / (2.0 * sqrt_mid);
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] = pair.second * derivative;
        }
        
        // Error from second derivative
        // |sqrt''(x)| = 1/(4*x^(3/2)) ≤ 1/(4*x_min^(3/2))
        Scalar second_deriv_bound = 1.0 / (4.0 * std::pow(interval.lower(), 1.5));
        Scalar max_error = second_deriv_bound * x_rad * x_rad / 2.0;
        
        int new_noise = result.allocate_noise_symbol();
        result.deviations_[new_noise] = max_error;
        
        result.next_noise_id_ = next_noise_id_ + 1;
        return result;
    }
    
    AffineForm sin() const {
        // sin(x) ≈ sin(x₀) + cos(x₀)(x - x₀)
        
        Interval interval = to_interval();
        Scalar x_mid = central_;
        Scalar x_rad = radius();
        
        AffineForm result;
        result.central_ = std::sin(x_mid);
        
        // First derivative: cos(x₀)
        Scalar cos_mid = std::cos(x_mid);
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] = pair.second * cos_mid;
        }
        
        // Error bound: |sin''(x)| = |sin(x)| ≤ 1
        Scalar max_error = x_rad * x_rad / 2.0;
        
        int new_noise = result.allocate_noise_symbol();
        result.deviations_[new_noise] = max_error;
        
        result.next_noise_id_ = next_noise_id_ + 1;
        return result;
    }
    
    AffineForm cos() const {
        // cos(x) ≈ cos(x₀) - sin(x₀)(x - x₀)
        
        Interval interval = to_interval();
        Scalar x_mid = central_;
        Scalar x_rad = radius();
        
        AffineForm result;
        result.central_ = std::cos(x_mid);
        
        // First derivative: -sin(x₀)
        Scalar sin_mid = -std::sin(x_mid);
        for (const auto& pair : deviations_) {
            result.deviations_[pair.first] = pair.second * sin_mid;
        }
        
        // Error bound
        Scalar max_error = x_rad * x_rad / 2.0;
        
        int new_noise = result.allocate_noise_symbol();
        result.deviations_[new_noise] = max_error;
        
        result.next_noise_id_ = next_noise_id_ + 1;
        return result;
    }
    
    // Measure of precision gain vs interval arithmetic
    Scalar precision_improvement_factor() const {
        Interval interval_version = to_interval();
        Scalar interval_width = interval_version.width();
        Scalar affine_width = 2.0 * radius();
        
        if (affine_width > 0) {
            return interval_width / affine_width;
        }
        return 1.0;
    }
    
    // Number of noise symbols
    size_t num_noise_symbols() const {
        return deviations_.size();
    }
    
private:
    Scalar central_;  // Central value x₀
    std::map<int, Scalar> deviations_;  // Noise coefficients xᵢ
    mutable int next_noise_id_;  // For allocating new noise symbols
    
    int allocate_noise_symbol() const {
        return next_noise_id_++;
    }
};

// Vector of affine forms for multi-dimensional operations
class AffineVector {
public:
    using Scalar = double;
    
    AffineVector() = default;
    
    explicit AffineVector(size_t dim) : forms_(dim) {}
    
    explicit AffineVector(const std::vector<Interval>& intervals) {
        forms_.reserve(intervals.size());
        for (const auto& interval : intervals) {
            forms_.emplace_back(interval);
        }
    }
    
    size_t size() const { return forms_.size(); }
    
    AffineForm& operator[](size_t i) { return forms_[i]; }
    const AffineForm& operator[](size_t i) const { return forms_[i]; }
    
    // Convert to interval vector
    std::vector<Interval> to_intervals() const {
        std::vector<Interval> result;
        result.reserve(forms_.size());
        for (const auto& form : forms_) {
            result.push_back(form.to_interval());
        }
        return result;
    }
    
    // Measure average precision improvement
    Scalar average_precision_improvement() const {
        if (forms_.empty()) return 1.0;
        
        Scalar sum = 0.0;
        for (const auto& form : forms_) {
            sum += form.precision_improvement_factor();
        }
        return sum / forms_.size();
    }
    
private:
    std::vector<AffineForm> forms_;
};

} // namespace certified
} // namespace hnf
