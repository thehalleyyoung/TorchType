#ifndef TROPICAL_ARITHMETIC_HPP
#define TROPICAL_ARITHMETIC_HPP

#include <vector>
#include <limits>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <set>
#include <map>
#include <iostream>

namespace tropical {

// Tropical number in max-plus semiring
// Represents max-plus algebra where ⊕ = max and ⊗ = +
class TropicalNumber {
private:
    double value_;
    static constexpr double NEG_INF = -std::numeric_limits<double>::infinity();

public:
    TropicalNumber() : value_(NEG_INF) {}
    explicit TropicalNumber(double v) : value_(v) {}
    
    double value() const { return value_; }
    bool is_zero() const { return std::isinf(value_) && value_ < 0; }
    
    // Tropical addition (maximum)
    TropicalNumber operator+(const TropicalNumber& other) const {
        return TropicalNumber(std::max(value_, other.value_));
    }
    
    // Tropical multiplication (addition)
    TropicalNumber operator*(const TropicalNumber& other) const {
        if (is_zero() || other.is_zero()) {
            return TropicalNumber(); // tropical zero
        }
        return TropicalNumber(value_ + other.value_);
    }
    
    TropicalNumber& operator+=(const TropicalNumber& other) {
        value_ = std::max(value_, other.value_);
        return *this;
    }
    
    TropicalNumber& operator*=(const TropicalNumber& other) {
        if (other.is_zero()) {
            value_ = NEG_INF;
        } else if (!is_zero()) {
            value_ += other.value_;
        }
        return *this;
    }
    
    bool operator==(const TropicalNumber& other) const {
        if (is_zero() && other.is_zero()) return true;
        if (is_zero() || other.is_zero()) return false;
        return std::abs(value_ - other.value_) < 1e-10;
    }
    
    bool operator<(const TropicalNumber& other) const {
        return value_ < other.value_;
    }
    
    static TropicalNumber zero() { return TropicalNumber(); }
    static TropicalNumber one() { return TropicalNumber(0.0); }
};

// Exponent vector for tropical monomials
using Exponent = std::vector<int>;

// Tropical monomial: coefficient ⊗ x₁^{a₁} ⊗ ... ⊗ xₙ^{aₙ}
// In max-plus: coefficient + a₁*x₁ + ... + aₙ*xₙ
class TropicalMonomial {
private:
    TropicalNumber coeff_;
    Exponent exp_;
    
public:
    TropicalMonomial(const TropicalNumber& c, const Exponent& e) 
        : coeff_(c), exp_(e) {}
    
    TropicalMonomial(double c, const Exponent& e) 
        : coeff_(TropicalNumber(c)), exp_(e) {}
    
    const TropicalNumber& coefficient() const { return coeff_; }
    const Exponent& exponent() const { return exp_; }
    int dimension() const { return exp_.size(); }
    
    // Evaluate at a point
    TropicalNumber evaluate(const std::vector<TropicalNumber>& point) const {
        if (point.size() != exp_.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        TropicalNumber result = coeff_;
        for (size_t i = 0; i < exp_.size(); ++i) {
            for (int j = 0; j < exp_[i]; ++j) {
                result *= point[i];
            }
        }
        return result;
    }
    
    // Multiply monomials (add exponents, multiply coefficients)
    TropicalMonomial operator*(const TropicalMonomial& other) const {
        if (exp_.size() != other.exp_.size()) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        Exponent new_exp(exp_.size());
        for (size_t i = 0; i < exp_.size(); ++i) {
            new_exp[i] = exp_[i] + other.exp_[i];
        }
        
        return TropicalMonomial(coeff_ * other.coeff_, new_exp);
    }
    
    bool operator<(const TropicalMonomial& other) const {
        return exp_ < other.exp_;
    }
};

// Tropical polynomial: ⊕ᵢ (cᵢ ⊗ x^{aᵢ}) = maxᵢ(cᵢ + ⟨aᵢ, x⟩)
class TropicalPolynomial {
private:
    std::vector<TropicalMonomial> monomials_;
    int dim_;
    
    void simplify() {
        // Remove tropical zero monomials
        monomials_.erase(
            std::remove_if(monomials_.begin(), monomials_.end(),
                [](const TropicalMonomial& m) { return m.coefficient().is_zero(); }),
            monomials_.end()
        );
        
        // Sort monomials for canonical form
        std::sort(monomials_.begin(), monomials_.end());
    }
    
public:
    TropicalPolynomial(int dimension) : dim_(dimension) {}
    
    TropicalPolynomial(const std::vector<TropicalMonomial>& monomials, int dimension)
        : monomials_(monomials), dim_(dimension) {
        simplify();
    }
    
    void add_monomial(const TropicalMonomial& m) {
        if (m.dimension() != dim_) {
            throw std::invalid_argument("Monomial dimension mismatch");
        }
        monomials_.push_back(m);
    }
    
    const std::vector<TropicalMonomial>& monomials() const { return monomials_; }
    int dimension() const { return dim_; }
    int num_monomials() const { return monomials_.size(); }
    
    // Evaluate polynomial at point (compute max over all monomials)
    TropicalNumber evaluate(const std::vector<TropicalNumber>& point) const {
        if (point.size() != static_cast<size_t>(dim_)) {
            throw std::invalid_argument("Point dimension mismatch");
        }
        
        TropicalNumber result = TropicalNumber::zero();
        for (const auto& m : monomials_) {
            result += m.evaluate(point);
        }
        return result;
    }
    
    // Tropical addition (combine monomials)
    TropicalPolynomial operator+(const TropicalPolynomial& other) const {
        if (dim_ != other.dim_) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        std::vector<TropicalMonomial> combined = monomials_;
        combined.insert(combined.end(), other.monomials_.begin(), other.monomials_.end());
        return TropicalPolynomial(combined, dim_);
    }
    
    // Tropical multiplication (distribute monomials)
    TropicalPolynomial operator*(const TropicalPolynomial& other) const {
        if (dim_ != other.dim_) {
            throw std::invalid_argument("Dimension mismatch");
        }
        
        std::vector<TropicalMonomial> result_monomials;
        for (const auto& m1 : monomials_) {
            for (const auto& m2 : other.monomials_) {
                result_monomials.push_back(m1 * m2);
            }
        }
        
        return TropicalPolynomial(result_monomials, dim_);
    }
};

// Newton polytope computation for tropical polynomials
class NewtonPolytope {
private:
    std::vector<std::vector<double>> vertices_;
    int dim_;
    
    // Compute convex hull of exponent vectors (Gift wrapping/QuickHull)
    void compute_convex_hull(const std::vector<Exponent>& exponents);
    
    // Check if point is extreme (vertex of convex hull)
    bool is_extreme_point(const std::vector<double>& point,
                          const std::vector<std::vector<double>>& all_points) const;
    
public:
    NewtonPolytope(const TropicalPolynomial& poly);
    
    const std::vector<std::vector<double>>& vertices() const { return vertices_; }
    int dimension() const { return dim_; }
    int num_vertices() const { return vertices_.size(); }
    
    // Volume computation (for estimating linear region count)
    double volume() const;
    
    // Upper bound on linear regions from polytope complexity
    int linear_region_upper_bound() const;
};

// Tropical variety (solution set of tropical polynomial equations)
class TropicalVariety {
private:
    std::vector<TropicalPolynomial> defining_polynomials_;
    int ambient_dim_;
    
public:
    TropicalVariety(const std::vector<TropicalPolynomial>& polys, int dim)
        : defining_polynomials_(polys), ambient_dim_(dim) {}
    
    // Compute linear regions (piecewise-linear structure)
    int count_linear_regions() const;
    
    // Dimension of the variety
    int dimension() const;
};

} // namespace tropical

#endif // TROPICAL_ARITHMETIC_HPP
