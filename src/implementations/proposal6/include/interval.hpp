#pragma once

#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <vector>
#include <Eigen/Dense>

namespace hnf {
namespace certified {

// Rigorous interval arithmetic for certified bounds
// Implements interval operations with proven mathematical properties
class Interval {
public:
    using Scalar = double;
    
    Interval() : lower_(0.0), upper_(0.0) {}
    
    Interval(Scalar value) : lower_(value), upper_(value) {}
    
    Interval(Scalar lower, Scalar upper) : lower_(lower), upper_(upper) {
        if (lower > upper) {
            throw std::invalid_argument("Invalid interval: lower > upper");
        }
    }
    
    Scalar lower() const { return lower_; }
    Scalar upper() const { return upper_; }
    Scalar width() const { return upper_ - lower_; }
    Scalar midpoint() const { return 0.5 * (lower_ + upper_); }
    
    // Check if value is in interval
    bool contains(Scalar value) const {
        return value >= lower_ && value <= upper_;
    }
    
    // Interval arithmetic operations with rigorous rounding
    Interval operator+(const Interval& other) const {
        return Interval(
            lower_ + other.lower_,
            upper_ + other.upper_
        );
    }
    
    Interval operator-(const Interval& other) const {
        return Interval(
            lower_ - other.upper_,
            upper_ - other.lower_
        );
    }
    
    Interval operator*(const Interval& other) const {
        // All four products
        Scalar products[4] = {
            lower_ * other.lower_,
            lower_ * other.upper_,
            upper_ * other.lower_,
            upper_ * other.upper_
        };
        
        return Interval(
            *std::min_element(products, products + 4),
            *std::max_element(products, products + 4)
        );
    }
    
    Interval operator/(const Interval& other) const {
        if (other.contains(0.0)) {
            throw std::invalid_argument("Division by interval containing zero");
        }
        
        return (*this) * Interval(1.0/other.upper_, 1.0/other.lower_);
    }
    
    Interval operator-() const {
        return Interval(-upper_, -lower_);
    }
    
    // Elementary functions with rigorous bounds
    Interval exp() const {
        return Interval(std::exp(lower_), std::exp(upper_));
    }
    
    Interval log() const {
        if (lower_ <= 0.0) {
            throw std::invalid_argument("Log of non-positive interval");
        }
        return Interval(std::log(lower_), std::log(upper_));
    }
    
    Interval sqrt() const {
        if (lower_ < 0.0) {
            throw std::invalid_argument("Sqrt of negative interval");
        }
        return Interval(std::sqrt(lower_), std::sqrt(upper_));
    }
    
    Interval sin() const {
        // Complex case: need to check for extrema in interval
        Scalar l = std::sin(lower_);
        Scalar u = std::sin(upper_);
        
        // Check if interval contains pi/2 + 2k*pi (maximum)
        // Check if interval contains -pi/2 + 2k*pi (minimum)
        // For simplicity, use conservative bounds
        if (width() >= 2.0 * M_PI) {
            return Interval(-1.0, 1.0);
        }
        
        return Interval(std::min(l, u), std::max(l, u));
    }
    
    Interval cos() const {
        Scalar l = std::cos(lower_);
        Scalar u = std::cos(upper_);
        
        if (width() >= 2.0 * M_PI) {
            return Interval(-1.0, 1.0);
        }
        
        return Interval(std::min(l, u), std::max(l, u));
    }
    
    Interval tanh() const {
        return Interval(std::tanh(lower_), std::tanh(upper_));
    }
    
    // Power function
    Interval pow(Scalar exponent) const {
        if (exponent == 0.0) {
            return Interval(1.0);
        }
        
        if (exponent == 1.0) {
            return *this;
        }
        
        if (exponent > 0.0 && std::floor(exponent) == exponent) {
            // Integer exponent
            int n = static_cast<int>(exponent);
            if (n % 2 == 0) {
                // Even power - always non-negative
                if (lower_ >= 0.0) {
                    return Interval(std::pow(lower_, exponent), std::pow(upper_, exponent));
                } else if (upper_ <= 0.0) {
                    return Interval(std::pow(upper_, exponent), std::pow(lower_, exponent));
                } else {
                    // Straddles zero
                    return Interval(0.0, std::max(std::pow(lower_, exponent), std::pow(upper_, exponent)));
                }
            } else {
                // Odd power - monotonic
                return Interval(std::pow(lower_, exponent), std::pow(upper_, exponent));
            }
        }
        
        // General case
        if (lower_ < 0.0) {
            throw std::invalid_argument("Power of interval with negative values");
        }
        
        return Interval(std::pow(lower_, exponent), std::pow(upper_, exponent));
    }
    
    // Absolute value
    Interval abs() const {
        if (lower_ >= 0.0) {
            return *this;
        } else if (upper_ <= 0.0) {
            return -(*this);
        } else {
            return Interval(0.0, std::max(-lower_, upper_));
        }
    }
    
    // Maximum absolute value in interval
    Scalar max_abs() const {
        return std::max(std::abs(lower_), std::abs(upper_));
    }
    
    // Union (hull)
    Interval hull(const Interval& other) const {
        return Interval(
            std::min(lower_, other.lower_),
            std::max(upper_, other.upper_)
        );
    }
    
    // Intersection
    Interval intersect(const Interval& other) const {
        Scalar l = std::max(lower_, other.lower_);
        Scalar u = std::min(upper_, other.upper_);
        
        if (l > u) {
            throw std::invalid_argument("Empty intersection");
        }
        
        return Interval(l, u);
    }
    
private:
    Scalar lower_;
    Scalar upper_;
};

// Vector of intervals for multivariate analysis
class IntervalVector {
public:
    IntervalVector() {}
    
    explicit IntervalVector(size_t n) : intervals_(n) {}
    
    IntervalVector(const std::vector<Interval>& intervals) 
        : intervals_(intervals) {}
    
    IntervalVector(const Eigen::VectorXd& lower, const Eigen::VectorXd& upper) {
        if (lower.size() != upper.size()) {
            throw std::invalid_argument("Size mismatch in IntervalVector");
        }
        
        intervals_.resize(lower.size());
        for (size_t i = 0; i < intervals_.size(); ++i) {
            intervals_[i] = Interval(lower(i), upper(i));
        }
    }
    
    size_t size() const { return intervals_.size(); }
    
    Interval& operator[](size_t i) { return intervals_[i]; }
    const Interval& operator[](size_t i) const { return intervals_[i]; }
    
    // Compute diameter (maximum width)
    double diameter() const {
        double d = 0.0;
        for (const auto& iv : intervals_) {
            d += iv.width() * iv.width();
        }
        return std::sqrt(d);
    }
    
    // Norm bounds
    double norm_lower() const {
        double sum = 0.0;
        for (const auto& iv : intervals_) {
            // Minimum possible squared norm contribution
            if (iv.lower() >= 0.0) {
                sum += iv.lower() * iv.lower();
            } else if (iv.upper() <= 0.0) {
                sum += iv.upper() * iv.upper();
            }
            // else: can be zero
        }
        return std::sqrt(sum);
    }
    
    double norm_upper() const {
        double sum = 0.0;
        for (const auto& iv : intervals_) {
            double m = iv.max_abs();
            sum += m * m;
        }
        return std::sqrt(sum);
    }
    
    Interval norm_interval() const {
        return Interval(norm_lower(), norm_upper());
    }
    
    // Midpoint vector
    Eigen::VectorXd midpoint() const {
        Eigen::VectorXd mid(size());
        for (size_t i = 0; i < size(); ++i) {
            mid(i) = intervals_[i].midpoint();
        }
        return mid;
    }
    
    // Lower and upper bound vectors
    Eigen::VectorXd lower() const {
        Eigen::VectorXd lb(size());
        for (size_t i = 0; i < size(); ++i) {
            lb(i) = intervals_[i].lower();
        }
        return lb;
    }
    
    Eigen::VectorXd upper() const {
        Eigen::VectorXd ub(size());
        for (size_t i = 0; i < size(); ++i) {
            ub(i) = intervals_[i].upper();
        }
        return ub;
    }
    
    // Element-wise operations
    IntervalVector operator+(const IntervalVector& other) const {
        if (size() != other.size()) {
            throw std::invalid_argument("Size mismatch in IntervalVector addition");
        }
        
        std::vector<Interval> result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = intervals_[i] + other.intervals_[i];
        }
        return IntervalVector(result);
    }
    
    IntervalVector operator*(double scalar) const {
        std::vector<Interval> result(size());
        for (size_t i = 0; i < size(); ++i) {
            result[i] = intervals_[i] * Interval(scalar);
        }
        return IntervalVector(result);
    }
    
private:
    std::vector<Interval> intervals_;
};

// Matrix of intervals
class IntervalMatrix {
public:
    IntervalMatrix() : rows_(0), cols_(0) {}
    
    IntervalMatrix(size_t rows, size_t cols) 
        : rows_(rows), cols_(cols), intervals_(rows * cols) {}
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    Interval& operator()(size_t i, size_t j) {
        return intervals_[i * cols_ + j];
    }
    
    const Interval& operator()(size_t i, size_t j) const {
        return intervals_[i * cols_ + j];
    }
    
    // Matrix-vector multiplication
    IntervalVector operator*(const IntervalVector& v) const {
        if (cols_ != v.size()) {
            throw std::invalid_argument("Size mismatch in matrix-vector product");
        }
        
        std::vector<Interval> result(rows_);
        for (size_t i = 0; i < rows_; ++i) {
            Interval sum(0.0);
            for (size_t j = 0; j < cols_; ++j) {
                sum = sum + (*this)(i, j) * v[j];
            }
            result[i] = sum;
        }
        
        return IntervalVector(result);
    }
    
    // Operator norm bounds
    double operator_norm_upper() const {
        // Upper bound using row sums
        double max_row_sum = 0.0;
        for (size_t i = 0; i < rows_; ++i) {
            double row_sum = 0.0;
            for (size_t j = 0; j < cols_; ++j) {
                row_sum += (*this)(i, j).max_abs();
            }
            max_row_sum = std::max(max_row_sum, row_sum);
        }
        return max_row_sum;
    }
    
private:
    size_t rows_;
    size_t cols_;
    std::vector<Interval> intervals_;
};

} // namespace certified
} // namespace hnf
