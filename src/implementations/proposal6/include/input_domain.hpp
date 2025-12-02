#pragma once

#include "interval.hpp"
#include <Eigen/Dense>
#include <string>
#include <optional>
#include <random>

namespace hnf {
namespace certified {

// Specification of valid inputs for certification
// Based on Proposal 6, Section 1: Input Domain Specification
class InputDomain {
public:
    enum class Distribution {
        UNIFORM,
        GAUSSIAN,
        BOUNDED_UNKNOWN
    };
    
    InputDomain(const Eigen::VectorXd& lower_bounds,
                const Eigen::VectorXd& upper_bounds,
                Distribution dist = Distribution::BOUNDED_UNKNOWN)
        : lower_bounds_(lower_bounds),
          upper_bounds_(upper_bounds),
          distribution_(dist),
          dimension_(lower_bounds.size()) {
        
        if (lower_bounds.size() != upper_bounds.size()) {
            throw std::invalid_argument("Dimension mismatch in InputDomain");
        }
        
        for (int i = 0; i < dimension_; ++i) {
            if (lower_bounds(i) > upper_bounds(i)) {
                throw std::invalid_argument("Invalid bounds: lower > upper");
            }
        }
    }
    
    // Construct domain from data statistics
    static InputDomain from_dataset(
        const std::vector<Eigen::VectorXd>& data,
        double percentile = 99.9) {
        
        if (data.empty()) {
            throw std::invalid_argument("Empty dataset");
        }
        
        int dim = data[0].size();
        Eigen::MatrixXd data_matrix(data.size(), dim);
        
        for (size_t i = 0; i < data.size(); ++i) {
            if (data[i].size() != dim) {
                throw std::invalid_argument("Inconsistent dimensions in dataset");
            }
            data_matrix.row(i) = data[i];
        }
        
        Eigen::VectorXd lower(dim);
        Eigen::VectorXd upper(dim);
        
        for (int j = 0; j < dim; ++j) {
            std::vector<double> column(data.size());
            for (size_t i = 0; i < data.size(); ++i) {
                column[i] = data_matrix(i, j);
            }
            std::sort(column.begin(), column.end());
            
            int lower_idx = static_cast<int>((100.0 - percentile) / 200.0 * data.size());
            int upper_idx = static_cast<int>((100.0 + percentile) / 200.0 * data.size());
            upper_idx = std::min(upper_idx, static_cast<int>(data.size()) - 1);
            
            lower(j) = column[lower_idx];
            upper(j) = column[upper_idx];
        }
        
        return InputDomain(lower, upper);
    }
    
    // Getters
    const Eigen::VectorXd& lower_bounds() const { return lower_bounds_; }
    const Eigen::VectorXd& upper_bounds() const { return upper_bounds_; }
    int dimension() const { return dimension_; }
    Distribution distribution() const { return distribution_; }
    
    // Set distribution parameters (for Gaussian)
    void set_gaussian_params(const Eigen::VectorXd& mean, const Eigen::VectorXd& std) {
        if (mean.size() != dimension_ || std.size() != dimension_) {
            throw std::invalid_argument("Parameter size mismatch");
        }
        mean_ = mean;
        std_ = std;
        distribution_ = Distribution::GAUSSIAN;
    }
    
    std::optional<Eigen::VectorXd> mean() const { return mean_; }
    std::optional<Eigen::VectorXd> std() const { return std_; }
    
    // Compute diameter of domain (Euclidean)
    double diameter() const {
        return (upper_bounds_ - lower_bounds_).norm();
    }
    
    // Maximum component-wise width
    double max_width() const {
        return (upper_bounds_ - lower_bounds_).maxCoeff();
    }
    
    // Volume of domain (product of widths)
    double volume() const {
        double vol = 1.0;
        for (int i = 0; i < dimension_; ++i) {
            vol *= (upper_bounds_(i) - lower_bounds_(i));
        }
        return vol;
    }
    
    // Sample points from domain
    std::vector<Eigen::VectorXd> sample(int n, unsigned int seed = 42) const {
        std::mt19937 gen(seed);
        std::vector<Eigen::VectorXd> samples;
        samples.reserve(n);
        
        if (distribution_ == Distribution::GAUSSIAN && mean_ && std_) {
            // Sample from Gaussian, clipped to bounds
            for (int i = 0; i < n; ++i) {
                Eigen::VectorXd point(dimension_);
                for (int j = 0; j < dimension_; ++j) {
                    std::normal_distribution<double> dist((*mean_)(j), (*std_)(j));
                    double val = dist(gen);
                    val = std::max(lower_bounds_(j), std::min(upper_bounds_(j), val));
                    point(j) = val;
                }
                samples.push_back(point);
            }
        } else {
            // Uniform sampling in bounding box
            for (int i = 0; i < n; ++i) {
                Eigen::VectorXd point(dimension_);
                for (int j = 0; j < dimension_; ++j) {
                    std::uniform_real_distribution<double> dist(
                        lower_bounds_(j), upper_bounds_(j));
                    point(j) = dist(gen);
                }
                samples.push_back(point);
            }
        }
        
        return samples;
    }
    
    // Alias for clarity
    std::vector<Eigen::VectorXd> sample_uniform(int n, unsigned int seed = 42) const {
        return sample(n, seed);
    }
    
    // Sample on boundary
    std::vector<Eigen::VectorXd> sample_boundary(int n, unsigned int seed = 42) const {
        std::mt19937 gen(seed);
        std::vector<Eigen::VectorXd> samples;
        samples.reserve(n);
        
        std::uniform_int_distribution<int> dim_dist(0, dimension_ - 1);
        std::uniform_int_distribution<int> side_dist(0, 1);
        
        for (int i = 0; i < n; ++i) {
            Eigen::VectorXd point(dimension_);
            
            // Sample uniformly in interior
            for (int j = 0; j < dimension_; ++j) {
                std::uniform_real_distribution<double> dist(
                    lower_bounds_(j), upper_bounds_(j));
                point(j) = dist(gen);
            }
            
            // Pick a random dimension and fix it to boundary
            int fix_dim = dim_dist(gen);
            point(fix_dim) = side_dist(gen) == 0 ? lower_bounds_(fix_dim) : upper_bounds_(fix_dim);
            
            samples.push_back(point);
        }
        
        return samples;
    }
    
    // Convert to interval representation
    IntervalVector to_interval_vector() const {
        return IntervalVector(lower_bounds_, upper_bounds_);
    }
    
    // Check if point is in domain
    bool contains(const Eigen::VectorXd& point) const {
        if (point.size() != dimension_) {
            return false;
        }
        
        for (int i = 0; i < dimension_; ++i) {
            if (point(i) < lower_bounds_(i) || point(i) > upper_bounds_(i)) {
                return false;
            }
        }
        
        return true;
    }
    
    // Subdivide domain into smaller sub-domains
    std::vector<InputDomain> subdivide(int subdivisions_per_dim = 2) const {
        std::vector<InputDomain> subdomains;
        
        // Compute subdivision widths
        Eigen::VectorXd widths = (upper_bounds_ - lower_bounds_) / subdivisions_per_dim;
        
        // Generate all combinations
        std::function<void(int, Eigen::VectorXd&, Eigen::VectorXd&)> generate;
        generate = [&](int dim_idx, Eigen::VectorXd& lower, Eigen::VectorXd& upper) {
            if (dim_idx == dimension_) {
                subdomains.emplace_back(lower, upper, distribution_);
                return;
            }
            
            for (int i = 0; i < subdivisions_per_dim; ++i) {
                lower(dim_idx) = lower_bounds_(dim_idx) + i * widths(dim_idx);
                upper(dim_idx) = lower_bounds_(dim_idx) + (i + 1) * widths(dim_idx);
                generate(dim_idx + 1, lower, upper);
            }
        };
        
        Eigen::VectorXd lower(dimension_);
        Eigen::VectorXd upper(dimension_);
        generate(0, lower, upper);
        
        return subdomains;
    }
    
private:
    Eigen::VectorXd lower_bounds_;
    Eigen::VectorXd upper_bounds_;
    Distribution distribution_;
    int dimension_;
    
    std::optional<Eigen::VectorXd> mean_;
    std::optional<Eigen::VectorXd> std_;
};

} // namespace certified
} // namespace hnf
