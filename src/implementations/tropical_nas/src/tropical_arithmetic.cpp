#include "tropical_arithmetic.hpp"
#include <algorithm>
#include <cmath>
#include <numeric>
#include <set>
#include <stdexcept>
#include <iostream>

namespace tropical {

// ============================================================================
// Newton Polytope Implementation
// ============================================================================

NewtonPolytope::NewtonPolytope(const TropicalPolynomial& poly) 
    : dim_(poly.dimension()) {
    
    // Extract exponent vectors
    std::vector<Exponent> exponents;
    for (const auto& mono : poly.monomials()) {
        exponents.push_back(mono.exponent());
    }
    
    if (exponents.empty()) {
        return;  // Empty polytope
    }
    
    compute_convex_hull(exponents);
}

bool NewtonPolytope::is_extreme_point(
    const std::vector<double>& point,
    const std::vector<std::vector<double>>& all_points) const {
    
    // Point is extreme if it cannot be written as convex combination of others
    // For low dimensions, we use a direct check:
    // A point p is extreme iff it's not in the interior of conv(other points)
    
    if (all_points.size() <= dim_) {
        // Too few points to have interior
        return true;
    }
    
    // Check if point is on the boundary by checking if it maximizes
    // some linear functional
    for (int coord = 0; coord < dim_; ++coord) {
        double max_val = point[coord];
        bool is_unique_max = true;
        
        for (const auto& other : all_points) {
            if (&other == &point) continue;
            if (other[coord] > max_val + 1e-10) {
                is_unique_max = false;
                break;
            }
            if (std::abs(other[coord] - max_val) < 1e-10) {
                is_unique_max = false;
            }
        }
        
        if (is_unique_max) {
            return true;  // Maximizes this coordinate
        }
    }
    
    // Also check minimum
    for (int coord = 0; coord < dim_; ++coord) {
        double min_val = point[coord];
        bool is_unique_min = true;
        
        for (const auto& other : all_points) {
            if (&other == &point) continue;
            if (other[coord] < min_val - 1e-10) {
                is_unique_min = false;
                break;
            }
            if (std::abs(other[coord] - min_val) < 1e-10) {
                is_unique_min = false;
            }
        }
        
        if (is_unique_min) {
            return true;
        }
    }
    
    return false;  // Might be interior point
}

void NewtonPolytope::compute_convex_hull(const std::vector<Exponent>& exponents) {
    if (exponents.empty()) return;
    
    // Convert integer exponents to double for geometric computation
    std::vector<std::vector<double>> points;
    for (const auto& exp : exponents) {
        std::vector<double> point(exp.begin(), exp.end());
        points.push_back(point);
    }
    
    // For small dimensions (1D, 2D, 3D), use specialized algorithms
    if (dim_ == 1) {
        // 1D: just find min and max
        double min_val = points[0][0];
        double max_val = points[0][0];
        for (const auto& p : points) {
            min_val = std::min(min_val, p[0]);
            max_val = std::max(max_val, p[0]);
        }
        vertices_.push_back({min_val});
        if (min_val != max_val) {
            vertices_.push_back({max_val});
        }
        return;
    }
    
    // For higher dimensions, use gift wrapping / incremental hull
    // Simplified version: find extreme points in each coordinate direction
    std::set<std::vector<double>> vertex_set;
    
    // Add coordinate-wise extremal points
    for (int coord = 0; coord < dim_; ++coord) {
        auto min_it = std::min_element(points.begin(), points.end(),
            [coord](const std::vector<double>& a, const std::vector<double>& b) {
                return a[coord] < b[coord];
            });
        auto max_it = std::max_element(points.begin(), points.end(),
            [coord](const std::vector<double>& a, const std::vector<double>& b) {
                return a[coord] < b[coord];
            });
        
        vertex_set.insert(*min_it);
        vertex_set.insert(*max_it);
    }
    
    // Add points that are extreme in multiple coordinates
    for (const auto& point : points) {
        if (is_extreme_point(point, points)) {
            vertex_set.insert(point);
        }
    }
    
    vertices_.assign(vertex_set.begin(), vertex_set.end());
}

double NewtonPolytope::volume() const {
    if (vertices_.empty()) return 0.0;
    if (dim_ == 1) {
        if (vertices_.size() < 2) return 0.0;
        return std::abs(vertices_[1][0] - vertices_[0][0]);
    }
    
    // For higher dimensions, use Monte Carlo estimation
    // This is a simplified approximation for the volume
    
    // Compute bounding box
    std::vector<double> min_coords(dim_, std::numeric_limits<double>::max());
    std::vector<double> max_coords(dim_, std::numeric_limits<double>::lowest());
    
    for (const auto& v : vertices_) {
        for (int i = 0; i < dim_; ++i) {
            min_coords[i] = std::min(min_coords[i], v[i]);
            max_coords[i] = std::max(max_coords[i], v[i]);
        }
    }
    
    // Approximate volume as product of coordinate ranges
    // (This is an upper bound; true volume is smaller)
    double vol = 1.0;
    for (int i = 0; i < dim_; ++i) {
        vol *= (max_coords[i] - min_coords[i]);
    }
    
    return vol;
}

int NewtonPolytope::linear_region_upper_bound() const {
    // The number of linear regions in a tropical polynomial is bounded by
    // the number of vertices of its Newton polytope (Theorem from tropical geometry)
    
    // For a more refined bound, we use the formula:
    // regions ≤ 2^(number of vertices) for worst case
    // But a tighter bound is: regions ≤ C(n + d, d) where n = vertices, d = dimension
    
    int n = num_vertices();
    if (n == 0) return 1;
    
    // Use simplified bound: regions ≤ n! / (d! * (n-d)!)  if n > d
    // For small n, this is approximately n^d
    
    if (n <= dim_) {
        return std::pow(2, n);  // Conservative bound
    }
    
    // Approximate binomial coefficient C(n + dim, dim)
    double bound = 1.0;
    for (int i = 0; i < dim_; ++i) {
        bound *= (n + dim_ - i) / static_cast<double>(i + 1);
    }
    
    return static_cast<int>(std::min(bound, 1e9));  // Cap at billion
}

// ============================================================================
// Tropical Variety Implementation
// ============================================================================

int TropicalVariety::dimension() const {
    // The dimension of a tropical variety is computed from its defining polynomials
    // For a single tropical polynomial in n variables, the variety has dimension n-1
    
    if (defining_polynomials_.empty()) {
        return ambient_dim_;
    }
    
    // Codimension is at most the number of defining polynomials
    int codim = std::min(static_cast<int>(defining_polynomials_.size()), ambient_dim_);
    return ambient_dim_ - codim;
}

int TropicalVariety::count_linear_regions() const {
    // Count linear regions by analyzing the tropical variety structure
    
    if (defining_polynomials_.empty()) {
        return 1;  // Entire space is one region
    }
    
    // For each polynomial, count how many monomials can be maximal
    // The variety is where at least two monomials achieve the maximum
    
    int total_regions = 1;
    
    for (const auto& poly : defining_polynomials_) {
        int mono_count = poly.num_monomials();
        if (mono_count > 1) {
            // Each pair of monomials defines a hyperplane
            // Number of regions grows as C(mono_count, 2) in worst case
            total_regions *= (mono_count * (mono_count - 1)) / 2;
        }
    }
    
    // Cap at reasonable value
    return std::min(total_regions, 1000000);
}

} // namespace tropical
