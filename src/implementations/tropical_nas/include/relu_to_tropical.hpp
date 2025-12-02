#ifndef RELU_TO_TROPICAL_HPP
#define RELU_TO_TROPICAL_HPP

#include "tropical_arithmetic.hpp"
#include <torch/torch.h>
#include <vector>
#include <memory>

namespace tropical {

// ReLU network layer representation
struct ReLULayer {
    torch::Tensor weights;  // [out_dim, in_dim]
    torch::Tensor biases;   // [out_dim]
    
    ReLULayer(const torch::Tensor& w, const torch::Tensor& b) 
        : weights(w), biases(b) {}
    
    int input_dim() const { return weights.size(1); }
    int output_dim() const { return weights.size(0); }
};

// Complete ReLU network
class ReLUNetwork {
private:
    std::vector<ReLULayer> layers_;
    int input_dim_;
    int output_dim_;
    
public:
    ReLUNetwork(int input_dim) : input_dim_(input_dim), output_dim_(input_dim) {}
    
    void add_layer(const torch::Tensor& weights, const torch::Tensor& biases) {
        layers_.emplace_back(weights, biases);
        output_dim_ = weights.size(0);
    }
    
    const std::vector<ReLULayer>& layers() const { return layers_; }
    int input_dim() const { return input_dim_; }
    int output_dim() const { return output_dim_; }
    int num_layers() const { return layers_.size(); }
    int num_parameters() const;
    
    // Forward pass for numerical evaluation
    torch::Tensor forward(const torch::Tensor& input) const;
};

// Convert ReLU network to tropical representation
class TropicalConverter {
private:
    // Recursively build tropical polynomial from ReLU compositions
    std::vector<TropicalPolynomial> convert_layer(
        const ReLULayer& layer,
        const std::vector<TropicalPolynomial>& input_polys);
    
    // Convert single ReLU neuron: max(0, w·x + b)
    // In tropical: max(0, c₁ + a₁·x₁ + ... + aₙ·xₙ)
    // = 0 ⊕ (c₁ ⊗ x^{a})
    TropicalPolynomial convert_relu_neuron(
        const std::vector<double>& weights,
        double bias,
        int input_dim);
    
public:
    // Convert full network to tropical polynomials (one per output neuron)
    std::vector<TropicalPolynomial> convert(const ReLUNetwork& network);
    
    // Convert to single tropical polynomial (for single-output networks)
    TropicalPolynomial convert_single_output(const ReLUNetwork& network);
};

// Linear region enumeration for ReLU networks
class LinearRegionEnumerator {
private:
    ReLUNetwork network_;
    
    // Hyperplane arrangement from ReLU activations
    struct Hyperplane {
        std::vector<double> normal;  // coefficients
        double offset;
    };
    
    std::vector<Hyperplane> extract_hyperplanes() const;
    
    // Cell decomposition of hyperplane arrangement
    struct LinearRegion {
        std::vector<bool> activation_pattern;  // which ReLUs are active
        std::vector<double> representative_point;
    };
    
    std::vector<LinearRegion> enumerate_regions_exact() const;
    
    // Sampling-based approximation for large networks
    std::vector<LinearRegion> enumerate_regions_sampling(int num_samples) const;
    
public:
    LinearRegionEnumerator(const ReLUNetwork& net) : network_(net) {}
    
    // Exact count (exponential time, only for small networks)
    int count_exact() const;
    
    // Approximate count via random sampling
    int count_approximate(int num_samples = 100000) const;
    
    // Upper bound from tropical polytope
    int count_upper_bound() const;
    
    // Lower bound from activation pattern diversity
    int count_lower_bound() const;
};

// Complexity measure for comparing architectures
struct NetworkComplexity {
    int num_parameters;
    int num_linear_regions_exact;      // -1 if too expensive to compute
    int num_linear_regions_approx;
    int num_linear_regions_upper;
    int num_linear_regions_lower;
    double efficiency_ratio;  // regions / parameters
    
    // Newton polytope volume (geometric complexity)
    double polytope_volume;
    int polytope_vertices;
    
    void print() const;
};

// Compute all complexity measures for a network
NetworkComplexity compute_network_complexity(const ReLUNetwork& network, 
                                               bool compute_exact = false);

// Compare two architectures by tropical complexity
struct ArchitectureComparison {
    NetworkComplexity arch1;
    NetworkComplexity arch2;
    
    std::string winner_by_efficiency;
    std::string winner_by_capacity;
    double efficiency_improvement;
    
    void print() const;
};

ArchitectureComparison compare_architectures(const ReLUNetwork& net1, 
                                               const ReLUNetwork& net2,
                                               bool compute_exact = false);

} // namespace tropical

#endif // RELU_TO_TROPICAL_HPP
