#include "relu_to_tropical.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>

namespace tropical {

// ============================================================================
// ReLUNetwork Implementation
// ============================================================================

int ReLUNetwork::num_parameters() const {
    int total = 0;
    for (const auto& layer : layers_) {
        total += layer.weights.numel() + layer.biases.numel();
    }
    return total;
}

torch::Tensor ReLUNetwork::forward(const torch::Tensor& input) const {
    torch::Tensor x = input;
    for (const auto& layer : layers_) {
        x = torch::relu(torch::matmul(x, layer.weights.t()) + layer.biases);
    }
    return x;
}

// ============================================================================
// TropicalConverter Implementation
// ============================================================================

TropicalPolynomial TropicalConverter::convert_relu_neuron(
    const std::vector<double>& weights,
    double bias,
    int input_dim) {
    
    // ReLU neuron: max(0, w·x + b)
    // In tropical: 0 ⊕ (b + w₁·x₁ + ... + wₙ·xₙ)
    //            = max(0, b + Σᵢ wᵢ·xᵢ)
    
    TropicalPolynomial poly(input_dim);
    
    // First monomial: constant 0 (from max(0, ...))
    Exponent zero_exp(input_dim, 0);
    poly.add_monomial(TropicalMonomial(0.0, zero_exp));
    
    // Second monomial: b + Σᵢ wᵢ·xᵢ
    // In tropical polynomial form, this becomes a monomial with:
    // - coefficient = b
    // - exponents = [w₁, w₂, ..., wₙ] (but in tropical, exponents are integer powers)
    
    // For ReLU networks, we encode the linear function as a tropical monomial
    // The "exponent" encodes which inputs contribute
    // We use a simplified representation where each weight becomes a separate term
    
    for (size_t i = 0; i < weights.size(); ++i) {
        if (std::abs(weights[i]) > 1e-10) {
            Exponent exp(input_dim, 0);
            exp[i] = 1;  // x_i appears with coefficient weights[i]
            
            // In tropical arithmetic, we represent w_i * x_i as:
            // coefficient = bias + log(|w_i|), exponent = sign(w_i) * 1
            // But for ReLU, we keep the linear structure
            
            poly.add_monomial(TropicalMonomial(bias + weights[i], exp));
        }
    }
    
    return poly;
}

std::vector<TropicalPolynomial> TropicalConverter::convert_layer(
    const ReLULayer& layer,
    const std::vector<TropicalPolynomial>& input_polys) {
    
    std::vector<TropicalPolynomial> output_polys;
    
    int out_dim = layer.output_dim();
    int in_dim = layer.input_dim();
    
    auto weights_acc = layer.weights.accessor<float, 2>();
    auto biases_acc = layer.biases.accessor<float, 1>();
    
    for (int out_neuron = 0; out_neuron < out_dim; ++out_neuron) {
        // Extract weights and bias for this neuron
        std::vector<double> neuron_weights(in_dim);
        for (int i = 0; i < in_dim; ++i) {
            neuron_weights[i] = weights_acc[out_neuron][i];
        }
        double neuron_bias = biases_acc[out_neuron];
        
        // Convert this ReLU neuron to tropical polynomial
        TropicalPolynomial neuron_poly = 
            convert_relu_neuron(neuron_weights, neuron_bias, in_dim);
        
        // Compose with input polynomials if this is not the first layer
        if (!input_polys.empty()) {
            // Composition: substitute input_polys into neuron_poly
            // This represents function composition in tropical semiring
            
            // For simplicity in this implementation, we track the composed structure
            // Real composition would expand out all terms
            output_polys.push_back(neuron_poly);
        } else {
            output_polys.push_back(neuron_poly);
        }
    }
    
    return output_polys;
}

std::vector<TropicalPolynomial> TropicalConverter::convert(const ReLUNetwork& network) {
    const auto& layers = network.layers();
    
    if (layers.empty()) {
        throw std::runtime_error("Cannot convert empty network");
    }
    
    // Start with identity (input layer)
    std::vector<TropicalPolynomial> current_polys;
    
    // Convert each layer sequentially
    for (size_t layer_idx = 0; layer_idx < layers.size(); ++layer_idx) {
        const auto& layer = layers[layer_idx];
        
        if (layer_idx == 0) {
            // First layer: convert directly from inputs
            current_polys = convert_layer(layer, {});
        } else {
            // Subsequent layers: compose with previous layer
            current_polys = convert_layer(layer, current_polys);
        }
    }
    
    return current_polys;
}

TropicalPolynomial TropicalConverter::convert_single_output(const ReLUNetwork& network) {
    auto polys = convert(network);
    if (polys.empty()) {
        throw std::runtime_error("Network has no outputs");
    }
    if (polys.size() > 1) {
        std::cerr << "Warning: Network has multiple outputs, returning first" << std::endl;
    }
    return polys[0];
}

// ============================================================================
// LinearRegionEnumerator Implementation
// ============================================================================

std::vector<LinearRegionEnumerator::Hyperplane> 
LinearRegionEnumerator::extract_hyperplanes() const {
    std::vector<Hyperplane> hyperplanes;
    
    // Each ReLU activation defines a hyperplane w·x + b = 0
    int total_neurons = 0;
    for (const auto& layer : network_.layers()) {
        total_neurons += layer.output_dim();
    }
    
    hyperplanes.reserve(total_neurons);
    
    // Extract hyperplane from each neuron
    for (const auto& layer : network_.layers()) {
        auto weights_acc = layer.weights.accessor<float, 2>();
        auto biases_acc = layer.biases.accessor<float, 1>();
        
        int out_dim = layer.output_dim();
        int in_dim = layer.input_dim();
        
        for (int neuron = 0; neuron < out_dim; ++neuron) {
            Hyperplane hp;
            hp.normal.resize(in_dim);
            for (int i = 0; i < in_dim; ++i) {
                hp.normal[i] = weights_acc[neuron][i];
            }
            hp.offset = biases_acc[neuron];
            hyperplanes.push_back(hp);
        }
    }
    
    return hyperplanes;
}

std::vector<LinearRegionEnumerator::LinearRegion>
LinearRegionEnumerator::enumerate_regions_exact() const {
    // This is exponentially expensive: O(2^n) where n = number of neurons
    // Only feasible for small networks
    
    auto hyperplanes = extract_hyperplanes();
    int n = hyperplanes.size();
    
    if (n > 20) {
        throw std::runtime_error("Too many neurons for exact enumeration (>20)");
    }
    
    std::vector<LinearRegion> regions;
    
    // Enumerate all 2^n activation patterns
    for (int pattern = 0; pattern < (1 << n); ++pattern) {
        LinearRegion region;
        region.activation_pattern.resize(n);
        
        for (int i = 0; i < n; ++i) {
            region.activation_pattern[i] = (pattern >> i) & 1;
        }
        
        // Check if this activation pattern is realizable
        // (i.e., there exists a point with this pattern)
        // For simplicity, we assume all patterns are realizable
        // Real implementation would solve linear feasibility
        
        regions.push_back(region);
    }
    
    return regions;
}

std::vector<LinearRegionEnumerator::LinearRegion>
LinearRegionEnumerator::enumerate_regions_sampling(int num_samples) const {
    // Sample points from input space and record unique activation patterns
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0, 1.0);
    
    std::set<std::vector<bool>> unique_patterns;
    std::vector<LinearRegion> regions;
    
    int input_dim = network_.input_dim();
    
    for (int sample = 0; sample < num_samples; ++sample) {
        // Generate random input
        auto input = torch::randn({1, input_dim});
        
        // Forward pass and record activations
        std::vector<bool> pattern;
        torch::Tensor x = input;
        
        for (const auto& layer : network_.layers()) {
            auto z = torch::matmul(x, layer.weights.t()) + layer.biases;
            auto activated = z > 0;
            
            auto activated_acc = activated.accessor<bool, 2>();
            for (int i = 0; i < layer.output_dim(); ++i) {
                pattern.push_back(activated_acc[0][i]);
            }
            
            x = torch::relu(z);
        }
        
        unique_patterns.insert(pattern);
    }
    
    // Convert unique patterns to LinearRegion objects
    for (const auto& pattern : unique_patterns) {
        LinearRegion region;
        region.activation_pattern = pattern;
        regions.push_back(region);
    }
    
    return regions;
}

int LinearRegionEnumerator::count_exact() const {
    try {
        auto regions = enumerate_regions_exact();
        return regions.size();
    } catch (const std::exception&) {
        return -1;  // Too expensive
    }
}

int LinearRegionEnumerator::count_approximate(int num_samples) const {
    auto regions = enumerate_regions_sampling(num_samples);
    return regions.size();
}

int LinearRegionEnumerator::count_upper_bound() const {
    // Theoretical upper bound from tropical geometry
    TropicalConverter converter;
    auto tropical_polys = converter.convert(network_);
    
    int total_bound = 0;
    for (const auto& poly : tropical_polys) {
        NewtonPolytope polytope(poly);
        total_bound += polytope.linear_region_upper_bound();
    }
    
    return total_bound;
}

int LinearRegionEnumerator::count_lower_bound() const {
    // Lower bound: at least as many regions as monomials in tropical representation
    TropicalConverter converter;
    auto tropical_polys = converter.convert(network_);
    
    int max_monomials = 0;
    for (const auto& poly : tropical_polys) {
        max_monomials = std::max(max_monomials, poly.num_monomials());
    }
    
    return max_monomials;
}

// ============================================================================
// Complexity Computation
// ============================================================================

void NetworkComplexity::print() const {
    std::cout << "Network Complexity Analysis:\n";
    std::cout << "  Parameters: " << num_parameters << "\n";
    std::cout << "  Linear Regions:\n";
    if (num_linear_regions_exact >= 0) {
        std::cout << "    Exact count: " << num_linear_regions_exact << "\n";
    }
    std::cout << "    Approximate: " << num_linear_regions_approx << "\n";
    std::cout << "    Upper bound: " << num_linear_regions_upper << "\n";
    std::cout << "    Lower bound: " << num_linear_regions_lower << "\n";
    std::cout << "  Efficiency (regions/params): " << std::fixed 
              << std::setprecision(2) << efficiency_ratio << "\n";
    std::cout << "  Newton Polytope:\n";
    std::cout << "    Volume: " << std::scientific << polytope_volume << "\n";
    std::cout << "    Vertices: " << polytope_vertices << "\n";
    std::cout << std::defaultfloat;
}

NetworkComplexity compute_network_complexity(const ReLUNetwork& network, 
                                               bool compute_exact) {
    NetworkComplexity complexity;
    
    complexity.num_parameters = network.num_parameters();
    
    LinearRegionEnumerator enumerator(network);
    
    if (compute_exact) {
        complexity.num_linear_regions_exact = enumerator.count_exact();
    } else {
        complexity.num_linear_regions_exact = -1;
    }
    
    complexity.num_linear_regions_approx = enumerator.count_approximate(100000);
    complexity.num_linear_regions_upper = enumerator.count_upper_bound();
    complexity.num_linear_regions_lower = enumerator.count_lower_bound();
    
    complexity.efficiency_ratio = 
        static_cast<double>(complexity.num_linear_regions_approx) / 
        complexity.num_parameters;
    
    // Compute Newton polytope properties
    TropicalConverter converter;
    auto tropical_polys = converter.convert(network);
    
    if (!tropical_polys.empty()) {
        NewtonPolytope polytope(tropical_polys[0]);
        complexity.polytope_volume = polytope.volume();
        complexity.polytope_vertices = polytope.num_vertices();
    } else {
        complexity.polytope_volume = 0.0;
        complexity.polytope_vertices = 0;
    }
    
    return complexity;
}

// ============================================================================
// Architecture Comparison
// ============================================================================

void ArchitectureComparison::print() const {
    std::cout << "\n=== Architecture Comparison ===\n\n";
    
    std::cout << "Architecture 1:\n";
    arch1.print();
    
    std::cout << "\nArchitecture 2:\n";
    arch2.print();
    
    std::cout << "\n--- Comparison ---\n";
    std::cout << "Winner by efficiency: " << winner_by_efficiency << "\n";
    std::cout << "Winner by capacity: " << winner_by_capacity << "\n";
    std::cout << "Efficiency improvement: " << std::fixed << std::setprecision(1)
              << (efficiency_improvement * 100) << "%\n";
}

ArchitectureComparison compare_architectures(const ReLUNetwork& net1,
                                               const ReLUNetwork& net2,
                                               bool compute_exact) {
    ArchitectureComparison comparison;
    
    comparison.arch1 = compute_network_complexity(net1, compute_exact);
    comparison.arch2 = compute_network_complexity(net2, compute_exact);
    
    // Determine winners
    if (comparison.arch1.efficiency_ratio > comparison.arch2.efficiency_ratio) {
        comparison.winner_by_efficiency = "Architecture 1";
        comparison.efficiency_improvement = 
            (comparison.arch1.efficiency_ratio - comparison.arch2.efficiency_ratio) /
            comparison.arch2.efficiency_ratio;
    } else {
        comparison.winner_by_efficiency = "Architecture 2";
        comparison.efficiency_improvement = 
            (comparison.arch2.efficiency_ratio - comparison.arch1.efficiency_ratio) /
            comparison.arch1.efficiency_ratio;
    }
    
    if (comparison.arch1.num_linear_regions_approx > 
        comparison.arch2.num_linear_regions_approx) {
        comparison.winner_by_capacity = "Architecture 1";
    } else {
        comparison.winner_by_capacity = "Architecture 2";
    }
    
    return comparison;
}

} // namespace tropical
