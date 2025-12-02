#include "tropical_arithmetic.hpp"
#include "relu_to_tropical.hpp"
#include "tropical_architecture_search.hpp"
#include <iostream>
#include <cassert>
#include <cmath>

using namespace tropical;

void test_tropical_arithmetic() {
    std::cout << "\n=== Testing Tropical Arithmetic ===\n";
    
    // Test tropical numbers
    TropicalNumber a(5.0);
    TropicalNumber b(3.0);
    TropicalNumber zero = TropicalNumber::zero();
    TropicalNumber one = TropicalNumber::one();
    
    // Tropical addition (max)
    TropicalNumber sum = a + b;
    assert(std::abs(sum.value() - 5.0) < 1e-10);
    std::cout << "✓ Tropical addition (max): " << sum.value() << "\n";
    
    // Tropical multiplication (addition)
    TropicalNumber prod = a * b;
    assert(std::abs(prod.value() - 8.0) < 1e-10);
    std::cout << "✓ Tropical multiplication (add): " << prod.value() << "\n";
    
    // Zero element
    TropicalNumber z1 = a + zero;
    assert(z1 == a);
    TropicalNumber z2 = a * zero;
    assert(z2.is_zero());
    std::cout << "✓ Tropical zero element works\n";
    
    // One element
    TropicalNumber o1 = a * one;
    assert(std::abs(o1.value() - a.value()) < 1e-10);
    std::cout << "✓ Tropical one element works\n";
}

void test_tropical_monomials() {
    std::cout << "\n=== Testing Tropical Monomials ===\n";
    
    // Create monomials
    Exponent exp1{2, 1};  // x^2 * y
    Exponent exp2{1, 3};  // x * y^3
    
    TropicalMonomial m1(1.5, exp1);
    TropicalMonomial m2(2.0, exp2);
    
    // Evaluate at point
    std::vector<TropicalNumber> point{TropicalNumber(0.5), TropicalNumber(1.0)};
    TropicalNumber val1 = m1.evaluate(point);
    // 1.5 + 2*0.5 + 1*1.0 = 1.5 + 1.0 + 1.0 = 3.5
    assert(std::abs(val1.value() - 3.5) < 1e-10);
    std::cout << "✓ Monomial evaluation: " << val1.value() << "\n";
    
    // Multiply monomials
    TropicalMonomial m3 = m1 * m2;
    // exp: {2+1, 1+3} = {3, 4}
    // coeff: 1.5 + 2.0 = 3.5
    assert(m3.exponent()[0] == 3);
    assert(m3.exponent()[1] == 4);
    assert(std::abs(m3.coefficient().value() - 3.5) < 1e-10);
    std::cout << "✓ Monomial multiplication\n";
}

void test_tropical_polynomials() {
    std::cout << "\n=== Testing Tropical Polynomials ===\n";
    
    // Create polynomial: max(1 + 2x + y, 0.5 + x + 3y)
    TropicalPolynomial poly(2);
    
    poly.add_monomial(TropicalMonomial(1.0, {2, 1}));
    poly.add_monomial(TropicalMonomial(0.5, {1, 3}));
    
    assert(poly.num_monomials() == 2);
    std::cout << "✓ Polynomial has " << poly.num_monomials() << " monomials\n";
    
    // Evaluate
    std::vector<TropicalNumber> point{TropicalNumber(0.5), TropicalNumber(1.0)};
    TropicalNumber result = poly.evaluate(point);
    // max(1 + 2*0.5 + 1*1.0, 0.5 + 1*0.5 + 3*1.0)
    // = max(1 + 1 + 1, 0.5 + 0.5 + 3)
    // = max(3, 4) = 4
    assert(std::abs(result.value() - 4.0) < 1e-10);
    std::cout << "✓ Polynomial evaluation: " << result.value() << "\n";
    
    // Polynomial addition
    TropicalPolynomial poly2(2);
    poly2.add_monomial(TropicalMonomial(2.0, {1, 1}));
    
    TropicalPolynomial sum = poly + poly2;
    assert(sum.num_monomials() == 3);
    std::cout << "✓ Polynomial addition\n";
}

void test_newton_polytope() {
    std::cout << "\n=== Testing Newton Polytope ===\n";
    
    // Create simple polynomial
    TropicalPolynomial poly(2);
    poly.add_monomial(TropicalMonomial(1.0, {0, 0}));
    poly.add_monomial(TropicalMonomial(1.0, {1, 0}));
    poly.add_monomial(TropicalMonomial(1.0, {0, 1}));
    poly.add_monomial(TropicalMonomial(1.0, {1, 1}));
    
    NewtonPolytope polytope(poly);
    
    std::cout << "✓ Newton polytope has " << polytope.num_vertices() << " vertices\n";
    std::cout << "  Volume: " << polytope.volume() << "\n";
    
    int bound = polytope.linear_region_upper_bound();
    std::cout << "  Linear region upper bound: " << bound << "\n";
    assert(bound > 0);
}

void test_relu_network() {
    std::cout << "\n=== Testing ReLU Network ===\n";
    
    // Create simple network: 2 → 3 → 1
    ReLUNetwork network(2);
    
    auto w1 = torch::tensor({{1.0, 0.5}, {-0.5, 1.0}, {0.0, -1.0}});
    auto b1 = torch::tensor({0.1, -0.2, 0.3});
    network.add_layer(w1, b1);
    
    auto w2 = torch::tensor({{0.8, -0.3, 0.5}});
    auto b2 = torch::tensor({-0.1});
    network.add_layer(w2, b2);
    
    int params = network.num_parameters();
    std::cout << "✓ Network has " << params << " parameters\n";
    assert(params == 2*3 + 3 + 3*1 + 1);  // 13 parameters
    
    // Forward pass
    auto input = torch::tensor({{1.0, 2.0}});
    auto output = network.forward(input);
    
    std::cout << "✓ Network forward pass output: " << output << "\n";
    assert(output.numel() == 1);
}

void test_tropical_converter() {
    std::cout << "\n=== Testing ReLU to Tropical Conversion ===\n";
    
    // Create tiny network: 2 → 2 → 1
    ReLUNetwork network(2);
    
    auto w1 = torch::tensor({{1.0, -1.0}, {-1.0, 1.0}});
    auto b1 = torch::tensor({0.0, 0.0});
    network.add_layer(w1, b1);
    
    auto w2 = torch::tensor({{1.0, 1.0}});
    auto b2 = torch::tensor({0.0});
    network.add_layer(w2, b2);
    
    TropicalConverter converter;
    auto tropical_polys = converter.convert(network);
    
    std::cout << "✓ Converted network to " << tropical_polys.size() 
              << " tropical polynomials\n";
    
    for (size_t i = 0; i < tropical_polys.size(); ++i) {
        std::cout << "  Polynomial " << i << " has " 
                  << tropical_polys[i].num_monomials() << " monomials\n";
    }
}

void test_linear_region_counting() {
    std::cout << "\n=== Testing Linear Region Counting ===\n";
    
    // Small network for exact counting
    ReLUNetwork network(2);
    
    auto w1 = torch::tensor({{1.0, 0.0}, {0.0, 1.0}});
    auto b1 = torch::tensor({0.0, 0.0});
    network.add_layer(w1, b1);
    
    LinearRegionEnumerator enumerator(network);
    
    int approx_count = enumerator.count_approximate(10000);
    std::cout << "✓ Approximate region count: " << approx_count << "\n";
    
    int upper = enumerator.count_upper_bound();
    std::cout << "✓ Upper bound: " << upper << "\n";
    
    int lower = enumerator.count_lower_bound();
    std::cout << "✓ Lower bound: " << lower << "\n";
    
    assert(lower <= approx_count);
    assert(approx_count <= upper);
}

void test_network_complexity() {
    std::cout << "\n=== Testing Network Complexity Analysis ===\n";
    
    // Create two networks to compare
    ReLUNetwork net1(4);
    net1.add_layer(torch::randn({8, 4}) * 0.1, torch::randn({8}) * 0.1);
    net1.add_layer(torch::randn({2, 8}) * 0.1, torch::randn({2}) * 0.1);
    
    ReLUNetwork net2(4);
    net2.add_layer(torch::randn({16, 4}) * 0.1, torch::randn({16}) * 0.1);
    net2.add_layer(torch::randn({2, 16}) * 0.1, torch::randn({2}) * 0.1);
    
    auto complexity1 = compute_network_complexity(net1, false);
    auto complexity2 = compute_network_complexity(net2, false);
    
    std::cout << "\nNetwork 1 (4→8→2):\n";
    complexity1.print();
    
    std::cout << "\nNetwork 2 (4→16→2):\n";
    complexity2.print();
    
    auto comparison = compare_architectures(net1, net2, false);
    comparison.print();
}

void test_random_search() {
    std::cout << "\n=== Testing Random Architecture Search ===\n";
    
    SearchConstraints constraints;
    constraints.min_layers = 1;
    constraints.max_layers = 3;
    constraints.min_width = 4;
    constraints.max_width = 16;
    constraints.max_parameters = 500;
    
    auto objective = std::make_shared<RegionsPerParameterObjective>();
    
    RandomSearch search(constraints, objective, 42);
    
    auto results = search.search(4, 2, 20);  // 20 iterations for testing
    
    std::cout << "\nTop 3 architectures found:\n";
    for (int i = 0; i < std::min(3, static_cast<int>(results.size())); ++i) {
        std::cout << "\n" << (i+1) << ". ";
        results[i].print();
    }
}

void test_evolutionary_search() {
    std::cout << "\n=== Testing Evolutionary Architecture Search ===\n";
    
    SearchConstraints constraints;
    constraints.min_layers = 1;
    constraints.max_layers = 3;
    constraints.min_width = 4;
    constraints.max_width = 16;
    constraints.max_parameters = 300;
    
    auto objective = std::make_shared<PolytopeVolumeObjective>();
    
    EvolutionarySearch search(constraints, objective, 10, 42);  // Population of 10
    
    auto results = search.search(4, 2, 5);  // 5 generations for testing
    
    std::cout << "\nFinal population (top 3):\n";
    for (int i = 0; i < std::min(3, static_cast<int>(results.size())); ++i) {
        std::cout << "\n" << (i+1) << ". " << results[i].architecture.to_string();
        std::cout << "  (obj=" << results[i].objective_value << ")\n";
    }
}

int main() {
    std::cout << "==============================================\n";
    std::cout << "TROPICAL GEOMETRY NAS - COMPREHENSIVE TESTS\n";
    std::cout << "==============================================\n";
    
    try {
        test_tropical_arithmetic();
        test_tropical_monomials();
        test_tropical_polynomials();
        test_newton_polytope();
        test_relu_network();
        test_tropical_converter();
        test_linear_region_counting();
        test_network_complexity();
        test_random_search();
        test_evolutionary_search();
        
        std::cout << "\n==============================================\n";
        std::cout << "ALL TESTS PASSED! ✓\n";
        std::cout << "==============================================\n";
        
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "\nTEST FAILED: " << e.what() << "\n";
        return 1;
    }
}
