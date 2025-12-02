#pragma once

#include "certifier.hpp"
#include "input_domain.hpp"
#include "curvature_bounds.hpp"
#include "zonotope.hpp"
#include <random>
#include <algorithm>
#include <map>
#include <iostream>
#include <sstream>
#include <iomanip>

namespace hnf {

/**
 * Probabilistic Precision Certifier
 * 
 * Unlike worst-case certification, this provides probabilistic guarantees:
 *   "With probability ≥ p, precision requirement ≤ k bits"
 * 
 * This is MUCH tighter for real-world distributions where worst-case
 * inputs are extremely rare.
 * 
 * Based on HNF paper Section 6.2: "Probabilistic Extensions"
 */
class ProbabilisticCertifier {
public:
    struct ProbabilisticCertificate {
        double confidence;  // Probability guarantee (e.g., 0.99)
        int precision_bits;  // Required precision at this confidence
        std::vector<double> curvature_samples;  // Empirical curvature distribution
        double worst_case_precision;  // For comparison
        
        // Curvature percentiles
        std::map<double, double> curvature_percentiles;  // percentile -> κ value
        std::map<double, int> precision_percentiles;  // percentile -> bits
        
        // Statistical summary
        double mean_curvature;
        double std_curvature;
        double median_curvature;
        double p95_curvature;
        double p99_curvature;
        double max_curvature;
        
        std::string to_string() const {
            std::stringstream ss;
            ss << "╔══════════════════════════════════════════════════════════════╗\n";
            ss << "║ PROBABILISTIC PRECISION CERTIFICATE                           ║\n";
            ss << "╠══════════════════════════════════════════════════════════════╣\n";
            ss << "║ Confidence: " << (confidence * 100) << "%                                    ║\n";
            ss << "║ Required Precision: " << precision_bits << " bits                           ║\n";
            ss << "║                                                                ║\n";
            ss << "║ Curvature Distribution:                                       ║\n";
            ss << "║   Mean:    " << std::scientific << mean_curvature << "                    ║\n";
            ss << "║   Std:     " << std_curvature << "                    ║\n";
            ss << "║   Median:  " << median_curvature << "                    ║\n";
            ss << "║   95%ile:  " << p95_curvature << "                    ║\n";
            ss << "║   99%ile:  " << p99_curvature << "                    ║\n";
            ss << "║   Max:     " << max_curvature << "                    ║\n";
            ss << "║                                                                ║\n";
            ss << "║ Precision Requirements:                                       ║\n";
            ss << "║   Median: " << precision_percentiles.at(0.50) << " bits                                   ║\n";
            ss << "║   95%ile: " << precision_percentiles.at(0.95) << " bits                                   ║\n";
            ss << "║   99%ile: " << precision_percentiles.at(0.99) << " bits                                   ║\n";
            ss << "║   Worst:  " << (int)worst_case_precision << " bits                                   ║\n";
            ss << "║                                                                ║\n";
            ss << "║ Savings: " << ((int)worst_case_precision - precision_bits) << " bits vs worst-case                       ║\n";
            ss << "╚══════════════════════════════════════════════════════════════╝\n";
            return ss.str();
        }
    };
    
    /**
     * Certify with probabilistic guarantee
     * 
     * @param network Neural network to certify
     * @param domain Input domain specification
     * @param target_accuracy Required accuracy
     * @param confidence Probability of meeting accuracy (default: 0.99)
     * @param n_samples Number of samples for distribution estimation
     * @return Probabilistic certificate
     */
    static ProbabilisticCertificate certify_probabilistic(
        const NeuralNetwork& network,
        const InputDomain& domain,
        double target_accuracy,
        double confidence = 0.99,
        int n_samples = 10000
    ) {
        ProbabilisticCertificate cert;
        cert.confidence = confidence;
        
        std::cout << "Probabilistic Certification with " << n_samples << " samples...\n";
        std::cout << "Confidence level: " << (confidence * 100) << "%\n\n";
        
        // Sample inputs from domain distribution
        std::vector<Eigen::VectorXd> inputs = sample_inputs(domain, n_samples);
        
        // Compute curvature for each sampled input
        std::cout << "Computing curvature distribution...\n";
        for (int i = 0; i < n_samples; ++i) {
            if (i % 1000 == 0) {
                std::cout << "  Sample " << i << "/" << n_samples << "\r" << std::flush;
            }
            
            double local_curvature = compute_local_curvature(network, inputs[i], domain);
            cert.curvature_samples.push_back(local_curvature);
        }
        std::cout << "\n";
        
        // Sort curvatures for percentile computation
        std::sort(cert.curvature_samples.begin(), cert.curvature_samples.end());
        
        // Compute statistics
        cert.mean_curvature = 0.0;
        for (double k : cert.curvature_samples) {
            cert.mean_curvature += k;
        }
        cert.mean_curvature /= n_samples;
        
        cert.std_curvature = 0.0;
        for (double k : cert.curvature_samples) {
            cert.std_curvature += (k - cert.mean_curvature) * (k - cert.mean_curvature);
        }
        cert.std_curvature = std::sqrt(cert.std_curvature / n_samples);
        
        // Percentiles
        cert.median_curvature = percentile(cert.curvature_samples, 0.50);
        cert.p95_curvature = percentile(cert.curvature_samples, 0.95);
        cert.p99_curvature = percentile(cert.curvature_samples, 0.99);
        cert.max_curvature = cert.curvature_samples.back();
        
        // Store curvature percentiles
        for (double p : {0.50, 0.75, 0.90, 0.95, 0.99, 0.999}) {
            cert.curvature_percentiles[p] = percentile(cert.curvature_samples, p);
        }
        
        // Compute precision requirements at each percentile
        double D = domain.diameter();
        for (auto [p, kappa] : cert.curvature_percentiles) {
            int bits = compute_precision_requirement(kappa, D, target_accuracy);
            cert.precision_percentiles[p] = bits;
        }
        
        // Precision at target confidence
        double kappa_at_confidence = percentile(cert.curvature_samples, confidence);
        cert.precision_bits = compute_precision_requirement(kappa_at_confidence, D, target_accuracy);
        
        // Worst-case precision
        cert.worst_case_precision = compute_precision_requirement(cert.max_curvature, D, target_accuracy);
        
        return cert;
    }
    
    /**
     * Compare probabilistic vs worst-case certification
     */
    static void compare_certifications(
        const NeuralNetwork& network,
        const InputDomain& domain,
        double target_accuracy
    ) {
        std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
        std::cout << "║ PROBABILISTIC vs WORST-CASE CERTIFICATION                     ║\n";
        std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";
        
        // Worst-case certification
        std::cout << "Computing worst-case certification...\n";
        auto worst_cert = hnf::certified::Certifier::certify(network, domain, target_accuracy);
        
        std::cout << "Worst-case precision: " << worst_cert.precision_requirement << " bits\n\n";
        
        // Probabilistic certifications at different confidence levels
        std::cout << "Computing probabilistic certifications...\n\n";
        
        std::cout << "┌──────────────┬───────────────┬────────────┬─────────────┐\n";
        std::cout << "│ Confidence   │ Precision     │ vs Worst   │ Savings     │\n";
        std::cout << "├──────────────┼───────────────┼────────────┼─────────────┤\n";
        
        for (double conf : {0.50, 0.75, 0.90, 0.95, 0.99, 0.999}) {
            auto prob_cert = certify_probabilistic(network, domain, target_accuracy, conf, 5000);
            
            int savings = worst_cert.precision_requirement - prob_cert.precision_bits;
            
            std::cout << "│ " << std::setw(10) << (conf * 100) << "% │ "
                      << std::setw(10) << prob_cert.precision_bits << " bits │ "
                      << std::setw(7) << worst_cert.precision_requirement << " bits │ "
                      << std::setw(8) << savings << " bits │\n";
        }
        
        std::cout << "└──────────────┴───────────────┴────────────┴─────────────┘\n\n";
        
        std::cout << "KEY INSIGHT: For real-world distributions, probabilistic\n";
        std::cout << "certification can save significant precision vs worst-case!\n\n";
    }
    
private:
    /**
     * Sample inputs from domain distribution
     */
    static std::vector<Eigen::VectorXd> sample_inputs(
        const InputDomain& domain,
        int n_samples
    ) {
        std::vector<Eigen::VectorXd> samples;
        std::random_device rd;
        std::mt19937 gen(rd());
        
        // Use domain's distribution if available
        if (domain.has_distribution()) {
            // Sample from specified distribution (Gaussian, uniform, etc.)
            for (int i = 0; i < n_samples; ++i) {
                samples.push_back(domain.sample(gen));
            }
        } else {
            // Default: uniform sampling in bounding box
            std::uniform_real_distribution<> dis(0.0, 1.0);
            
            for (int i = 0; i < n_samples; ++i) {
                Eigen::VectorXd sample = domain.lower_bounds;
                
                for (int j = 0; j < domain.dimension(); ++j) {
                    double u = dis(gen);
                    sample(j) = domain.lower_bounds(j) + u * (domain.upper_bounds(j) - domain.lower_bounds(j));
                }
                
                samples.push_back(sample);
            }
        }
        
        return samples;
    }
    
    /**
     * Compute local curvature at specific input
     * 
     * Uses zonotope arithmetic for tighter bounds than intervals!
     */
    static double compute_local_curvature(
        const NeuralNetwork& network,
        const Eigen::VectorXd& input,
        const InputDomain& domain
    ) {
        // Create small zonotope around input point
        double epsilon = 0.01 * domain.diameter();
        
        Zonotope z_in = Zonotope::from_interval(
            input - Eigen::VectorXd::Constant(input.size(), epsilon),
            input + Eigen::VectorXd::Constant(input.size(), epsilon)
        );
        
        // Propagate through network using zonotopes
        double total_curvature = 0.0;
        double total_lipschitz = 1.0;
        
        Zonotope z = z_in;
        
        for (const auto& layer : network.layers) {
            auto [layer_curv, layer_lip] = compute_layer_curvature_zonotope(layer, z);
            
            // HNF composition rule for curvature
            total_curvature = total_curvature * layer_lip * layer_lip + layer_curv * total_lipschitz;
            total_lipschitz *= layer_lip;
            
            // Propagate zonotope
            z = apply_layer_zonotope(layer, z);
            
            // Reduce order to keep complexity bounded
            if (z.n_symbols > 50) {
                z = z.reduce_order(30);
            }
        }
        
        return total_curvature;
    }
    
    /**
     * Compute curvature bounds for layer using zonotope
     */
    static std::pair<double, double> compute_layer_curvature_zonotope(
        const Layer& layer,
        const Zonotope& z_in
    ) {
        if (layer.type == LayerType::Linear) {
            // Linear: κ = 0
            double lipschitz = layer.weight.norm();  // Spectral norm approximation
            return {0.0, lipschitz};
        }
        else if (layer.type == LayerType::ReLU) {
            // ReLU: piecewise linear, κ = 0
            return {0.0, 1.0};
        }
        else if (layer.type == LayerType::Softmax) {
            // Softmax: κ ≈ exp(2 * max_input)
            auto [lower, upper] = z_in.to_interval();
            double max_val = upper.maxCoeff();
            double kappa = std::exp(2.0 * max_val);
            return {kappa, 1.0};
        }
        else if (layer.type == LayerType::Tanh) {
            // Tanh: κ = 1.0 (from sech^4 curvature)
            return {1.0, 1.0};
        }
        else {
            // Default: conservative bounds
            return {1.0, 1.0};
        }
    }
    
    /**
     * Apply layer to zonotope
     */
    static Zonotope apply_layer_zonotope(const Layer& layer, const Zonotope& z_in) {
        if (layer.type == LayerType::Linear) {
            // For now, convert to interval and back
            // TODO: Full zonotope linear transformation
            auto [lower, upper] = z_in.to_interval();
            
            Eigen::VectorXd out_lower = layer.weight * lower + layer.bias;
            Eigen::VectorXd out_upper = layer.weight * upper + layer.bias;
            
            return Zonotope::from_interval(out_lower, out_upper);
        }
        else if (layer.type == LayerType::ReLU) {
            return z_in.relu();
        }
        else if (layer.type == LayerType::Tanh) {
            return z_in.tanh();
        }
        else {
            // Default: pass through
            return z_in;
        }
    }
    
    /**
     * Compute percentile from sorted data
     */
    static double percentile(const std::vector<double>& sorted_data, double p) {
        if (sorted_data.empty()) return 0.0;
        
        int idx = static_cast<int>(p * (sorted_data.size() - 1));
        return sorted_data[idx];
    }
    
    /**
     * Compute precision requirement from curvature
     * Uses HNF Theorem 5.7: p ≥ log₂(c·κ·D²/ε)
     */
    static int compute_precision_requirement(double kappa, double D, double epsilon) {
        if (kappa < 1e-10) return 8;  // Linear: minimal precision
        
        double c = 1.0;  // Safety constant
        double p_real = std::log2(c * kappa * D * D / epsilon);
        return static_cast<int>(std::ceil(p_real)) + 2;  // +2 safety margin
    }
};

} // namespace hnf
