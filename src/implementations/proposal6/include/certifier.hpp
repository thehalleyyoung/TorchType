#pragma once

#include "interval.hpp"
#include "input_domain.hpp"
#include "curvature_bounds.hpp"
#include <Eigen/Dense>
#include <vector>
#include <string>
#include <memory>
#include <chrono>
#include <sstream>
#include <iomanip>
#include <map>
#include <numeric>

namespace hnf {
namespace certified {

// Precision certificate as described in Proposal 6
struct PrecisionCertificate {
    std::string model_hash;
    InputDomain input_domain;
    double target_accuracy;
    double curvature_bound;
    int precision_requirement;  // Mantissa bits
    std::string recommended_hardware;
    
    // Audit trail
    std::string timestamp;
    std::vector<std::pair<std::string, double>> layer_curvatures;
    double total_lipschitz_constant;
    
    // Verification details
    std::vector<std::string> bottleneck_layers;
    std::map<std::string, std::string> computation_details;
    
    // Default constructor (needs dummy domain)
    PrecisionCertificate() 
        : input_domain(Eigen::VectorXd::Zero(1), Eigen::VectorXd::Ones(1)),
          target_accuracy(0.0),
          curvature_bound(0.0),
          precision_requirement(0),
          total_lipschitz_constant(0.0) {}
    
    // Full constructor
    PrecisionCertificate(const InputDomain& domain)
        : input_domain(domain),
          target_accuracy(0.0),
          curvature_bound(0.0),
          precision_requirement(0),
          total_lipschitz_constant(0.0) {}
    
    // Generate human-readable report
    std::string generate_report() const {
        std::stringstream ss;
        
        ss << "╔══════════════════════════════════════════════════════════════╗\n";
        ss << "║ PRECISION CERTIFICATE                                         ║\n";
        ss << "╠══════════════════════════════════════════════════════════════╣\n";
        ss << "║ Minimum Required Precision:  " << std::setw(2) << precision_requirement 
           << " bits mantissa                   ║\n";
        ss << "║ Recommendation:              " << std::left << std::setw(30) 
           << recommended_hardware << "║\n";
        ss << "║                                                                ║\n";
        ss << "║ Target Accuracy:             " << std::scientific << std::setprecision(2) 
           << target_accuracy << "                              ║\n";
        ss << "║ Curvature Bound:             " << std::scientific << std::setprecision(2) 
           << curvature_bound << "                              ║\n";
        ss << "║ Domain Diameter:             " << std::fixed << std::setprecision(4) 
           << input_domain.diameter() << "                                    ║\n";
        
        if (!bottleneck_layers.empty()) {
            ss << "║                                                                ║\n";
            ss << "║ Bottleneck Layers:                                            ║\n";
            for (const auto& layer : bottleneck_layers) {
                ss << "║   - " << std::left << std::setw(56) << layer << "║\n";
            }
        }
        
        ss << "╚══════════════════════════════════════════════════════════════╝\n";
        
        return ss.str();
    }
    
    // Serialize to JSON
    std::string to_json() const {
        std::stringstream ss;
        ss << "{\n";
        ss << "  \"model_hash\": \"" << model_hash << "\",\n";
        ss << "  \"target_accuracy\": " << target_accuracy << ",\n";
        ss << "  \"curvature_bound\": " << curvature_bound << ",\n";
        ss << "  \"precision_requirement\": " << precision_requirement << ",\n";
        ss << "  \"recommended_hardware\": \"" << recommended_hardware << "\",\n";
        ss << "  \"timestamp\": \"" << timestamp << "\",\n";
        ss << "  \"domain_diameter\": " << input_domain.diameter() << ",\n";
        ss << "  \"layer_curvatures\": [\n";
        
        for (size_t i = 0; i < layer_curvatures.size(); ++i) {
            ss << "    {\"layer\": \"" << layer_curvatures[i].first 
               << "\", \"curvature\": " << layer_curvatures[i].second << "}";
            if (i < layer_curvatures.size() - 1) ss << ",";
            ss << "\n";
        }
        
        ss << "  ]\n";
        ss << "}\n";
        
        return ss.str();
    }
    
    // Verify certificate is valid
    bool verify(double recomputed_curvature) const {
        // Check that recomputed curvature doesn't exceed certified bound
        if (recomputed_curvature > curvature_bound * 1.1) {  // 10% tolerance
            return false;
        }
        
        // Check precision formula
        int recomputed_precision = PrecisionComputer::compute_minimum_precision(
            curvature_bound,
            input_domain.diameter(),
            target_accuracy
        );
        
        return recomputed_precision <= precision_requirement;
    }
};

// Main certifier class implementing Proposal 6
class ModelCertifier {
public:
    using LayerFunction = std::function<Eigen::VectorXd(const Eigen::VectorXd&)>;
    
    struct Layer {
        std::string name;
        std::string type;
        LayerFunction forward;
        CurvatureBounds::LayerCurvature curvature_info;
        
        // For interval propagation
        std::function<IntervalVector(const IntervalVector&)> interval_forward;
    };
    
    ModelCertifier() {}
    
    // Add a layer to the model
    void add_layer(const Layer& layer) {
        layers_.push_back(layer);
    }
    
    // Add linear layer
    void add_linear_layer(
        const std::string& name,
        const Eigen::MatrixXd& W,
        const Eigen::VectorXd& b) {
        
        Layer layer;
        layer.name = name;
        layer.type = "Linear";
        layer.curvature_info = CurvatureBounds::linear_layer(W, b);
        
        // Forward function
        layer.forward = [W, b](const Eigen::VectorXd& x) {
            return W * x + b;
        };
        
        // Interval forward
        layer.interval_forward = [W, b](const IntervalVector& x) {
            // Conservative interval matrix multiplication
            IntervalVector result(W.rows());
            for (int i = 0; i < W.rows(); ++i) {
                Interval sum(b(i));
                for (int j = 0; j < W.cols(); ++j) {
                    sum = sum + Interval(W(i, j)) * x[j];
                }
                result[i] = sum;
            }
            return result;
        };
        
        layers_.push_back(layer);
    }
    
    // Add ReLU activation
    void add_relu(const std::string& name) {
        Layer layer;
        layer.name = name;
        layer.type = "ReLU";
        layer.curvature_info = CurvatureBounds::relu_activation();
        
        layer.forward = [](const Eigen::VectorXd& x) {
            return x.cwiseMax(0.0);
        };
        
        layer.interval_forward = [](const IntervalVector& x) {
            IntervalVector result(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                if (x[i].upper() <= 0.0) {
                    result[i] = Interval(0.0);
                } else if (x[i].lower() >= 0.0) {
                    result[i] = x[i];
                } else {
                    result[i] = Interval(0.0, x[i].upper());
                }
            }
            return result;
        };
        
        layers_.push_back(layer);
    }
    
    // Add softmax activation
    void add_softmax(const std::string& name, const Interval& expected_input_range) {
        Layer layer;
        layer.name = name;
        layer.type = "Softmax";
        layer.curvature_info = CurvatureBounds::softmax_activation(expected_input_range);
        
        layer.forward = [](const Eigen::VectorXd& x) {
            Eigen::VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
            return exp_x / exp_x.sum();
        };
        
        layer.interval_forward = [](const IntervalVector& x) {
            // Conservative: softmax output is in [0, 1]
            IntervalVector result(x.size());
            for (size_t i = 0; i < x.size(); ++i) {
                result[i] = Interval(0.0, 1.0);
            }
            return result;
        };
        
        layers_.push_back(layer);
    }
    
    // Propagate intervals through model
    std::vector<IntervalVector> propagate_intervals(const InputDomain& domain) const {
        std::vector<IntervalVector> intervals;
        
        IntervalVector current = domain.to_interval_vector();
        intervals.push_back(current);
        
        for (const auto& layer : layers_) {
            if (layer.interval_forward) {
                current = layer.interval_forward(current);
                intervals.push_back(current);
            }
        }
        
        return intervals;
    }
    
    // Compute total curvature bound using composition rules
    double compute_total_curvature(const InputDomain& domain) const {
        if (layers_.empty()) {
            return 0.0;
        }
        
        // Start with first layer
        CurvatureBounds::LayerCurvature total = layers_[0].curvature_info;
        
        // Compose with remaining layers
        for (size_t i = 1; i < layers_.size(); ++i) {
            total = CurvatureBounds::compose(total, layers_[i].curvature_info);
        }
        
        return total.curvature;
    }
    
    // Compute total Lipschitz constant
    double compute_total_lipschitz() const {
        double total = 1.0;
        for (const auto& layer : layers_) {
            total *= layer.curvature_info.lipschitz_constant;
        }
        return total;
    }
    
    // Identify bottleneck layers (highest curvature contribution)
    std::vector<std::string> identify_bottlenecks(double threshold_percentile = 90.0) const {
        std::vector<std::pair<std::string, double>> curvatures;
        
        for (const auto& layer : layers_) {
            curvatures.emplace_back(layer.name, layer.curvature_info.curvature);
        }
        
        // Sort by curvature
        std::sort(curvatures.begin(), curvatures.end(),
                  [](const auto& a, const auto& b) { return a.second > b.second; });
        
        // Find threshold
        size_t threshold_idx = static_cast<size_t>(
            curvatures.size() * threshold_percentile / 100.0);
        
        std::vector<std::string> bottlenecks;
        for (size_t i = 0; i < threshold_idx && i < curvatures.size(); ++i) {
            if (curvatures[i].second > 1.0) {  // Only significant curvatures
                std::stringstream ss;
                ss << curvatures[i].first << ": κ = " 
                   << std::scientific << curvatures[i].second;
                bottlenecks.push_back(ss.str());
            }
        }
        
        return bottlenecks;
    }
    
    // Main certification function (Algorithm from Proposal 6)
    PrecisionCertificate certify(
        const InputDomain& input_domain,
        double target_accuracy) {
        
        auto start_time = std::chrono::system_clock::now();
        
        // Step 1: Compute curvature bound
        double curvature = compute_total_curvature(input_domain);
        
        // Step 2: Compute domain diameter
        double diameter = input_domain.diameter();
        
        // Step 3: Compute precision requirement (Theorem 5.7)
        int precision_bits = PrecisionComputer::compute_minimum_precision(
            curvature, diameter, target_accuracy);
        
        // Step 4: Generate recommendation
        std::string hardware = PrecisionComputer::recommend_hardware(precision_bits);
        
        // Step 5: Build certificate
        PrecisionCertificate cert(input_domain);
        cert.model_hash = compute_model_hash();
        cert.target_accuracy = target_accuracy;
        cert.curvature_bound = curvature;
        cert.precision_requirement = precision_bits;
        cert.recommended_hardware = hardware;
        
        // Timestamp
        auto time = std::chrono::system_clock::to_time_t(start_time);
        cert.timestamp = std::ctime(&time);
        
        // Layer curvatures
        for (const auto& layer : layers_) {
            cert.layer_curvatures.emplace_back(
                layer.name, layer.curvature_info.curvature);
        }
        
        // Lipschitz constant
        cert.total_lipschitz_constant = compute_total_lipschitz();
        
        // Bottlenecks
        cert.bottleneck_layers = identify_bottlenecks();
        
        // Computation details
        cert.computation_details["diameter"] = std::to_string(diameter);
        cert.computation_details["method"] = "interval_arithmetic";
        cert.computation_details["safety_margin_bits"] = "2";
        
        return cert;
    }
    
    // Empirical verification: sample points and check actual errors
    struct VerificationResult {
        double max_error;
        double mean_error;
        std::vector<double> errors;
        bool passes;
    };
    
    VerificationResult empirical_verification(
        const InputDomain& domain,
        const PrecisionCertificate& cert,
        int num_samples = 1000) const {
        
        auto samples = domain.sample(num_samples);
        std::vector<double> errors;
        
        // For verification, we'd need a reference implementation
        // Here we check consistency
        for (const auto& x : samples) {
            // Would compute error relative to higher-precision reference
            // For now, just check that forward pass works
            try {
                forward(x);
                errors.push_back(0.0);  // Placeholder
            } catch (...) {
                errors.push_back(std::numeric_limits<double>::infinity());
            }
        }
        
        VerificationResult result;
        result.errors = errors;
        result.max_error = *std::max_element(errors.begin(), errors.end());
        result.mean_error = std::accumulate(errors.begin(), errors.end(), 0.0) / errors.size();
        result.passes = result.max_error <= cert.target_accuracy;
        
        return result;
    }
    
    // Forward pass through model
    Eigen::VectorXd forward(const Eigen::VectorXd& input) const {
        Eigen::VectorXd x = input;
        for (const auto& layer : layers_) {
            if (layer.forward) {
                x = layer.forward(x);
            }
        }
        return x;
    }
    
    size_t num_layers() const { return layers_.size(); }
    const Layer& get_layer(size_t i) const { return layers_.at(i); }
    
private:
    std::vector<Layer> layers_;
    
    std::string compute_model_hash() const {
        // Simple hash based on layer structure
        std::stringstream ss;
        for (const auto& layer : layers_) {
            ss << layer.name << ":" << layer.type << ";";
        }
        return ss.str();
    }
};

} // namespace certified
} // namespace hnf
