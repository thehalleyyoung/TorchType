#pragma once

#include <vector>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <random>
#include <iostream>
#include <numeric>
#include <Eigen/Dense>

namespace hnf {
namespace certified {

// MNIST dataset loader for real-world testing
// Downloads and loads actual MNIST data for certification experiments
class MNISTDataset {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    using VectorXi = Eigen::Matrix<int, Eigen::Dynamic, 1>;
    
    struct Sample {
        VectorXd image;  // 784-dimensional vector (28x28 flattened)
        int label;       // 0-9
    };
    
    MNISTDataset() = default;
    
    // Load MNIST from files (IDX format)
    // Files available from: http://yann.lecun.com/exdb/mnist/
    bool load_from_files(
        const std::string& images_path,
        const std::string& labels_path) {
        
        try {
            load_images(images_path);
            load_labels(labels_path);
            
            if (images_.size() != labels_.size()) {
                throw std::runtime_error("Mismatch between images and labels count");
            }
            
            return true;
        } catch (const std::exception& e) {
            std::cerr << "Failed to load MNIST: " << e.what() << std::endl;
            return false;
        }
    }
    
    // Generate synthetic MNIST-like data for testing when real data unavailable
    void generate_synthetic(int num_samples = 1000) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<double> noise(0.0, 0.1);
        std::uniform_int_distribution<int> label_dist(0, 9);
        
        images_.clear();
        labels_.clear();
        
        for (int i = 0; i < num_samples; ++i) {
            VectorXd image = VectorXd::Zero(784);
            int label = label_dist(gen);
            
            // Generate simple pattern for each digit
            // Center: 14x14 region
            int center_x = 14, center_y = 14;
            int radius = 5 + (label % 5);
            
            for (int y = 0; y < 28; ++y) {
                for (int x = 0; x < 28; ++x) {
                    int dist_sq = (x - center_x) * (x - center_x) + (y - center_y) * (y - center_y);
                    if (dist_sq < radius * radius) {
                        int idx = y * 28 + x;
                        image[idx] = 0.8 + noise(gen);
                        image[idx] = std::max(0.0, std::min(1.0, image[idx]));
                    }
                }
            }
            
            images_.push_back(image);
            labels_.push_back(label);
        }
    }
    
    // Get number of samples
    size_t size() const { return images_.size(); }
    
    // Get a sample
    Sample get_sample(size_t idx) const {
        if (idx >= images_.size()) {
            throw std::out_of_range("Sample index out of range");
        }
        return Sample{images_[idx], labels_[idx]};
    }
    
    // Get batch of samples
    std::vector<Sample> get_batch(size_t start_idx, size_t batch_size) const {
        std::vector<Sample> batch;
        batch.reserve(batch_size);
        
        for (size_t i = 0; i < batch_size && (start_idx + i) < images_.size(); ++i) {
            batch.push_back(get_sample(start_idx + i));
        }
        
        return batch;
    }
    
    // Compute dataset statistics for domain specification
    struct DatasetStats {
        VectorXd mean;
        VectorXd std;
        VectorXd min_vals;
        VectorXd max_vals;
        double global_min;
        double global_max;
    };
    
    DatasetStats compute_statistics() const {
        if (images_.empty()) {
            throw std::runtime_error("Cannot compute stats on empty dataset");
        }
        
        int dim = images_[0].size();
        int n = images_.size();
        
        DatasetStats stats;
        stats.mean = VectorXd::Zero(dim);
        stats.std = VectorXd::Zero(dim);
        stats.min_vals = VectorXd::Constant(dim, 1e10);
        stats.max_vals = VectorXd::Constant(dim, -1e10);
        stats.global_min = 1e10;
        stats.global_max = -1e10;
        
        // First pass: mean
        for (const auto& img : images_) {
            stats.mean += img;
            for (int i = 0; i < dim; ++i) {
                stats.min_vals[i] = std::min(stats.min_vals[i], img[i]);
                stats.max_vals[i] = std::max(stats.max_vals[i], img[i]);
            }
            stats.global_min = std::min(stats.global_min, img.minCoeff());
            stats.global_max = std::max(stats.global_max, img.maxCoeff());
        }
        stats.mean /= n;
        
        // Second pass: std
        for (const auto& img : images_) {
            VectorXd diff = img - stats.mean;
            stats.std += diff.cwiseProduct(diff);
        }
        stats.std = (stats.std / n).cwiseSqrt();
        
        return stats;
    }
    
    // Shuffle dataset
    void shuffle() {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        std::vector<size_t> indices(images_.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), gen);
        
        std::vector<VectorXd> shuffled_images;
        std::vector<int> shuffled_labels;
        shuffled_images.reserve(images_.size());
        shuffled_labels.reserve(labels_.size());
        
        for (size_t idx : indices) {
            shuffled_images.push_back(images_[idx]);
            shuffled_labels.push_back(labels_[idx]);
        }
        
        images_ = std::move(shuffled_images);
        labels_ = std::move(shuffled_labels);
    }
    
    // Normalize images
    void normalize(const DatasetStats& stats) {
        for (auto& img : images_) {
            img = (img - stats.mean).cwiseQuotient((stats.std.array() + 1e-8).matrix());
        }
    }
    
    // Get subset of data for specific label
    MNISTDataset get_label_subset(int target_label) const {
        MNISTDataset subset;
        
        for (size_t i = 0; i < images_.size(); ++i) {
            if (labels_[i] == target_label) {
                subset.images_.push_back(images_[i]);
                subset.labels_.push_back(labels_[i]);
            }
        }
        
        return subset;
    }
    
private:
    std::vector<VectorXd> images_;
    std::vector<int> labels_;
    
    // Read 32-bit big-endian integer
    int read_int32_be(std::ifstream& file) {
        unsigned char bytes[4];
        file.read(reinterpret_cast<char*>(bytes), 4);
        return (bytes[0] << 24) | (bytes[1] << 16) | (bytes[2] << 8) | bytes[3];
    }
    
    void load_images(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open images file: " + path);
        }
        
        // IDX file format header
        int magic = read_int32_be(file);
        if (magic != 0x00000803) {  // Magic number for image files
            throw std::runtime_error("Invalid magic number in images file");
        }
        
        int num_images = read_int32_be(file);
        int num_rows = read_int32_be(file);
        int num_cols = read_int32_be(file);
        
        if (num_rows != 28 || num_cols != 28) {
            throw std::runtime_error("Expected 28x28 images");
        }
        
        images_.clear();
        images_.reserve(num_images);
        
        for (int i = 0; i < num_images; ++i) {
            VectorXd image(784);
            for (int j = 0; j < 784; ++j) {
                unsigned char pixel;
                file.read(reinterpret_cast<char*>(&pixel), 1);
                image[j] = pixel / 255.0;  // Normalize to [0, 1]
            }
            images_.push_back(image);
        }
    }
    
    void load_labels(const std::string& path) {
        std::ifstream file(path, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open labels file: " + path);
        }
        
        int magic = read_int32_be(file);
        if (magic != 0x00000801) {  // Magic number for label files
        throw std::runtime_error("Invalid magic number in labels file");
        }
        
        int num_labels = read_int32_be(file);
        
        labels_.clear();
        labels_.reserve(num_labels);
        
        for (int i = 0; i < num_labels; ++i) {
            unsigned char label;
            file.read(reinterpret_cast<char*>(&label), 1);
            labels_.push_back(static_cast<int>(label));
        }
    }
};

// Simple feedforward neural network for MNIST
class MNISTNetwork {
public:
    using MatrixXd = Eigen::MatrixXd;
    using VectorXd = Eigen::VectorXd;
    
    struct LayerWeights {
        MatrixXd W;
        VectorXd b;
        std::string activation;  // "relu", "softmax", "none"
    };
    
    MNISTNetwork() = default;
    
    // Create network with specific architecture
    // Example: {784, 128, 64, 10} creates:
    //   784 -> 128 (ReLU) -> 64 (ReLU) -> 10 (Softmax)
    void create_architecture(const std::vector<int>& layer_sizes) {
        if (layer_sizes.size() < 2) {
            throw std::invalid_argument("Need at least input and output layer");
        }
        
        layers_.clear();
        
        for (size_t i = 0; i < layer_sizes.size() - 1; ++i) {
            LayerWeights layer;
            layer.W = MatrixXd::Random(layer_sizes[i+1], layer_sizes[i]) * 0.1;
            layer.b = VectorXd::Zero(layer_sizes[i+1]);
            
            // Last layer gets softmax, others get ReLU
            if (i == layer_sizes.size() - 2) {
                layer.activation = "softmax";
            } else {
                layer.activation = "relu";
            }
            
            layers_.push_back(layer);
        }
    }
    
    // Forward pass
    VectorXd forward(const VectorXd& input) const {
        VectorXd x = input;
        
        for (const auto& layer : layers_) {
            x = layer.W * x + layer.b;
            
            if (layer.activation == "relu") {
                x = x.cwiseMax(0.0);
            } else if (layer.activation == "softmax") {
                x = softmax(x);
            }
        }
        
        return x;
    }
    
    // Get layer weights
    const std::vector<LayerWeights>& get_layers() const {
        return layers_;
    }
    
    // Set specific layer weights (for testing)
    void set_layer(size_t idx, const MatrixXd& W, const VectorXd& b, const std::string& activation = "relu") {
        if (idx >= layers_.size()) {
            layers_.resize(idx + 1);
        }
        layers_[idx].W = W;
        layers_[idx].b = b;
        layers_[idx].activation = activation;
    }
    
    // Compute accuracy on dataset
    double compute_accuracy(const MNISTDataset& dataset) const {
        int correct = 0;
        
        for (size_t i = 0; i < dataset.size(); ++i) {
            auto sample = dataset.get_sample(i);
            VectorXd output = forward(sample.image);
            
            int pred_label;
            output.maxCoeff(&pred_label);
            
            if (pred_label == sample.label) {
                ++correct;
            }
        }
        
        return static_cast<double>(correct) / dataset.size();
    }
    
    // Simple gradient descent training (for demonstration)
    void train_simple(
        const MNISTDataset& train_data,
        int num_epochs = 10,
        double learning_rate = 0.01,
        int batch_size = 32) {
        
        for (int epoch = 0; epoch < num_epochs; ++epoch) {
            double total_loss = 0.0;
            
            for (size_t i = 0; i < train_data.size(); i += batch_size) {
                auto batch = train_data.get_batch(i, batch_size);
                
                // Compute gradients and update (simplified)
                for (const auto& sample : batch) {
                    VectorXd output = forward(sample.image);
                    
                    // Cross-entropy loss gradient
                    VectorXd target = VectorXd::Zero(output.size());
                    target[sample.label] = 1.0;
                    
                    VectorXd error = output - target;
                    total_loss += -std::log(output[sample.label] + 1e-8);
                    
                    // Backward pass (simplified - only updates last layer)
                    if (!layers_.empty()) {
                        VectorXd prev_activation = layers_.size() > 1 ? 
                            forward_to_layer(sample.image, layers_.size() - 1) : sample.image;
                        
                        layers_.back().W -= learning_rate * error * prev_activation.transpose();
                        layers_.back().b -= learning_rate * error;
                    }
                }
            }
            
            std::cout << "Epoch " << (epoch + 1) << "/" << num_epochs 
                      << " - Loss: " << (total_loss / train_data.size()) << std::endl;
        }
    }
    
private:
    std::vector<LayerWeights> layers_;
    
    VectorXd softmax(const VectorXd& x) const {
        VectorXd exp_x = (x.array() - x.maxCoeff()).exp();
        return exp_x / exp_x.sum();
    }
    
    VectorXd forward_to_layer(const VectorXd& input, size_t target_layer) const {
        VectorXd x = input;
        
        for (size_t i = 0; i < target_layer && i < layers_.size(); ++i) {
            x = layers_[i].W * x + layers_[i].b;
            
            if (layers_[i].activation == "relu") {
                x = x.cwiseMax(0.0);
            } else if (layers_[i].activation == "softmax") {
                x = softmax(x);
            }
        }
        
        return x;
    }
};

} // namespace certified
} // namespace hnf
