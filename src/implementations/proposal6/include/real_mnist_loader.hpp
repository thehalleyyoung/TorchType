#pragma once

#include <Eigen/Dense>
#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <cstdint>
#include <algorithm>

namespace hnf {
namespace certified {

// Real MNIST data loader - no synthetic data!
// Downloads and parses actual MNIST binary files
class RealMNISTLoader {
public:
    struct MNISTImage {
        Eigen::VectorXd pixels;  // 784 dimensional
        int label;
    };
    
    struct MNISTDataset {
        std::vector<MNISTImage> images;
        int num_rows;
        int num_cols;
        
        Eigen::VectorXd compute_mean() const {
            if (images.empty()) throw std::runtime_error("Empty dataset");
            Eigen::VectorXd mean = Eigen::VectorXd::Zero(images[0].pixels.size());
            for (const auto& img : images) {
                mean += img.pixels;
            }
            return mean / static_cast<double>(images.size());
        }
        
        Eigen::VectorXd compute_std() const {
            Eigen::VectorXd mean = compute_mean();
            Eigen::VectorXd variance = Eigen::VectorXd::Zero(mean.size());
            for (const auto& img : images) {
                Eigen::VectorXd diff = img.pixels - mean;
                variance += diff.array().square().matrix();
            }
            variance /= static_cast<double>(images.size());
            return variance.array().sqrt().matrix();
        }
        
        std::pair<Eigen::VectorXd, Eigen::VectorXd> compute_bounds(double percentile = 99.9) const {
            int dim = images[0].pixels.size();
            Eigen::VectorXd lower = Eigen::VectorXd::Constant(dim, 1e10);
            Eigen::VectorXd upper = Eigen::VectorXd::Constant(dim, -1e10);
            
            for (const auto& img : images) {
                for (int i = 0; i < dim; ++i) {
                    lower(i) = std::min(lower(i), img.pixels(i));
                    upper(i) = std::max(upper(i), img.pixels(i));
                }
            }
            
            return {lower, upper};
        }
    };
    
private:
    // Read big-endian 32-bit integer
    static uint32_t read_int32(std::ifstream& file) {
        uint32_t value = 0;
        for (int i = 0; i < 4; ++i) {
            uint8_t byte;
            file.read(reinterpret_cast<char*>(&byte), 1);
            value = (value << 8) | byte;
        }
        return value;
    }
    
public:
    // Load MNIST images from IDX file format
    static MNISTDataset load_images(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open MNIST image file: " + filename + 
                "\nPlease download from: http://yann.lecun.com/exdb/mnist/");
        }
        
        // Read header
        uint32_t magic = read_int32(file);
        if (magic != 2051) {
            throw std::runtime_error("Invalid MNIST image file magic number");
        }
        
        uint32_t num_images = read_int32(file);
        uint32_t num_rows = read_int32(file);
        uint32_t num_cols = read_int32(file);
        
        std::cout << "Loading " << num_images << " MNIST images (" 
                  << num_rows << "x" << num_cols << ")..." << std::flush;
        
        MNISTDataset dataset;
        dataset.num_rows = num_rows;
        dataset.num_cols = num_cols;
        dataset.images.reserve(num_images);
        
        // Read images
        int pixels_per_image = num_rows * num_cols;
        std::vector<uint8_t> buffer(pixels_per_image);
        
        for (uint32_t i = 0; i < num_images; ++i) {
            file.read(reinterpret_cast<char*>(buffer.data()), pixels_per_image);
            
            MNISTImage img;
            img.pixels.resize(pixels_per_image);
            
            // Normalize to [0, 1]
            for (int j = 0; j < pixels_per_image; ++j) {
                img.pixels(j) = buffer[j] / 255.0;
            }
            
            dataset.images.push_back(img);
        }
        
        std::cout << " Done!\n";
        return dataset;
    }
    
    // Load MNIST labels from IDX file format
    static void load_labels(const std::string& filename, MNISTDataset& dataset) {
        std::ifstream file(filename, std::ios::binary);
        if (!file.is_open()) {
            throw std::runtime_error("Cannot open MNIST label file: " + filename +
                "\nPlease download from: http://yann.lecun.com/exdb/mnist/");
        }
        
        // Read header
        uint32_t magic = read_int32(file);
        if (magic != 2049) {
            throw std::runtime_error("Invalid MNIST label file magic number");
        }
        
        uint32_t num_labels = read_int32(file);
        
        if (num_labels != dataset.images.size()) {
            throw std::runtime_error("Number of labels doesn't match number of images");
        }
        
        std::cout << "Loading " << num_labels << " MNIST labels..." << std::flush;
        
        // Read labels
        for (uint32_t i = 0; i < num_labels; ++i) {
            uint8_t label;
            file.read(reinterpret_cast<char*>(&label), 1);
            dataset.images[i].label = label;
        }
        
        std::cout << " Done!\n";
    }
    
    // Download MNIST files if they don't exist
    static void ensure_mnist_downloaded(const std::string& data_dir) {
        std::cout << "Checking for MNIST data in " << data_dir << "...\n";
        
        // Check if files exist
        std::vector<std::string> required_files = {
            data_dir + "/train-images-idx3-ubyte",
            data_dir + "/train-labels-idx1-ubyte",
            data_dir + "/t10k-images-idx3-ubyte",
            data_dir + "/t10k-labels-idx1-ubyte"
        };
        
        bool all_exist = true;
        for (const auto& file : required_files) {
            std::ifstream f(file);
            if (!f.good()) {
                all_exist = false;
                std::cout << "  Missing: " << file << "\n";
            }
        }
        
        if (!all_exist) {
            std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
            std::cout << "║ MNIST Dataset Download Required                              ║\n";
            std::cout << "╠══════════════════════════════════════════════════════════════╣\n";
            std::cout << "║ Please download MNIST from:                                   ║\n";
            std::cout << "║   http://yann.lecun.com/exdb/mnist/                          ║\n";
            std::cout << "║                                                                ║\n";
            std::cout << "║ Required files (gunzip after download):                      ║\n";
            std::cout << "║   - train-images-idx3-ubyte.gz                               ║\n";
            std::cout << "║   - train-labels-idx1-ubyte.gz                               ║\n";
            std::cout << "║   - t10k-images-idx3-ubyte.gz                                ║\n";
            std::cout << "║   - t10k-labels-idx1-ubyte.gz                                ║\n";
            std::cout << "║                                                                ║\n";
            std::cout << "║ Or run:                                                       ║\n";
            std::cout << "║   cd " << data_dir << " && ./download_mnist.sh               ║\n";
            std::cout << "╚══════════════════════════════════════════════════════════════╝\n";
            
            // Create download script
            std::ofstream script(data_dir + "/download_mnist.sh");
            script << "#!/bin/bash\n";
            script << "echo 'Downloading MNIST dataset...'\n";
            script << "curl -O http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz\n";
            script << "curl -O http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz\n";
            script << "curl -O http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz\n";
            script << "curl -O http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz\n";
            script << "echo 'Decompressing...'\n";
            script << "gunzip *.gz\n";
            script << "echo 'Done!'\n";
            script.close();
            
            #ifdef __unix__
            chmod((data_dir + "/download_mnist.sh").c_str(), 0755);
            #endif
        }
    }
    
    // Load full MNIST training set
    static MNISTDataset load_training_set(const std::string& data_dir = "./data") {
        ensure_mnist_downloaded(data_dir);
        
        MNISTDataset dataset = load_images(data_dir + "/train-images-idx3-ubyte");
        load_labels(data_dir + "/train-labels-idx1-ubyte", dataset);
        
        return dataset;
    }
    
    // Load MNIST test set
    static MNISTDataset load_test_set(const std::string& data_dir = "./data") {
        ensure_mnist_downloaded(data_dir);
        
        MNISTDataset dataset = load_images(data_dir + "/t10k-images-idx3-ubyte");
        load_labels(data_dir + "/t10k-labels-idx1-ubyte", dataset);
        
        return dataset;
    }
    
    // Normalize dataset (z-score normalization)
    static void normalize_dataset(MNISTDataset& dataset) {
        Eigen::VectorXd mean = dataset.compute_mean();
        Eigen::VectorXd std = dataset.compute_std();
        
        std::cout << "Normalizing dataset (z-score)...\n";
        std::cout << "  Mean range: [" << mean.minCoeff() << ", " << mean.maxCoeff() << "]\n";
        std::cout << "  Std range: [" << std.minCoeff() << ", " << std.maxCoeff() << "]\n";
        
        for (auto& img : dataset.images) {
            img.pixels = (img.pixels - mean).array() / (std.array() + 1e-8);
        }
        
        auto bounds = dataset.compute_bounds();
        std::cout << "  Normalized range: [" << bounds.first.minCoeff() 
                  << ", " << bounds.second.maxCoeff() << "]\n";
    }
};

} // namespace certified
} // namespace hnf
