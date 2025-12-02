#pragma once

#include <vector>
#include <string>
#include <fstream>
#include <iostream>
#include <cstdint>
#include <algorithm>
#include <random>

namespace hnf {
namespace rewriter {

// Simple MNIST data structure
struct MNISTData {
    std::vector<std::vector<double>> images;  // Each image is 784 doubles (28x28)
    std::vector<int> labels;  // Labels 0-9
    
    size_t size() const { return labels.size(); }
    
    void normalize() {
        for (auto& img : images) {
            for (auto& pixel : img) {
                pixel /= 255.0;  // Normalize to [0, 1]
            }
        }
    }
    
    void shuffle(unsigned int seed = 42) {
        std::mt19937 gen(seed);
        std::vector<size_t> indices(size());
        for (size_t i = 0; i < size(); ++i) indices[i] = i;
        std::shuffle(indices.begin(), indices.end(), gen);
        
        std::vector<std::vector<double>> new_images;
        std::vector<int> new_labels;
        for (size_t idx : indices) {
            new_images.push_back(images[idx]);
            new_labels.push_back(labels[idx]);
        }
        images = std::move(new_images);
        labels = std::move(new_labels);
    }
};

// MNIST data loader
class MNISTLoader {
public:
    // Load MNIST from ubyte format files
    static MNISTData load_from_files(const std::string& images_path, 
                                     const std::string& labels_path) {
        MNISTData data;
        
        // Load images
        std::ifstream img_file(images_path, std::ios::binary);
        if (!img_file.is_open()) {
            std::cerr << "Warning: Could not open " << images_path << std::endl;
            std::cerr << "Generating synthetic data instead..." << std::endl;
            return generate_synthetic_mnist(1000);
        }
        
        // Read magic number and dimensions
        uint32_t magic, num_images, rows, cols;
        img_file.read(reinterpret_cast<char*>(&magic), 4);
        img_file.read(reinterpret_cast<char*>(&num_images), 4);
        img_file.read(reinterpret_cast<char*>(&rows), 4);
        img_file.read(reinterpret_cast<char*>(&cols), 4);
        
        // Convert from big endian to little endian
        magic = __builtin_bswap32(magic);
        num_images = __builtin_bswap32(num_images);
        rows = __builtin_bswap32(rows);
        cols = __builtin_bswap32(cols);
        
        if (magic != 2051) {
            std::cerr << "Invalid MNIST image file magic number" << std::endl;
            return generate_synthetic_mnist(1000);
        }
        
        // Read images
        data.images.resize(num_images);
        for (size_t i = 0; i < num_images; ++i) {
            data.images[i].resize(rows * cols);
            for (size_t j = 0; j < rows * cols; ++j) {
                unsigned char pixel;
                img_file.read(reinterpret_cast<char*>(&pixel), 1);
                data.images[i][j] = static_cast<double>(pixel);
            }
        }
        
        img_file.close();
        
        // Load labels
        std::ifstream lbl_file(labels_path, std::ios::binary);
        if (!lbl_file.is_open()) {
            std::cerr << "Warning: Could not open " << labels_path << std::endl;
            return generate_synthetic_mnist(1000);
        }
        
        lbl_file.read(reinterpret_cast<char*>(&magic), 4);
        uint32_t num_labels;
        lbl_file.read(reinterpret_cast<char*>(&num_labels), 4);
        
        magic = __builtin_bswap32(magic);
        num_labels = __builtin_bswap32(num_labels);
        
        if (magic != 2049 || num_labels != num_images) {
            std::cerr << "Invalid MNIST label file" << std::endl;
            return generate_synthetic_mnist(1000);
        }
        
        data.labels.resize(num_labels);
        for (size_t i = 0; i < num_labels; ++i) {
            unsigned char label;
            lbl_file.read(reinterpret_cast<char*>(&label), 1);
            data.labels[i] = static_cast<int>(label);
        }
        
        lbl_file.close();
        
        data.normalize();
        return data;
    }
    
    // Generate synthetic MNIST-like data for testing
    static MNISTData generate_synthetic_mnist(size_t num_samples = 1000) {
        MNISTData data;
        std::mt19937 gen(42);
        std::normal_distribution<double> dist(0.5, 0.2);
        std::uniform_int_distribution<int> label_dist(0, 9);
        
        data.images.resize(num_samples);
        data.labels.resize(num_samples);
        
        for (size_t i = 0; i < num_samples; ++i) {
            data.images[i].resize(784);
            for (size_t j = 0; j < 784; ++j) {
                data.images[i][j] = std::max(0.0, std::min(1.0, dist(gen)));
            }
            data.labels[i] = label_dist(gen);
        }
        
        return data;
    }
    
    // Download MNIST data (requires curl)
    static bool download_mnist(const std::string& output_dir = ".") {
        const char* urls[] = {
            "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz",
            "http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz"
        };
        
        const char* filenames[] = {
            "train-images-idx3-ubyte.gz",
            "train-labels-idx1-ubyte.gz",
            "t10k-images-idx3-ubyte.gz",
            "t10k-labels-idx1-ubyte.gz"
        };
        
        std::cout << "Downloading MNIST dataset..." << std::endl;
        
        for (int i = 0; i < 4; ++i) {
            std::string cmd = "curl -L -o " + output_dir + "/" + filenames[i] + " " + urls[i];
            int ret = system(cmd.c_str());
            if (ret != 0) {
                std::cerr << "Failed to download " << filenames[i] << std::endl;
                return false;
            }
            
            // Decompress
            std::string gunzip_cmd = "gunzip -f " + output_dir + "/" + filenames[i];
            ret = system(gunzip_cmd.c_str());
            if (ret != 0) {
                std::cerr << "Failed to decompress " << filenames[i] << std::endl;
                return false;
            }
        }
        
        std::cout << "MNIST dataset downloaded successfully!" << std::endl;
        return true;
    }
};

} // namespace rewriter
} // namespace hnf
