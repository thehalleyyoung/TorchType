#include "tropical_architecture_search.hpp"
#include <torch/torch.h>
#include <iostream>
#include <fstream>
#include <vector>

using namespace tropical;

// Simple MNIST data loader
class MNISTDataset {
private:
    torch::Tensor images;
    torch::Tensor labels;
    
    uint32_t read_int(std::ifstream& file) {
        uint32_t val = 0;
        file.read(reinterpret_cast<char*>(&val), 4);
        return __builtin_bswap32(val);  // Convert from big-endian
    }
    
public:
    bool load(const std::string& image_path, const std::string& label_path, int max_samples = -1) {
        // Load images
        std::ifstream image_file(image_path, std::ios::binary);
        if (!image_file) {
            std::cerr << "Cannot open " << image_path << std::endl;
            return false;
        }
        
        uint32_t magic = read_int(image_file);
        uint32_t num_images = read_int(image_file);
        uint32_t rows = read_int(image_file);
        uint32_t cols = read_int(image_file);
        
        if (magic != 2051) {
            std::cerr << "Invalid MNIST image file magic number\n";
            return false;
        }
        
        if (max_samples > 0 && static_cast<uint32_t>(max_samples) < num_images) {
            num_images = max_samples;
        }
        
        std::vector<uint8_t> buffer(num_images * rows * cols);
        image_file.read(reinterpret_cast<char*>(buffer.data()), buffer.size());
        
        images = torch::zeros({static_cast<int>(num_images), static_cast<int>(rows * cols)});
        auto images_acc = images.accessor<float, 2>();
        
        for (size_t i = 0; i < num_images; ++i) {
            for (size_t j = 0; j < rows * cols; ++j) {
                images_acc[i][j] = buffer[i * rows * cols + j] / 255.0f;
            }
        }
        
        // Load labels
        std::ifstream label_file(label_path, std::ios::binary);
        if (!label_file) {
            std::cerr << "Cannot open " << label_path << std::endl;
            return false;
        }
        
        magic = read_int(label_file);
        uint32_t num_labels = read_int(label_file);
        
        if (magic != 2049) {
            std::cerr << "Invalid MNIST label file magic number\n";
            return false;
        }
        
        if (max_samples > 0 && static_cast<uint32_t>(max_samples) < num_labels) {
            num_labels = max_samples;
        }
        
        std::vector<uint8_t> label_buffer(num_labels);
        label_file.read(reinterpret_cast<char*>(label_buffer.data()), label_buffer.size());
        
        labels = torch::zeros({static_cast<int>(num_labels)}, torch::kLong);
        auto labels_acc = labels.accessor<int64_t, 1>();
        
        for (size_t i = 0; i < num_labels; ++i) {
            labels_acc[i] = label_buffer[i];
        }
        
        std::cout << "Loaded " << num_images << " images of size " 
                  << rows << "x" << cols << std::endl;
        
        return true;
    }
    
    torch::Tensor get_images() const { return images; }
    torch::Tensor get_labels() const { return labels; }
    int size() const { return images.size(0); }
};

// Train a network created from an architecture spec
EvaluationResult train_network(
    ReLUNetwork& network,
    const torch::Tensor& train_data,
    const torch::Tensor& train_labels,
    const torch::Tensor& test_data,
    const torch::Tensor& test_labels,
    int max_epochs = 20,
    double lr = 0.01) {
    
    std::cout << "Training network with " << network.num_parameters() << " parameters...\n";
    
    // Create PyTorch module from ReLU network
    struct Net : torch::nn::Module {
        std::vector<torch::nn::Linear> layers;
        
        Net(const ReLUNetwork& net) {
            const auto& relu_layers = net.layers();
            for (size_t i = 0; i < relu_layers.size(); ++i) {
                auto layer = torch::nn::Linear(
                    relu_layers[i].weights.size(1),
                    relu_layers[i].weights.size(0)
                );
                layer->weight = relu_layers[i].weights.clone();
                layer->bias = relu_layers[i].biases.clone();
                layers.push_back(register_module("fc" + std::to_string(i), layer));
            }
        }
        
        torch::Tensor forward(torch::Tensor x) {
            for (size_t i = 0; i < layers.size() - 1; ++i) {
                x = torch::relu(layers[i]->forward(x));
            }
            x = layers.back()->forward(x);  // No ReLU on last layer
            return x;
        }
    };
    
    Net model(network);
    torch::optim::SGD optimizer(model.parameters(), torch::optim::SGDOptions(lr).momentum(0.9));
    
    int batch_size = 128;
    int num_batches = train_data.size(0) / batch_size;
    
    EvaluationResult result;
    
    for (int epoch = 0; epoch < max_epochs; ++epoch) {
        model.train();
        float epoch_loss = 0.0;
        int correct = 0;
        int total = 0;
        
        // Training
        for (int batch = 0; batch < num_batches; ++batch) {
            int start = batch * batch_size;
            int end = std::min(start + batch_size, static_cast<int>(train_data.size(0)));
            
            auto batch_data = train_data.slice(0, start, end);
            auto batch_labels = train_labels.slice(0, start, end);
            
            optimizer.zero_grad();
            auto output = model.forward(batch_data);
            auto loss = torch::cross_entropy_loss(output, batch_labels);
            loss.backward();
            optimizer.step();
            
            epoch_loss += loss.item<float>();
            
            auto pred = output.argmax(1);
            correct += pred.eq(batch_labels).sum().item<int>();
            total += batch_labels.size(0);
        }
        
        float train_acc = static_cast<float>(correct) / total;
        
        // Evaluation
        model.eval();
        torch::NoGradGuard no_grad;
        
        auto test_output = model.forward(test_data);
        auto test_loss = torch::cross_entropy_loss(test_output, test_labels);
        auto test_pred = test_output.argmax(1);
        int test_correct = test_pred.eq(test_labels).sum().item<int>();
        float test_acc = static_cast<float>(test_correct) / test_labels.size(0);
        
        std::cout << "Epoch " << (epoch + 1) << "/" << max_epochs 
                  << " - Loss: " << (epoch_loss / num_batches)
                  << " - Train Acc: " << (train_acc * 100) << "%"
                  << " - Test Acc: " << (test_acc * 100) << "%\n";
        
        result.train_accuracy = train_acc;
        result.test_accuracy = test_acc;
        result.train_loss = epoch_loss / num_batches;
        result.test_loss = test_loss.item<float>();
    }
    
    result.epochs_trained = max_epochs;
    
    return result;
}

int main(int argc, char* argv[]) {
    std::cout << "====================================================\n";
    std::cout << "TROPICAL GEOMETRY NAS - MNIST DEMONSTRATION\n";
    std::cout << "====================================================\n\n";
    
    // Check for data path argument
    std::string data_path = "./data/MNIST/raw";
    if (argc > 1) {
        data_path = argv[1];
    }
    
    // Load MNIST data
    std::cout << "Loading MNIST data from " << data_path << "...\n";
    
    MNISTDataset train_dataset, test_dataset;
    
    if (!train_dataset.load(data_path + "/train-images-idx3-ubyte",
                            data_path + "/train-labels-idx1-ubyte",
                            10000)) {  // Use subset for faster demo
        std::cerr << "Failed to load training data\n";
        std::cerr << "Usage: " << argv[0] << " <path_to_MNIST_data>\n";
        std::cerr << "Download MNIST from http://yann.lecun.com/exdb/mnist/\n";
        return 1;
    }
    
    if (!test_dataset.load(data_path + "/t10k-images-idx3-ubyte",
                           data_path + "/t10k-labels-idx1-ubyte",
                           2000)) {
        std::cerr << "Failed to load test data\n";
        return 1;
    }
    
    auto train_data = train_dataset.get_images();
    auto train_labels = train_dataset.get_labels();
    auto test_data = test_dataset.get_images();
    auto test_labels = test_dataset.get_labels();
    
    std::cout << "Train set: " << train_data.size(0) << " samples\n";
    std::cout << "Test set: " << test_data.size(0) << " samples\n\n";
    
    // Set up search
    SearchConstraints constraints;
    constraints.min_layers = 1;
    constraints.max_layers = 3;
    constraints.min_width = 16;
    constraints.max_width = 128;
    constraints.max_parameters = 10000;
    constraints.min_parameters = 100;
    
    std::cout << "=== Phase 1: Tropical Geometry Architecture Search ===\n\n";
    
    auto objective = std::make_shared<RegionsPerParameterObjective>();
    EvolutionarySearch search(constraints, objective, 15, 42);
    
    int input_dim = 784;  // 28x28 flattened
    int output_dim = 10;  // 10 digits
    
    auto search_results = search.search(input_dim, output_dim, 10);
    
    std::cout << "\n=== Top 5 Architectures by Tropical Complexity ===\n\n";
    
    int top_k = std::min(5, static_cast<int>(search_results.size()));
    for (int i = 0; i < top_k; ++i) {
        std::cout << (i + 1) << ". " << search_results[i].architecture.to_string() << "\n";
        std::cout << "   Efficiency: " << search_results[i].complexity.efficiency_ratio << "\n";
        std::cout << "   Linear regions: ~" << search_results[i].complexity.num_linear_regions_approx << "\n\n";
    }
    
    // Train top 3 architectures
    std::cout << "\n=== Phase 2: Training Top 3 Architectures ===\n\n";
    
    std::vector<std::pair<SearchResult, EvaluationResult>> final_results;
    
    for (int i = 0; i < std::min(3, top_k); ++i) {
        std::cout << "\n--- Training Architecture " << (i + 1) << " ---\n";
        std::cout << search_results[i].architecture.to_string() << "\n\n";
        
        // Create network
        ArchitectureEvaluator evaluator;
        ReLUNetwork network = evaluator.create_network(search_results[i].architecture);
        
        // Train
        auto eval_result = train_network(network, train_data, train_labels,
                                        test_data, test_labels, 15, 0.01);
        
        eval_result.tropical_efficiency = search_results[i].complexity.efficiency_ratio;
        
        final_results.push_back({search_results[i], eval_result});
    }
    
    // Summary
    std::cout << "\n\n====================================================\n";
    std::cout << "FINAL RESULTS SUMMARY\n";
    std::cout << "====================================================\n\n";
    
    for (size_t i = 0; i < final_results.size(); ++i) {
        std::cout << "Architecture " << (i + 1) << ": " 
                  << final_results[i].first.architecture.to_string() << "\n";
        std::cout << "  Test Accuracy: " << (final_results[i].second.test_accuracy * 100) << "%\n";
        std::cout << "  Parameters: " << final_results[i].first.architecture.total_parameters << "\n";
        std::cout << "  Tropical Efficiency: " << final_results[i].second.tropical_efficiency << "\n";
        std::cout << "  Linear Regions: ~" << final_results[i].first.complexity.num_linear_regions_approx << "\n\n";
    }
    
    // Find best by accuracy per parameter
    size_t best_idx = 0;
    double best_ratio = final_results[0].second.test_accuracy / 
                        final_results[0].first.architecture.total_parameters;
    
    for (size_t i = 1; i < final_results.size(); ++i) {
        double ratio = final_results[i].second.test_accuracy /
                       final_results[i].first.architecture.total_parameters;
        if (ratio > best_ratio) {
            best_ratio = ratio;
            best_idx = i;
        }
    }
    
    std::cout << "Best architecture (accuracy/parameter): #" << (best_idx + 1) << "\n";
    std::cout << final_results[best_idx].first.architecture.to_string() << "\n";
    std::cout << "Achieves " << (final_results[best_idx].second.test_accuracy * 100) 
              << "% with only " << final_results[best_idx].first.architecture.total_parameters 
              << " parameters!\n";
    
    std::cout << "\n====================================================\n";
    std::cout << "Demonstration Complete!\n";
    std::cout << "====================================================\n";
    
    return 0;
}
