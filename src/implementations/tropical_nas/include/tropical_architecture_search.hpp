#ifndef TROPICAL_ARCHITECTURE_SEARCH_HPP
#define TROPICAL_ARCHITECTURE_SEARCH_HPP

#include "relu_to_tropical.hpp"
#include "tropical_arithmetic.hpp"
#include <vector>
#include <functional>
#include <random>

namespace tropical {

// Architecture specification
struct ArchitectureSpec {
    int input_dim;
    int output_dim;
    std::vector<int> hidden_dims;  // dimensions of hidden layers
    int total_parameters;
    
    ArchitectureSpec(int in, int out, const std::vector<int>& hidden)
        : input_dim(in), output_dim(out), hidden_dims(hidden) {
        total_parameters = 0;
        int prev_dim = in;
        for (int h : hidden) {
            total_parameters += prev_dim * h + h;  // weights + biases
            prev_dim = h;
        }
        total_parameters += prev_dim * out + out;
    }
    
    std::string to_string() const;
};

// Search objective: maximize linear regions per parameter
class TropicalNASObjective {
public:
    virtual ~TropicalNASObjective() = default;
    
    // Evaluate architecture (higher is better)
    virtual double evaluate(const ArchitectureSpec& arch, 
                           const NetworkComplexity& complexity) const = 0;
    
    // Name of the objective
    virtual std::string name() const = 0;
};

// Standard objectives
class RegionsPerParameterObjective : public TropicalNASObjective {
public:
    double evaluate(const ArchitectureSpec& arch, 
                   const NetworkComplexity& complexity) const override {
        return complexity.efficiency_ratio;
    }
    
    std::string name() const override { return "Regions/Parameters"; }
};

class PolytopeVolumeObjective : public TropicalNASObjective {
public:
    double evaluate(const ArchitectureSpec& arch,
                   const NetworkComplexity& complexity) const override {
        // Normalize by parameter count to make comparable
        return complexity.polytope_volume / std::sqrt(arch.total_parameters);
    }
    
    std::string name() const override { return "Polytope Volume (normalized)"; }
};

class CapacityObjective : public TropicalNASObjective {
public:
    double evaluate(const ArchitectureSpec& arch,
                   const NetworkComplexity& complexity) const override {
        // Pure capacity (number of regions)
        return static_cast<double>(complexity.num_linear_regions_approx);
    }
    
    std::string name() const override { return "Total Capacity (regions)"; }
};

// Search space constraints
struct SearchConstraints {
    int min_layers = 1;
    int max_layers = 5;
    int min_width = 4;
    int max_width = 128;
    int max_parameters = 10000;
    int min_parameters = 10;
    
    // Check if architecture satisfies constraints
    bool satisfies(const ArchitectureSpec& arch) const;
};

// Search result
struct SearchResult {
    ArchitectureSpec architecture;
    NetworkComplexity complexity;
    double objective_value;
    int search_iteration;
    
    void print() const;
};

// Random search strategy
class RandomSearch {
private:
    SearchConstraints constraints_;
    std::shared_ptr<TropicalNASObjective> objective_;
    std::mt19937 rng_;
    
    ArchitectureSpec generate_random_architecture(int input_dim, int output_dim);
    
public:
    RandomSearch(const SearchConstraints& constraints,
                 std::shared_ptr<TropicalNASObjective> objective,
                 unsigned seed = 42)
        : constraints_(constraints), objective_(objective), rng_(seed) {}
    
    std::vector<SearchResult> search(int input_dim, int output_dim, 
                                      int num_iterations = 100);
};

// Evolutionary search strategy
class EvolutionarySearch {
private:
    SearchConstraints constraints_;
    std::shared_ptr<TropicalNASObjective> objective_;
    std::mt19937 rng_;
    
    // Population of architectures
    std::vector<SearchResult> population_;
    int population_size_;
    
    // Mutation operators
    ArchitectureSpec mutate_add_layer(const ArchitectureSpec& arch);
    ArchitectureSpec mutate_remove_layer(const ArchitectureSpec& arch);
    ArchitectureSpec mutate_widen_layer(const ArchitectureSpec& arch);
    ArchitectureSpec mutate_narrow_layer(const ArchitectureSpec& arch);
    
    // Crossover
    ArchitectureSpec crossover(const ArchitectureSpec& parent1,
                                const ArchitectureSpec& parent2);
    
    // Selection
    std::vector<SearchResult> tournament_selection(int tournament_size);
    
public:
    EvolutionarySearch(const SearchConstraints& constraints,
                       std::shared_ptr<TropicalNASObjective> objective,
                       int population_size = 20,
                       unsigned seed = 42)
        : constraints_(constraints), objective_(objective),
          rng_(seed), population_size_(population_size) {}
    
    std::vector<SearchResult> search(int input_dim, int output_dim,
                                      int num_generations = 50);
    
    const std::vector<SearchResult>& get_population() const { return population_; }
};

// Grid search (exhaustive within limits)
class GridSearch {
private:
    SearchConstraints constraints_;
    std::shared_ptr<TropicalNASObjective> objective_;
    
    void enumerate_architectures_recursive(
        std::vector<int>& current_layers,
        int input_dim,
        int output_dim,
        int remaining_layers,
        std::vector<ArchitectureSpec>& results);
    
public:
    GridSearch(const SearchConstraints& constraints,
               std::shared_ptr<TropicalNASObjective> objective)
        : constraints_(constraints), objective_(objective) {}
    
    std::vector<SearchResult> search(int input_dim, int output_dim);
};

// Evaluate a trained network on actual data
struct EvaluationResult {
    double train_accuracy;
    double test_accuracy;
    double train_loss;
    double test_loss;
    int epochs_trained;
    double tropical_efficiency;  // from complexity analysis
    
    void print() const;
};

// Train and evaluate architecture on dataset
class ArchitectureEvaluator {
private:
    torch::DeviceType device_;
    int max_epochs_;
    double learning_rate_;
    
public:
    ArchitectureEvaluator(torch::DeviceType device = torch::kCPU,
                          int max_epochs = 50,
                          double lr = 0.01)
        : device_(device), max_epochs_(max_epochs), learning_rate_(lr) {}
    
    // Create network from spec and initialize randomly
    ReLUNetwork create_network(const ArchitectureSpec& spec);
    
    // Train on dataset
    EvaluationResult train_and_evaluate(
        const ArchitectureSpec& spec,
        const torch::Tensor& train_data,
        const torch::Tensor& train_labels,
        const torch::Tensor& test_data,
        const torch::Tensor& test_labels);
};

// Complete NAS experiment: search + train + evaluate
class TropicalNASExperiment {
private:
    SearchConstraints constraints_;
    std::shared_ptr<TropicalNASObjective> objective_;
    ArchitectureEvaluator evaluator_;
    
public:
    TropicalNASExperiment(const SearchConstraints& constraints,
                          std::shared_ptr<TropicalNASObjective> objective,
                          const ArchitectureEvaluator& evaluator)
        : constraints_(constraints), objective_(objective), evaluator_(evaluator) {}
    
    // Run full experiment: search, train top-k, evaluate
    struct ExperimentResult {
        std::vector<SearchResult> search_results;
        std::vector<EvaluationResult> eval_results;
        int best_architecture_idx;
        
        void print_summary() const;
    };
    
    ExperimentResult run(int input_dim, int output_dim,
                         const torch::Tensor& train_data,
                         const torch::Tensor& train_labels,
                         const torch::Tensor& test_data,
                         const torch::Tensor& test_labels,
                         int num_search_iterations = 100,
                         int top_k_to_train = 5);
};

} // namespace tropical

#endif // TROPICAL_ARCHITECTURE_SEARCH_HPP
