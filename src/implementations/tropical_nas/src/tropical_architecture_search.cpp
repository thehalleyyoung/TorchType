#include "tropical_architecture_search.hpp"
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <sstream>

namespace tropical {

// ============================================================================
// ArchitectureSpec Implementation
// ============================================================================

std::string ArchitectureSpec::to_string() const {
    std::ostringstream oss;
    oss << "[" << input_dim;
    for (int h : hidden_dims) {
        oss << " → " << h;
    }
    oss << " → " << output_dim << "]";
    oss << " (" << total_parameters << " params)";
    return oss.str();
}

// ============================================================================
// SearchConstraints Implementation
// ============================================================================

bool SearchConstraints::satisfies(const ArchitectureSpec& arch) const {
    int num_layers = arch.hidden_dims.size() + 1;  // +1 for output layer
    
    if (num_layers < min_layers || num_layers > max_layers) {
        return false;
    }
    
    for (int width : arch.hidden_dims) {
        if (width < min_width || width > max_width) {
            return false;
        }
    }
    
    if (arch.total_parameters < min_parameters || 
        arch.total_parameters > max_parameters) {
        return false;
    }
    
    return true;
}

// ============================================================================
// SearchResult Implementation
// ============================================================================

void SearchResult::print() const {
    std::cout << "Architecture: " << architecture.to_string() << "\n";
    std::cout << "Objective value: " << std::fixed << std::setprecision(4) 
              << objective_value << "\n";
    complexity.print();
}

// ============================================================================
// RandomSearch Implementation
// ============================================================================

ArchitectureSpec RandomSearch::generate_random_architecture(int input_dim, int output_dim) {
    // Random number of layers
    std::uniform_int_distribution<int> layer_dist(
        constraints_.min_layers - 1, constraints_.max_layers - 1);
    int num_hidden = layer_dist(rng_);
    
    std::vector<int> hidden_dims;
    
    // Random width for each layer
    std::uniform_int_distribution<int> width_dist(
        constraints_.min_width, constraints_.max_width);
    
    for (int i = 0; i < num_hidden; ++i) {
        hidden_dims.push_back(width_dist(rng_));
    }
    
    return ArchitectureSpec(input_dim, output_dim, hidden_dims);
}

std::vector<SearchResult> RandomSearch::search(int input_dim, int output_dim,
                                                int num_iterations) {
    std::vector<SearchResult> results;
    
    std::cout << "Starting random search with " << num_iterations 
              << " iterations...\n";
    
    SearchResult best_result;
    best_result.objective_value = -std::numeric_limits<double>::infinity();
    
    for (int iter = 0; iter < num_iterations; ++iter) {
        // Generate random architecture
        ArchitectureSpec arch = generate_random_architecture(input_dim, output_dim);
        
        // Check constraints
        if (!constraints_.satisfies(arch)) {
            continue;
        }
        
        // Create network and compute complexity
        ReLUNetwork network(input_dim);
        
        // Initialize random weights for complexity analysis
        int prev_dim = input_dim;
        for (int hidden_dim : arch.hidden_dims) {
            auto weights = torch::randn({hidden_dim, prev_dim}) * 0.1;
            auto biases = torch::randn({hidden_dim}) * 0.1;
            network.add_layer(weights, biases);
            prev_dim = hidden_dim;
        }
        
        // Output layer
        auto out_weights = torch::randn({output_dim, prev_dim}) * 0.1;
        auto out_biases = torch::randn({output_dim}) * 0.1;
        network.add_layer(out_weights, out_biases);
        
        // Compute complexity (don't use exact for large networks)
        bool use_exact = (network.num_parameters() < 100);
        NetworkComplexity complexity = compute_network_complexity(network, use_exact);
        
        // Evaluate objective
        double obj_value = objective_->evaluate(arch, complexity);
        
        SearchResult result;
        result.architecture = arch;
        result.complexity = complexity;
        result.objective_value = obj_value;
        result.search_iteration = iter;
        
        results.push_back(result);
        
        if (obj_value > best_result.objective_value) {
            best_result = result;
            std::cout << "Iteration " << iter << ": New best architecture found!\n";
            std::cout << "  " << arch.to_string() << "\n";
            std::cout << "  Objective: " << obj_value << "\n";
        }
        
        if ((iter + 1) % 10 == 0) {
            std::cout << "Progress: " << (iter + 1) << "/" << num_iterations << "\n";
        }
    }
    
    // Sort results by objective value
    std::sort(results.begin(), results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.objective_value > b.objective_value;
        });
    
    return results;
}

// ============================================================================
// EvolutionarySearch Implementation
// ============================================================================

ArchitectureSpec EvolutionarySearch::mutate_add_layer(const ArchitectureSpec& arch) {
    auto hidden = arch.hidden_dims;
    
    std::uniform_int_distribution<int> pos_dist(0, hidden.size());
    int pos = pos_dist(rng_);
    
    std::uniform_int_distribution<int> width_dist(
        constraints_.min_width, constraints_.max_width);
    int new_width = width_dist(rng_);
    
    hidden.insert(hidden.begin() + pos, new_width);
    
    return ArchitectureSpec(arch.input_dim, arch.output_dim, hidden);
}

ArchitectureSpec EvolutionarySearch::mutate_remove_layer(const ArchitectureSpec& arch) {
    if (arch.hidden_dims.empty()) {
        return arch;  // Can't remove if no hidden layers
    }
    
    auto hidden = arch.hidden_dims;
    std::uniform_int_distribution<int> pos_dist(0, hidden.size() - 1);
    int pos = pos_dist(rng_);
    
    hidden.erase(hidden.begin() + pos);
    
    return ArchitectureSpec(arch.input_dim, arch.output_dim, hidden);
}

ArchitectureSpec EvolutionarySearch::mutate_widen_layer(const ArchitectureSpec& arch) {
    if (arch.hidden_dims.empty()) {
        return arch;
    }
    
    auto hidden = arch.hidden_dims;
    std::uniform_int_distribution<int> pos_dist(0, hidden.size() - 1);
    int pos = pos_dist(rng_);
    
    hidden[pos] = std::min(hidden[pos] + 8, constraints_.max_width);
    
    return ArchitectureSpec(arch.input_dim, arch.output_dim, hidden);
}

ArchitectureSpec EvolutionarySearch::mutate_narrow_layer(const ArchitectureSpec& arch) {
    if (arch.hidden_dims.empty()) {
        return arch;
    }
    
    auto hidden = arch.hidden_dims;
    std::uniform_int_distribution<int> pos_dist(0, hidden.size() - 1);
    int pos = pos_dist(rng_);
    
    hidden[pos] = std::max(hidden[pos] - 8, constraints_.min_width);
    
    return ArchitectureSpec(arch.input_dim, arch.output_dim, hidden);
}

ArchitectureSpec EvolutionarySearch::crossover(const ArchitectureSpec& parent1,
                                                 const ArchitectureSpec& parent2) {
    // Single-point crossover of hidden layer dimensions
    if (parent1.hidden_dims.empty() || parent2.hidden_dims.empty()) {
        return parent1;  // Fallback
    }
    
    std::uniform_int_distribution<int> point_dist(
        0, std::min(parent1.hidden_dims.size(), parent2.hidden_dims.size()) - 1);
    int crossover_point = point_dist(rng_);
    
    std::vector<int> child_hidden;
    for (int i = 0; i < crossover_point; ++i) {
        child_hidden.push_back(parent1.hidden_dims[i]);
    }
    for (size_t i = crossover_point; i < parent2.hidden_dims.size(); ++i) {
        child_hidden.push_back(parent2.hidden_dims[i]);
    }
    
    return ArchitectureSpec(parent1.input_dim, parent1.output_dim, child_hidden);
}

std::vector<SearchResult> EvolutionarySearch::tournament_selection(int tournament_size) {
    std::vector<SearchResult> selected;
    
    std::uniform_int_distribution<int> pop_dist(0, population_.size() - 1);
    
    for (int i = 0; i < population_size_; ++i) {
        // Tournament: pick best from random subset
        SearchResult best = population_[pop_dist(rng_)];
        
        for (int j = 1; j < tournament_size; ++j) {
            const SearchResult& candidate = population_[pop_dist(rng_)];
            if (candidate.objective_value > best.objective_value) {
                best = candidate;
            }
        }
        
        selected.push_back(best);
    }
    
    return selected;
}

std::vector<SearchResult> EvolutionarySearch::search(int input_dim, int output_dim,
                                                       int num_generations) {
    std::cout << "Initializing population of " << population_size_ << "...\n";
    
    // Initialize population
    population_.clear();
    for (int i = 0; i < population_size_; ++i) {
        // Random initialization
        std::uniform_int_distribution<int> layer_dist(
            constraints_.min_layers - 1, constraints_.max_layers - 1);
        int num_hidden = layer_dist(rng_);
        
        std::vector<int> hidden_dims;
        std::uniform_int_distribution<int> width_dist(
            constraints_.min_width, constraints_.max_width);
        for (int j = 0; j < num_hidden; ++j) {
            hidden_dims.push_back(width_dist(rng_));
        }
        
        ArchitectureSpec arch(input_dim, output_dim, hidden_dims);
        
        // Create network and evaluate
        ReLUNetwork network(input_dim);
        int prev_dim = input_dim;
        for (int h : hidden_dims) {
            network.add_layer(torch::randn({h, prev_dim}) * 0.1,
                             torch::randn({h}) * 0.1);
            prev_dim = h;
        }
        network.add_layer(torch::randn({output_dim, prev_dim}) * 0.1,
                         torch::randn({output_dim}) * 0.1);
        
        bool use_exact = (network.num_parameters() < 100);
        NetworkComplexity complexity = compute_network_complexity(network, use_exact);
        
        SearchResult result;
        result.architecture = arch;
        result.complexity = complexity;
        result.objective_value = objective_->evaluate(arch, complexity);
        result.search_iteration = 0;
        
        population_.push_back(result);
    }
    
    // Evolution loop
    for (int gen = 0; gen < num_generations; ++gen) {
        std::cout << "\nGeneration " << (gen + 1) << "/" << num_generations << "\n";
        
        // Selection
        auto parents = tournament_selection(3);
        
        // Create offspring
        std::vector<SearchResult> offspring;
        
        for (size_t i = 0; i < parents.size(); i += 2) {
            // Crossover
            ArchitectureSpec child;
            if (i + 1 < parents.size()) {
                child = crossover(parents[i].architecture, parents[i+1].architecture);
            } else {
                child = parents[i].architecture;
            }
            
            // Mutation
            std::uniform_real_distribution<double> mut_prob(0.0, 1.0);
            if (mut_prob(rng_) < 0.3) {
                std::uniform_int_distribution<int> mut_type(0, 3);
                switch (mut_type(rng_)) {
                    case 0: child = mutate_add_layer(child); break;
                    case 1: child = mutate_remove_layer(child); break;
                    case 2: child = mutate_widen_layer(child); break;
                    case 3: child = mutate_narrow_layer(child); break;
                }
            }
            
            // Evaluate
            if (constraints_.satisfies(child)) {
                ReLUNetwork network(input_dim);
                int prev_dim = input_dim;
                for (int h : child.hidden_dims) {
                    network.add_layer(torch::randn({h, prev_dim}) * 0.1,
                                     torch::randn({h}) * 0.1);
                    prev_dim = h;
                }
                network.add_layer(torch::randn({output_dim, prev_dim}) * 0.1,
                                 torch::randn({output_dim}) * 0.1);
                
                bool use_exact = (network.num_parameters() < 100);
                NetworkComplexity complexity = compute_network_complexity(network, use_exact);
                
                SearchResult result;
                result.architecture = child;
                result.complexity = complexity;
                result.objective_value = objective_->evaluate(child, complexity);
                result.search_iteration = gen + 1;
                
                offspring.push_back(result);
            }
        }
        
        // Combine parents and offspring, select best
        std::vector<SearchResult> combined = population_;
        combined.insert(combined.end(), offspring.begin(), offspring.end());
        
        std::sort(combined.begin(), combined.end(),
            [](const SearchResult& a, const SearchResult& b) {
                return a.objective_value > b.objective_value;
            });
        
        population_.assign(combined.begin(), combined.begin() + population_size_);
        
        // Report best
        std::cout << "Best: " << population_[0].architecture.to_string() << "\n";
        std::cout << "Objective: " << population_[0].objective_value << "\n";
    }
    
    return population_;
}

// ============================================================================
// ArchitectureEvaluator Implementation
// ============================================================================

ReLUNetwork ArchitectureEvaluator::create_network(const ArchitectureSpec& spec) {
    ReLUNetwork network(spec.input_dim);
    
    int prev_dim = spec.input_dim;
    for (int hidden_dim : spec.hidden_dims) {
        // Xavier initialization
        float std = std::sqrt(2.0f / (prev_dim + hidden_dim));
        auto weights = torch::randn({hidden_dim, prev_dim}) * std;
        auto biases = torch::zeros({hidden_dim});
        network.add_layer(weights, biases);
        prev_dim = hidden_dim;
    }
    
    // Output layer
    float std = std::sqrt(2.0f / (prev_dim + spec.output_dim));
    auto out_weights = torch::randn({spec.output_dim, prev_dim}) * std;
    auto out_biases = torch::zeros({spec.output_dim});
    network.add_layer(out_weights, out_biases);
    
    return network;
}

void EvaluationResult::print() const {
    std::cout << "Evaluation Results:\n";
    std::cout << "  Train accuracy: " << (train_accuracy * 100) << "%\n";
    std::cout << "  Test accuracy: " << (test_accuracy * 100) << "%\n";
    std::cout << "  Train loss: " << train_loss << "\n";
    std::cout << "  Test loss: " << test_loss << "\n";
    std::cout << "  Epochs trained: " << epochs_trained << "\n";
    std::cout << "  Tropical efficiency: " << tropical_efficiency << "\n";
}

// ============================================================================
// Grid Search (simplified for small spaces)
// ============================================================================

void GridSearch::enumerate_architectures_recursive(
    std::vector<int>& current_layers,
    int input_dim,
    int output_dim,
    int remaining_layers,
    std::vector<ArchitectureSpec>& results) {
    
    if (remaining_layers == 0) {
        ArchitectureSpec arch(input_dim, output_dim, current_layers);
        if (constraints_.satisfies(arch)) {
            results.push_back(arch);
        }
        return;
    }
    
    // Try different widths for this layer
    for (int width = constraints_.min_width; 
         width <= constraints_.max_width; 
         width += 8) {  // Step by 8 to reduce search space
        
        current_layers.push_back(width);
        enumerate_architectures_recursive(current_layers, input_dim, output_dim,
                                         remaining_layers - 1, results);
        current_layers.pop_back();
    }
}

std::vector<SearchResult> GridSearch::search(int input_dim, int output_dim) {
    std::vector<SearchResult> results;
    
    // Enumerate all architectures within constraints
    for (int num_layers = constraints_.min_layers - 1; 
         num_layers <= constraints_.max_layers - 1; 
         ++num_layers) {
        
        std::vector<ArchitectureSpec> architectures;
        std::vector<int> current_layers;
        enumerate_architectures_recursive(current_layers, input_dim, output_dim,
                                         num_layers, architectures);
        
        std::cout << "Evaluating " << architectures.size() 
                  << " architectures with " << num_layers << " hidden layers...\n";
        
        for (const auto& arch : architectures) {
            // Create and evaluate
            ReLUNetwork network(input_dim);
            int prev_dim = input_dim;
            for (int h : arch.hidden_dims) {
                network.add_layer(torch::randn({h, prev_dim}) * 0.1,
                                 torch::randn({h}) * 0.1);
                prev_dim = h;
            }
            network.add_layer(torch::randn({output_dim, prev_dim}) * 0.1,
                             torch::randn({output_dim}) * 0.1);
            
            bool use_exact = (network.num_parameters() < 100);
            NetworkComplexity complexity = compute_network_complexity(network, use_exact);
            
            SearchResult result;
            result.architecture = arch;
            result.complexity = complexity;
            result.objective_value = objective_->evaluate(arch, complexity);
            result.search_iteration = results.size();
            
            results.push_back(result);
        }
    }
    
    // Sort by objective
    std::sort(results.begin(), results.end(),
        [](const SearchResult& a, const SearchResult& b) {
            return a.objective_value > b.objective_value;
        });
    
    return results;
}

} // namespace tropical
