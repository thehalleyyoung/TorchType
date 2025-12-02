cmake_minimum_required(VERSION 3.14)
project(proposal9_curvature_quantization)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/curvature_quantizer.cpp
)

# Create library
add_library(curvature_quantizer STATIC ${SOURCES})
target_link_libraries(curvature_quantizer "${TORCH_LIBRARIES}")

# Tests
add_executable(test_comprehensive tests/test_comprehensive.cpp)
target_link_libraries(test_comprehensive curvature_quantizer "${TORCH_LIBRARIES}")

# Examples
add_executable(mnist_quantization_demo examples/mnist_quantization_demo.cpp)
target_link_libraries(mnist_quantization_demo curvature_quantizer "${TORCH_LIBRARIES}")

add_executable(resnet_quantization examples/resnet_quantization.cpp)
target_link_libraries(resnet_quantization curvature_quantizer "${TORCH_LIBRARIES}")

add_executable(transformer_layer_quant examples/transformer_layer_quant.cpp)
target_link_libraries(transformer_layer_quant curvature_quantizer "${TORCH_LIBRARIES}")

# New comprehensive MNIST example
add_executable(mnist_real_quantization examples/mnist_real_quantization.cpp)
target_link_libraries(mnist_real_quantization curvature_quantizer "${TORCH_LIBRARIES}")

# Advanced test: curvature beats manual tuning
add_executable(test_curvature_beats_manual examples/test_curvature_beats_manual.cpp)
target_link_libraries(test_curvature_beats_manual curvature_quantizer "${TORCH_LIBRARIES}")

# ============================================================================
# ADVANCED HNF-THEORETIC EXAMPLES (New enhancements)
# ============================================================================

# Sheaf-theoretic precision analysis (Section 4 of HNF paper)
add_executable(sheaf_cohomology_quantization examples/sheaf_cohomology_quantization.cpp)
target_link_libraries(sheaf_cohomology_quantization curvature_quantizer "${TORCH_LIBRARIES}")

# Homotopy-theoretic algorithm space (Section 4.3 of HNF paper)
add_executable(homotopy_algorithm_space examples/homotopy_algorithm_space.cpp)
target_link_libraries(homotopy_algorithm_space curvature_quantizer "${TORCH_LIBRARIES}")

# Formal verification via SMT solving
add_executable(formal_verification examples/formal_verification.cpp)
target_link_libraries(formal_verification curvature_quantizer "${TORCH_LIBRARIES}")

# Copy test data directory if exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/data)

# Installation
install(TARGETS curvature_quantizer DESTINATION lib)
install(DIRECTORY include/ DESTINATION include)
