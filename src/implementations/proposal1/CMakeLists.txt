cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hnf_proposal1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find LibTorch (required)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/precision_tensor.cpp
    src/precision_nn.cpp
    src/mnist_trainer.cpp
    src/actual_training_demo.cpp
)

# Main library
add_library(hnf_proposal1 SHARED ${SOURCES})
target_link_libraries(hnf_proposal1 "${TORCH_LIBRARIES}")

# Set RPATH for MacOS
if(APPLE)
    set_target_properties(hnf_proposal1 PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Test executable
add_executable(test_proposal1 tests/test_comprehensive.cpp)
target_link_libraries(test_proposal1 hnf_proposal1 "${TORCH_LIBRARIES}")

# Comprehensive MNIST test executable
add_executable(test_comprehensive_mnist tests/test_comprehensive_mnist.cpp)
target_link_libraries(test_comprehensive_mnist hnf_proposal1 "${TORCH_LIBRARIES}")

# Advanced features test executable  
add_executable(test_advanced_features tests/test_advanced_features.cpp)
target_link_libraries(test_advanced_features hnf_proposal1 "${TORCH_LIBRARIES}")

# Example executable
add_executable(mnist_demo examples/mnist_demo.cpp)
target_link_libraries(mnist_demo hnf_proposal1 "${TORCH_LIBRARIES}")

# Advanced MNIST precision demo
add_executable(mnist_precision_demo examples/mnist_precision_demo.cpp)
target_link_libraries(mnist_precision_demo hnf_proposal1 "${TORCH_LIBRARIES}")

# Rigorous MNIST test with exact curvature formulas
add_executable(mnist_rigorous_test examples/mnist_rigorous_test.cpp)
target_link_libraries(mnist_rigorous_test hnf_proposal1 "${TORCH_LIBRARIES}")

# Comprehensive enhancements test - NEW!
add_executable(test_comprehensive_enhancements tests/test_comprehensive_enhancements.cpp)
target_link_libraries(test_comprehensive_enhancements hnf_proposal1 "${TORCH_LIBRARIES}")

# Enable testing
enable_testing()
add_test(NAME comprehensive_tests COMMAND test_proposal1)
add_test(NAME mnist_comprehensive_test COMMAND test_comprehensive_mnist)
add_test(NAME advanced_features_test COMMAND test_advanced_features)
add_test(NAME mnist_demonstration COMMAND mnist_demo)
add_test(NAME mnist_precision_demonstration COMMAND mnist_precision_demo)
add_test(NAME mnist_rigorous_validation COMMAND mnist_rigorous_test)
add_test(NAME comprehensive_enhancements_test COMMAND test_comprehensive_enhancements)

# Print build info
message(STATUS "")
message(STATUS "╔══════════════════════════════════════════════════════════╗")
message(STATUS "║  HNF Proposal #1: Precision-Aware AD                     ║")
message(STATUS "╚══════════════════════════════════════════════════════════╝")
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "")
