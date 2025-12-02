cmake_minimum_required(VERSION 3.14)
project(hnf_proposal4_rewriter)

# C++17 required
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler flags
if(APPLE)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find Eigen3 (optional - will use built-in fallback if not found)
find_package(Eigen3 QUIET)
if(EIGEN3_FOUND)
    include_directories(${EIGEN3_INCLUDE_DIR})
    add_definitions(-DHAS_EIGEN3)
    message(STATUS "Found Eigen3: ${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "Eigen3 not found - using simplified matrix operations")
endif()

# All headers are header-only, so we just need to compile the tests and examples

# Original test executable
add_executable(test_proposal4
    tests/test_comprehensive.cpp
)

# New comprehensive neural network test
add_executable(test_neural_network
    tests/test_neural_network.cpp
)

# MNIST feedforward network test (the big demonstration!)
add_executable(test_mnist_feedforward
    tests/test_mnist_feedforward.cpp
)

# NEW: Comprehensive enhanced test with all advanced features
add_executable(test_comprehensive_enhanced
    tests/test_comprehensive_enhanced.cpp
)

# NEW ULTIMATE ENHANCEMENTS:
# Real MNIST training demonstration
add_executable(test_mnist_training
    tests/test_mnist_training.cpp
)

# Z3-based formal verification
add_executable(test_z3_verification
    tests/test_z3_verification.cpp
)

# Comprehensive performance benchmarking
add_executable(test_benchmarking
    tests/test_benchmarking.cpp
)

# Example executable - transformer demo
add_executable(transformer_demo
    examples/transformer_demo.cpp
)

# Installation (optional)
install(TARGETS test_proposal4 test_neural_network test_mnist_feedforward test_comprehensive_enhanced 
        test_mnist_training test_z3_verification test_benchmarking transformer_demo
        RUNTIME DESTINATION bin)

install(DIRECTORY include/
        DESTINATION include/hnf/proposal4
        FILES_MATCHING PATTERN "*.hpp")
