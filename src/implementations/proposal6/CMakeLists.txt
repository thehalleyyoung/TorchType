cmake_minimum_required(VERSION 3.15)
project(Proposal6_CertifiedBounds CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find Eigen3
find_package(Eigen3 REQUIRED)

# Include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    add_compile_options(-Wall -Wextra -O3 -march=native)
endif()

# Source files (header-only, but we can create a library if needed)
set(HEADERS
    include/interval.hpp
    include/input_domain.hpp
    include/curvature_bounds.hpp
    include/certifier.hpp
    include/affine_form.hpp
    include/autodiff.hpp
    include/mnist_data.hpp
    include/real_mnist_loader.hpp
    include/neural_network.hpp
    include/z3_precision_prover.hpp
    include/zonotope.hpp
    include/probabilistic_certifier.hpp
    include/transformer_attention.hpp
)

# Create interface library for headers
add_library(certified_bounds INTERFACE)
target_include_directories(certified_bounds INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)
target_link_libraries(certified_bounds INTERFACE Eigen3::Eigen)

# Tests
add_executable(test_comprehensive tests/test_comprehensive.cpp)
target_link_libraries(test_comprehensive certified_bounds)

add_executable(test_advanced_features tests/test_advanced_features.cpp)
target_link_libraries(test_advanced_features certified_bounds)

# Z3 formal verification (if available)
find_library(Z3_LIBRARY NAMES z3 PATHS /usr/local/lib /opt/homebrew/lib /usr/lib)
find_path(Z3_INCLUDE_DIR NAMES z3++.h PATHS /usr/local/include /opt/homebrew/include /usr/include)

if(Z3_LIBRARY AND Z3_INCLUDE_DIR)
    message(STATUS "  Z3 found: ${Z3_LIBRARY}")
    add_executable(test_z3_formal_proofs tests/test_z3_formal_proofs.cpp)
    target_include_directories(test_z3_formal_proofs PRIVATE ${Z3_INCLUDE_DIR})
    target_link_libraries(test_z3_formal_proofs certified_bounds ${Z3_LIBRARY})
    
    add_executable(test_advanced_smt_prover tests/test_advanced_smt_prover.cpp)
    target_include_directories(test_advanced_smt_prover PRIVATE ${Z3_INCLUDE_DIR})
    target_link_libraries(test_advanced_smt_prover certified_bounds ${Z3_LIBRARY})
    
    message(STATUS "  ✓ Z3 formal verification enabled")
else()
    message(STATUS "  ⚠ Z3 not found - formal verification tests will be skipped")
    message(STATUS "    Install with: brew install z3 (macOS) or apt install libz3-dev (Linux)")
endif()

# Examples
add_executable(mnist_transformer_demo examples/mnist_transformer_demo.cpp)
target_link_libraries(mnist_transformer_demo certified_bounds)

add_executable(impossibility_demo examples/impossibility_demo.cpp)
target_link_libraries(impossibility_demo certified_bounds)

add_executable(comprehensive_mnist_demo examples/comprehensive_mnist_demo.cpp)
target_link_libraries(comprehensive_mnist_demo certified_bounds)

add_executable(comprehensive_validation examples/comprehensive_validation.cpp)
target_link_libraries(comprehensive_validation certified_bounds)

add_executable(zonotope_enhancement_demo examples/zonotope_enhancement_demo.cpp)
target_link_libraries(zonotope_enhancement_demo certified_bounds)

# Enable testing
enable_testing()
add_test(NAME ComprehensiveTests COMMAND test_comprehensive)
add_test(NAME AdvancedFeatureTests COMMAND test_advanced_features)

# Custom target to run all tests
add_custom_target(run_tests
    COMMAND ${CMAKE_CTEST_COMMAND} --output-on-failure
    DEPENDS test_comprehensive test_advanced_features
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Custom target to run demo
add_custom_target(run_demo
    COMMAND comprehensive_mnist_demo
    DEPENDS comprehensive_mnist_demo
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}
)

# Installation
install(DIRECTORY include/ DESTINATION include/hnf/certified)
install(TARGETS certified_bounds
    EXPORT CertifiedBoundsTargets
    INCLUDES DESTINATION include
)

# Print configuration
message(STATUS "Proposal 6: Certified Precision Bounds")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Eigen3: ${EIGEN3_INCLUDE_DIR}")
message(STATUS "  Build type: ${CMAKE_BUILD_TYPE}")
