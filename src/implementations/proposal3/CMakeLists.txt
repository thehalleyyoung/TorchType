cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hnf_attention_stability LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/attention_curvature.cpp
    src/attention_analyzer.cpp
    src/sheaf_cohomology.cpp
    src/real_training.cpp
    src/impossibility_verification.cpp
    src/mnist_attention_trainer.cpp
    src/formal_verification.cpp
)

# Headers
set(HEADERS
    include/attention_types.hpp
    include/attention_curvature.hpp
    include/attention_analyzer.hpp
    include/sheaf_cohomology.hpp
    include/real_training.hpp
    include/mnist_attention_trainer.hpp
    include/formal_verification.hpp
)

# Main library
add_library(hnf_attention SHARED ${SOURCES})
target_link_libraries(hnf_attention "${TORCH_LIBRARIES}")

# Test executable
add_executable(test_attention tests/test_comprehensive.cpp)
target_link_libraries(test_attention hnf_attention "${TORCH_LIBRARIES}")

# Enhanced test executable
add_executable(test_enhanced tests/test_enhanced.cpp)
target_link_libraries(test_enhanced hnf_attention "${TORCH_LIBRARIES}")

# Example executable
add_executable(vit_demo examples/vit_stability_demo.cpp)
target_link_libraries(vit_demo hnf_attention "${TORCH_LIBRARIES}")

# Comprehensive demo executable
add_executable(hnf_comprehensive_demo examples/hnf_comprehensive_demo.cpp)
target_link_libraries(hnf_comprehensive_demo hnf_attention "${TORCH_LIBRARIES}")

# New comprehensive enhancement demo
add_executable(comprehensive_enhancement_demo examples/comprehensive_enhancement_demo.cpp)
target_link_libraries(comprehensive_enhancement_demo hnf_attention "${TORCH_LIBRARIES}")

# Practical training demo (NEW)
add_executable(practical_training_demo examples/practical_training_demo.cpp)
target_link_libraries(practical_training_demo hnf_attention "${TORCH_LIBRARIES}")

# Set RPATH for shared library loading
if(APPLE)
    set_target_properties(hnf_attention PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(test_attention PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(test_enhanced PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(vit_demo PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(hnf_comprehensive_demo PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(comprehensive_enhancement_demo PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(practical_training_demo PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
elseif(UNIX)
    set_target_properties(hnf_attention PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(test_attention PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(test_enhanced PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(vit_demo PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(hnf_comprehensive_demo PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(comprehensive_enhancement_demo PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(practical_training_demo PROPERTIES
        INSTALL_RPATH "$ORIGIN"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Enable testing
enable_testing()
add_test(NAME attention_tests COMMAND test_attention)
add_test(NAME enhanced_tests COMMAND test_enhanced)

# Installation
install(TARGETS hnf_attention
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib)
install(FILES ${HEADERS} DESTINATION include/hnf/attention)
install(TARGETS test_attention vit_demo hnf_comprehensive_demo comprehensive_enhancement_demo practical_training_demo
        RUNTIME DESTINATION bin)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")
