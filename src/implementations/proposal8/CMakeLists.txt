cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(proposal8_kv_cache_precision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
set(SOURCES
    src/curvature_analyzer.cpp
    src/precision_mapper.cpp
    src/mixed_precision_buffer.cpp
    src/kv_cache_analyzer.cpp
    src/hnf_theorem_verifier.cpp
    src/real_data_validator.cpp
)

set(HEADERS
    include/kv_cache_types.hpp
    include/curvature_analyzer.hpp
    include/precision_mapper.hpp
    include/mixed_precision_buffer.hpp
    include/kv_cache_analyzer.hpp
    include/hnf_theorem_verifier.hpp
    include/real_data_validator.hpp
)

# Main library
add_library(kv_cache_precision SHARED ${SOURCES} ${HEADERS})
target_link_libraries(kv_cache_precision "${TORCH_LIBRARIES}")

# Test executable
set(TEST_SOURCES
    tests/test_comprehensive.cpp
)
add_executable(test_kv_cache ${TEST_SOURCES})
target_link_libraries(test_kv_cache kv_cache_precision "${TORCH_LIBRARIES}")

# Enhanced test executable
add_executable(test_kv_cache_enhanced tests/test_enhanced.cpp)
target_link_libraries(test_kv_cache_enhanced kv_cache_precision "${TORCH_LIBRARIES}")

# Real-world validation test
add_executable(test_real_world_validation tests/test_real_world_validation.cpp)
target_link_libraries(test_real_world_validation kv_cache_precision "${TORCH_LIBRARIES}")

# Example executables
set(EXAMPLE_SOURCES
    examples/simple_demo.cpp
    examples/transformer_demo.cpp
)

foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} kv_cache_precision "${TORCH_LIBRARIES}")
endforeach()

# Set RPATH for MacOS
if(APPLE)
    set_target_properties(kv_cache_precision PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

enable_testing()
add_test(NAME kv_cache_tests COMMAND test_kv_cache)
