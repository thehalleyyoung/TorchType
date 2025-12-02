cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hnf_condition_profiler LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Include Eigen from proposal2 if available
set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../proposal2/eigen-3.4.0")
if(EXISTS ${EIGEN3_INCLUDE_DIR})
    message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")
else()
    # Try system locations
    find_package(Eigen3 QUIET)
    if(NOT Eigen3_FOUND)
        message(WARNING "Eigen3 not found - some tests may not compile")
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})
if(EXISTS ${EIGEN3_INCLUDE_DIR})
    include_directories(${EIGEN3_INCLUDE_DIR})
elseif(EIGEN3_INCLUDE_DIRS)
    include_directories(${EIGEN3_INCLUDE_DIRS})
endif()

# Source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.hpp")

# Main library
add_library(hnf_profiler SHARED ${SOURCES})
target_link_libraries(hnf_profiler "${TORCH_LIBRARIES}")

# Test executables
add_executable(test_profiler tests/test_main.cpp)
target_link_libraries(test_profiler hnf_profiler "${TORCH_LIBRARIES}")

add_executable(test_comprehensive tests/test_comprehensive.cpp)
target_link_libraries(test_comprehensive hnf_profiler "${TORCH_LIBRARIES}")

add_executable(test_rigorous tests/test_rigorous.cpp)
target_link_libraries(test_rigorous hnf_profiler "${TORCH_LIBRARIES}")

add_executable(test_advanced tests/test_advanced.cpp)
target_link_libraries(test_advanced hnf_profiler "${TORCH_LIBRARIES}")

add_executable(test_advanced_simple tests/test_advanced_simple.cpp)
target_link_libraries(test_advanced_simple hnf_profiler "${TORCH_LIBRARIES}")

# Example executables
file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
foreach(example_src ${EXAMPLE_SOURCES})
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_link_libraries(${example_name} hnf_profiler "${TORCH_LIBRARIES}")
endforeach()

# Set RPATH for MacOS
if(APPLE)
    set_target_properties(hnf_profiler PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

enable_testing()
add_test(NAME profiler_tests COMMAND test_profiler)
