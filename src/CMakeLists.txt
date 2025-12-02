cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hnf_precision_aware_ad LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find Z3 - we'll build only what we need
find_package(Z3 QUIET)
if(NOT Z3_FOUND)
    message(STATUS "Z3 not found, will use built-in without SMT verification")
    set(USE_Z3 OFF)
else()
    set(USE_Z3 ON)
    include_directories(${Z3_CXX_INCLUDE_DIRS})
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Source files
file(GLOB SOURCES "src/*.cpp")
file(GLOB HEADERS "include/*.h" "include/*.hpp")

# Main library
add_library(hnf_precision SHARED ${SOURCES})
target_link_libraries(hnf_precision "${TORCH_LIBRARIES}")

if(USE_Z3)
    target_link_libraries(hnf_precision ${Z3_LIBRARIES})
    target_compile_definitions(hnf_precision PRIVATE USE_Z3)
endif()

# Test executable
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(test_hnf ${TEST_SOURCES})
target_link_libraries(test_hnf hnf_precision "${TORCH_LIBRARIES}")

if(USE_Z3)
    target_link_libraries(test_hnf ${Z3_LIBRARIES})
    target_compile_definitions(test_hnf PRIVATE USE_Z3)
endif()

# Example executable
file(GLOB EXAMPLE_SOURCES "examples/*.cpp")
if(EXAMPLE_SOURCES)
    add_executable(hnf_examples ${EXAMPLE_SOURCES})
    target_link_libraries(hnf_examples hnf_precision "${TORCH_LIBRARIES}")
    
    if(USE_Z3)
        target_link_libraries(hnf_examples ${Z3_LIBRARIES})
        target_compile_definitions(hnf_examples PRIVATE USE_Z3)
    endif()
endif()

# Set RPATH for MacOS
if(APPLE)
    set_target_properties(hnf_precision PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

enable_testing()
add_test(NAME hnf_tests COMMAND test_hnf)
