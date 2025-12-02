cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hnf_sheaf_mixed_precision LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization flags
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -Wall -Wextra")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -O0 -fsanitize=address")

# Find LibTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find Eigen3 for linear algebra
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    # Try to find Eigen3 include directory manually
    if(DEFINED EIGEN3_INCLUDE_DIR)
        message(STATUS "Using Eigen3 from: ${EIGEN3_INCLUDE_DIR}")
        add_library(Eigen3::Eigen INTERFACE IMPORTED)
        set_target_properties(Eigen3::Eigen PROPERTIES
            INTERFACE_INCLUDE_DIRECTORIES "${EIGEN3_INCLUDE_DIR}"
        )
    else()
        message(FATAL_ERROR "Eigen3 not found. Please set EIGEN3_INCLUDE_DIR")
    endif()
endif()

# Find Z3 theorem prover
find_package(Z3 4.8 QUIET)
if(NOT Z3_FOUND)
    # Try to find Z3 manually via pkg-config or homebrew
    find_library(Z3_LIBRARIES NAMES z3 PATHS /usr/local/lib /opt/homebrew/lib)
    find_path(Z3_INCLUDE_DIRS NAMES z3++.h PATHS /usr/local/include /opt/homebrew/include)
    
    if(Z3_LIBRARIES AND Z3_INCLUDE_DIRS)
        message(STATUS "Found Z3: ${Z3_LIBRARIES}")
        set(Z3_FOUND TRUE)
    else()
        message(WARNING "Z3 not found. Z3-based features will be disabled. Install with: brew install z3")
        set(Z3_FOUND FALSE)
    endif()
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/../../include)
include_directories(${TORCH_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})

# Source files for advanced sheaf theory
set(SHEAF_SOURCES
    src/advanced_sheaf_theory.cpp
)

# Library with implementations
add_library(hnf_sheaf ${SHEAF_SOURCES})
target_include_directories(hnf_sheaf PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)
target_link_libraries(hnf_sheaf PUBLIC
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
)

# Test executable - original
add_executable(test_sheaf_cohomology tests/test_comprehensive.cpp)
target_link_libraries(test_sheaf_cohomology 
    hnf_sheaf
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
)

# Test executable - advanced sheaf theory
add_executable(test_advanced_sheaf tests/test_advanced_sheaf.cpp)
target_link_libraries(test_advanced_sheaf 
    hnf_sheaf
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
)

# MNIST example (original)
add_executable(mnist_precision_demo examples/mnist_demo.cpp)
target_link_libraries(mnist_precision_demo
    hnf_sheaf
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
)

# Impossible without sheaf cohomology demo
add_executable(impossible_without_sheaf examples/impossible_without_sheaf.cpp)
target_link_libraries(impossible_without_sheaf
    hnf_sheaf
    ${TORCH_LIBRARIES}
    Eigen3::Eigen
)

# Comprehensive MNIST demo with Z3 and persistent cohomology
if(Z3_FOUND)
    add_executable(comprehensive_mnist_demo examples/comprehensive_mnist_demo.cpp)
    target_link_libraries(comprehensive_mnist_demo
        hnf_sheaf
        ${TORCH_LIBRARIES}
        Eigen3::Eigen
        ${Z3_LIBRARIES}
    )
    target_include_directories(comprehensive_mnist_demo PRIVATE ${Z3_INCLUDE_DIRS})
    
    # Add Z3 compile definition
    target_compile_definitions(comprehensive_mnist_demo PRIVATE HNF_HAS_Z3)
    
    if(APPLE)
        set_target_properties(comprehensive_mnist_demo PROPERTIES
            INSTALL_RPATH "${TORCH_LIB_DIR}"
            BUILD_WITH_INSTALL_RPATH TRUE
        )
    endif()
    
    add_test(NAME comprehensive_mnist_demo COMMAND comprehensive_mnist_demo)
endif()

# Set RPATH for MacOS
if(APPLE)
    # Get libtorch library path
    get_filename_component(TORCH_LIB_DIR "${TORCH_LIBRARIES}" DIRECTORY)
    
    set_target_properties(test_sheaf_cohomology PROPERTIES
        INSTALL_RPATH "${TORCH_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    set_target_properties(test_advanced_sheaf PROPERTIES
        INSTALL_RPATH "${TORCH_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    set_target_properties(mnist_precision_demo PROPERTIES
        INSTALL_RPATH "${TORCH_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
    set_target_properties(impossible_without_sheaf PROPERTIES
        INSTALL_RPATH "${TORCH_LIB_DIR}"
        BUILD_WITH_INSTALL_RPATH TRUE
    )
endif()

# Enable testing
enable_testing()
add_test(NAME sheaf_cohomology_tests COMMAND test_sheaf_cohomology)
add_test(NAME advanced_sheaf_tests COMMAND test_advanced_sheaf)
add_test(NAME mnist_demo COMMAND mnist_precision_demo)
add_test(NAME impossible_demo COMMAND impossible_without_sheaf)

# Copy Python C++ flags for compatibility
set_property(TARGET test_sheaf_cohomology PROPERTY CXX_STANDARD 17)
set_property(TARGET test_advanced_sheaf PROPERTY CXX_STANDARD 17)
set_property(TARGET mnist_precision_demo PROPERTY CXX_STANDARD 17)
set_property(TARGET impossible_without_sheaf PROPERTY CXX_STANDARD 17)

# Print configuration
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER}")
message(STATUS "C++ flags: ${CMAKE_CXX_FLAGS}")
message(STATUS "Torch found: ${TORCH_FOUND}")
message(STATUS "Torch version: ${Torch_VERSION}")
message(STATUS "Eigen3 found: ${EIGEN3_FOUND}")
message(STATUS "Z3 found: ${Z3_FOUND}")
if(Z3_FOUND)
    message(STATUS "Z3 libraries: ${Z3_LIBRARIES}")
    message(STATUS "Z3 include dirs: ${Z3_INCLUDE_DIRS}")
endif()
