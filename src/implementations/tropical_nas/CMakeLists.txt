cmake_minimum_required(VERSION 3.14 FATAL_ERROR)
project(tropical_nas)

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
    src/tropical_arithmetic.cpp
    src/relu_to_tropical.cpp
    src/tropical_architecture_search.cpp
)

# Create library
add_library(tropical_nas SHARED ${SOURCES})
target_link_libraries(tropical_nas "${TORCH_LIBRARIES}")

# Test executable
add_executable(test_tropical_nas tests/test_tropical_nas.cpp)
target_link_libraries(test_tropical_nas tropical_nas "${TORCH_LIBRARIES}")

# MNIST demo executable
add_executable(mnist_demo tests/mnist_demo.cpp)
target_link_libraries(mnist_demo tropical_nas "${TORCH_LIBRARIES}")

# Set RPATH for shared libraries
set_target_properties(tropical_nas PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_target_properties(test_tropical_nas PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
set_target_properties(mnist_demo PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
