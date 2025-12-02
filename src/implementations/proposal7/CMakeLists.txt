cmake_minimum_required(VERSION 3.18)
project(HNF_Proposal7_HomotopyLR LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find PyTorch
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Find GTest
find_package(GTest QUIET)
if(NOT GTest_FOUND)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG release-1.12.1
    )
    FetchContent_MakeAvailable(googletest)
endif()

# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${TORCH_INCLUDE_DIRS})

# Library
add_library(homotopy_lr SHARED
    src/homotopy_lr.cpp
)

target_link_libraries(homotopy_lr
    ${TORCH_LIBRARIES}
)

target_include_directories(homotopy_lr PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Tests
add_executable(test_homotopy_lr
    tests/test_homotopy_lr.cpp
)

target_link_libraries(test_homotopy_lr
    homotopy_lr
    ${TORCH_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)

# HNF Theory Validation Tests
add_executable(test_hnf_theory_validation
    tests/test_hnf_theory_validation.cpp
)

target_link_libraries(test_hnf_theory_validation
    homotopy_lr
    ${TORCH_LIBRARIES}
    GTest::gtest
    GTest::gtest_main
)

# MNIST Demo (original)
add_executable(mnist_demo
    examples/mnist_demo.cpp
)

target_link_libraries(mnist_demo
    homotopy_lr
    ${TORCH_LIBRARIES}
)

# Comprehensive MNIST Comparison
add_executable(mnist_comprehensive
    examples/mnist_comprehensive.cpp
)

target_link_libraries(mnist_comprehensive
    homotopy_lr
    ${TORCH_LIBRARIES}
)

# Installation
install(TARGETS homotopy_lr
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib
)

install(DIRECTORY include/
    DESTINATION include
)

# Enable testing
enable_testing()
add_test(NAME HomotopyLRTests COMMAND test_homotopy_lr)
add_test(NAME HNFTheoryValidation COMMAND test_hnf_theory_validation)

# Print info
message(STATUS "PyTorch version: ${Torch_VERSION}")
message(STATUS "PyTorch includes: ${TORCH_INCLUDE_DIRS}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")
