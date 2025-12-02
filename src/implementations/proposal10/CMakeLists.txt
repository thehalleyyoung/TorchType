cmake_minimum_required(VERSION 3.18 FATAL_ERROR)
project(hnf_stability_linter LANGUAGES CXX)

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
    src/stability_linter.cpp
    src/patterns.cpp
)

# Main library
add_library(stability_linter SHARED ${SOURCES})
target_link_libraries(stability_linter "${TORCH_LIBRARIES}")

# Test executable
add_executable(test_linter tests/test_linter.cpp)
target_link_libraries(test_linter stability_linter "${TORCH_LIBRARIES}")

# Example/demo executable
add_executable(demo_linter examples/demo_linter.cpp)
target_link_libraries(demo_linter stability_linter "${TORCH_LIBRARIES}")

# Set RPATH for MacOS
if(APPLE)
    set_target_properties(stability_linter PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(test_linter PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
    set_target_properties(demo_linter PROPERTIES
        INSTALL_RPATH "@loader_path"
        BUILD_WITH_INSTALL_RPATH TRUE)
endif()

# Enable testing
enable_testing()
add_test(NAME stability_linter_tests COMMAND test_linter)

# Installation
install(TARGETS stability_linter test_linter demo_linter
        LIBRARY DESTINATION lib
        RUNTIME DESTINATION bin)
install(DIRECTORY include/ DESTINATION include)
