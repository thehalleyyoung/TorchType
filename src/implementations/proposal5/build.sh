#!/bin/bash

# Build script for HNF Proposal 5: Condition Number Profiler

set -e

echo "=== Building HNF Condition Profiler ==="

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring with CMake..."
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')"

# Build
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "=== Build completed! ==="
echo ""
echo "Run tests with:     ./test_profiler"
echo "Run examples with:  ./transformer_profiling"
echo "                    ./mnist_precision"
echo ""
