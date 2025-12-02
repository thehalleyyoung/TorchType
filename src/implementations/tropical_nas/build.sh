#!/bin/bash

# Tropical NAS Build Script
# This script compiles the Tropical Geometry NAS implementation

set -e

echo "======================================================"
echo "Building Tropical Geometry NAS"
echo "======================================================"

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
if [ ! -d "build" ]; then
    mkdir build
fi

cd build

# Find PyTorch
TORCH_CMAKE_PATH=$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)' 2>/dev/null)

if [ -z "$TORCH_CMAKE_PATH" ]; then
    echo "Error: Could not find PyTorch installation"
    echo "Please install PyTorch: pip install torch"
    exit 1
fi

echo "Found PyTorch at: $TORCH_CMAKE_PATH"

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_PREFIX_PATH="$TORCH_CMAKE_PATH" \
         -DCMAKE_BUILD_TYPE=Release

# Build
echo "Building..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "======================================================"
echo "Build complete!"
echo "======================================================"
echo ""
echo "Executables created:"
echo "  - test_tropical_nas  : Comprehensive test suite"
echo "  - mnist_demo         : MNIST demonstration"
echo ""
echo "To run tests:"
echo "  export DYLD_LIBRARY_PATH=\"\$(python3 -c 'import torch; import os; print(os.path.join(torch.__path__[0], \"lib\"))')\":\$DYLD_LIBRARY_PATH"
echo "  ./test_tropical_nas"
echo ""
echo "To run MNIST demo (requires MNIST data):"
echo "  ./mnist_demo /path/to/MNIST/raw"
echo ""
