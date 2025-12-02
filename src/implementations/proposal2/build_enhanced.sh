#!/bin/bash
set -e

echo "====================================="
echo "Building HNF Proposal #2 Enhanced"
echo "====================================="

# Navigate to build directory
cd "$(dirname "$0")"
BUILD_DIR="build_enhanced"

# Create fresh build directory
if [ -d "$BUILD_DIR" ]; then
    echo "Cleaning old build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "Configuring with CMake..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch;print(torch.utils.cmake_prefix_path)')" \
    -DEIGEN3_INCLUDE_DIR="/opt/homebrew/include/eigen3"

echo ""
echo "Building..."
make -j8

echo ""
echo "====================================="
echo "Build Complete!"
echo "====================================="
echo ""
echo "Available executables:"
ls -lh test_sheaf_cohomology mnist_precision_demo comprehensive_mnist_demo 2>/dev/null || echo "Some executables may not have built (Z3 required for comprehensive demo)"
echo ""
echo "To run tests:"
echo "  ./test_sheaf_cohomology"
echo ""
echo "To run MNIST demo:"
echo "  ./mnist_precision_demo"
echo ""
echo "To run comprehensive demo (requires Z3):"
echo "  ./comprehensive_mnist_demo"
echo ""
