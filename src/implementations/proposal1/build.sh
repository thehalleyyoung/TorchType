#!/bin/bash

# Build script for HNF Proposal #1: Precision-Aware Automatic Differentiation

set -e

echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                          ║"
echo "║   HNF PROPOSAL #1: PRECISION-AWARE AUTOMATIC DIFFERENTIATION            ║"
echo "║   Build Script                                                          ║"
echo "║                                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""

# Check for LibTorch
if [ -z "$LIBTORCH_PATH" ]; then
    echo "⚠️  LIBTORCH_PATH not set. Trying to find libtorch..."
    
    # Common locations
    for path in \
        "$HOME/libtorch" \
        "/usr/local/libtorch" \
        "/opt/libtorch" \
        "$(python3 -c 'import torch; import os; print(os.path.dirname(torch.__file__))' 2>/dev/null)" \
        ; do
        if [ -d "$path" ]; then
            export LIBTORCH_PATH="$path"
            echo "✓ Found libtorch at: $LIBTORCH_PATH"
            break
        fi
    done
    
    if [ -z "$LIBTORCH_PATH" ]; then
        echo "✗ LibTorch not found. Please install PyTorch or download libtorch:"
        echo "   pip3 install torch"
        echo "   or download from: https://pytorch.org/get-started/locally/"
        exit 1
    fi
fi

# Set Torch_DIR for CMake
export Torch_DIR="$LIBTORCH_PATH/share/cmake/Torch"

echo ""
echo "Build Configuration:"
echo "  LibTorch: $LIBTORCH_PATH"
echo "  Torch_DIR: $Torch_DIR"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring CMake..."
cmake -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
      -DCMAKE_BUILD_TYPE=Release \
      ..

# Build
echo ""
echo "Building..."
make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo ""
echo "╔══════════════════════════════════════════════════════════════════════════╗"
echo "║  BUILD COMPLETE                                                          ║"
echo "╚══════════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Executables:"
echo "  ./build/test_proposal1  - Comprehensive test suite"
echo "  ./build/mnist_demo      - MNIST demonstration"
echo ""
echo "To run tests:"
echo "  cd build && ctest --verbose"
echo ""
echo "Or run directly:"
echo "  ./build/test_proposal1"
echo "  ./build/mnist_demo"
echo ""
