#!/bin/bash

# Build script for Proposal 6: Certified Precision Bounds

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Building Proposal 6: Certified Precision Bounds         ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║  Build complete!                                          ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "To run tests:  ./build/test_comprehensive"
echo "To run demo:   ./build/mnist_transformer_demo"
echo ""
