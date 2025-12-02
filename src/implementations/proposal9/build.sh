#!/bin/bash

# Build script for Proposal 9: Curvature-Guided Quantization

set -e

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Building Proposal 9: Curvature-Guided Quantization          ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo

# Create build directory
mkdir -p build
cd build

# Configure
echo "Configuring with CMake..."
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
         -DCMAKE_BUILD_TYPE=Release

# Build
echo
echo "Building..."
cmake --build . --config Release -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)

echo
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                      BUILD COMPLETE!                           ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo
echo "Executables created:"
echo "  - test_comprehensive            : Comprehensive test suite"
echo "  - mnist_quantization_demo       : MNIST quantization demo"
echo "  - resnet_quantization           : ResNet-18 quantization"
echo "  - transformer_layer_quant       : Transformer quantization"
echo
echo "To run tests:"
echo "  ./test_comprehensive"
echo
echo "To run demos:"
echo "  ./mnist_quantization_demo"
echo "  ./resnet_quantization"
echo "  ./transformer_layer_quant"
echo
