#!/bin/bash

# Build and run script for Proposal 8: KV-Cache Precision Analyzer

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║   Building Proposal 8: KV-Cache Precision Analyzer             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Create build directory
mkdir -p build
cd build

# Configure with CMake
echo "Configuring with CMake..."
cmake .. -DCMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
         -DCMAKE_BUILD_TYPE=Release

# Build
echo ""
echo "Building..."
make -j$(sysctl -n hw.ncpu)

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Build Complete!                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Run tests
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Running Tests                               ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

./test_kv_cache
TEST_RESULT=$?

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    Running Examples                            ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Run simple demo
if [ -f "./simple_demo" ]; then
    echo "Running simple demo..."
    echo "════════════════════════════════════════════════════════════════"
    ./simple_demo
    echo ""
fi

# Run transformer demo
if [ -f "./transformer_demo" ]; then
    echo "Running transformer demo..."
    echo "════════════════════════════════════════════════════════════════"
    ./transformer_demo
    echo ""
fi

cd ..

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    All Done!                                   ║"
echo "╚════════════════════════════════════════════════════════════════╝"

if [ $TEST_RESULT -eq 0 ]; then
    echo "✓ All tests passed!"
    exit 0
else
    echo "✗ Some tests failed"
    exit 1
fi
