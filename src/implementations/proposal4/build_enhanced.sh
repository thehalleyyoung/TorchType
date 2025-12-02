#!/bin/bash

# Enhanced build script for Proposal 4

echo "Building HNF Proposal #4 - Enhanced Implementation"
echo "=================================================="
echo ""

# Create build directory if it doesn't exist
if [ ! -d "build_enhanced" ]; then
    mkdir build_enhanced
fi

cd build_enhanced

# Run cmake
echo "Configuring with CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# Build all targets
echo ""
echo "Building all targets..."
make -j4

echo ""
echo "Build complete!"
echo ""
echo "Available executables:"
ls -lh test_* transformer_demo 2>/dev/null || echo "  (check build output for errors)"

echo ""
echo "To run comprehensive enhanced test:"
echo "  ./test_comprehensive_enhanced"
