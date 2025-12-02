#!/bin/bash

# Build script for HNF Stability Linter
# This script compiles the implementation without needing a build directory

set -e

PROJECT_DIR="/Users/halleyyoung/Documents/TorchType/src/implementations/proposal10"
BUILD_DIR="/Users/halleyyoung/Documents/TorchType/src/implementations/proposal10/output"
TORCH_PREFIX="/Users/halleyyoung/Library/Python/3.14/lib/python/site-packages/torch/share/cmake"

echo "=== HNF Stability Linter Build Script ==="
echo "Project: $PROJECT_DIR"
echo "Output:  $BUILD_DIR"
echo "Torch:   $TORCH_PREFIX"
echo ""

# Create output directory
rm -rf "$BUILD_DIR"
mkdir -p "$BUILD_DIR"

# Find torch libraries
TORCH_LIB_DIR="/Users/halleyyoung/Library/Python/3.14/lib/python/site-packages/torch/lib"
TORCH_INCLUDE_DIR="/Users/halleyyoung/Library/Python/3.14/lib/python/site-packages/torch/include"

# Check if torch directories exist
if [ ! -d "$TORCH_LIB_DIR" ]; then
    echo "Error: Torch library directory not found: $TORCH_LIB_DIR"
    exit 1
fi

if [ ! -d "$TORCH_INCLUDE_DIR" ]; then
    echo "Error: Torch include directory not found: $TORCH_INCLUDE_DIR"
    exit 1
fi

echo "Using LibTorch from: $TORCH_LIB_DIR"
echo ""

# Compile source files to object files
echo "Compiling source files..."

c++ -std=c++17 -O3 -c \
    -I"$PROJECT_DIR/include" \
    -I"$TORCH_INCLUDE_DIR" \
    -I"$TORCH_INCLUDE_DIR/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    "$PROJECT_DIR/src/stability_linter.cpp" \
    -o "$BUILD_DIR/stability_linter.o"

c++ -std=c++17 -O3 -c \
    -I"$PROJECT_DIR/include" \
    -I"$TORCH_INCLUDE_DIR" \
    -I"$TORCH_INCLUDE_DIR/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    "$PROJECT_DIR/src/patterns.cpp" \
    -o "$BUILD_DIR/patterns.o"

c++ -std=c++17 -O3 -c \
    -I"$PROJECT_DIR/include" \
    -I"$TORCH_INCLUDE_DIR" \
    -I"$TORCH_INCLUDE_DIR/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    "$PROJECT_DIR/src/sheaf_cohomology.cpp" \
    -o "$BUILD_DIR/sheaf_cohomology.o"

echo "✓ Source files compiled"
echo ""

# Create shared library
echo "Creating shared library..."

c++ -shared \
    "$BUILD_DIR/stability_linter.o" \
    "$BUILD_DIR/patterns.o" \
    "$BUILD_DIR/sheaf_cohomology.o" \
    -L"$TORCH_LIB_DIR" \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$TORCH_LIB_DIR" \
    -o "$BUILD_DIR/libstability_linter.dylib"

echo "✓ Library created: $BUILD_DIR/libstability_linter.dylib"
echo ""

# Compile test executable
echo "Compiling test executable..."

c++ -std=c++17 -O3 \
    -I"$PROJECT_DIR/include" \
    -I"$TORCH_INCLUDE_DIR" \
    -I"$TORCH_INCLUDE_DIR/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    "$PROJECT_DIR/tests/test_linter.cpp" \
    "$BUILD_DIR/libstability_linter.dylib" \
    -L"$TORCH_LIB_DIR" \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$TORCH_LIB_DIR" \
    -Wl,-rpath,"$BUILD_DIR" \
    -o "$BUILD_DIR/test_linter"

echo "✓ Test executable created: $BUILD_DIR/test_linter"
echo ""

# Compile demo executable
echo "Compiling demo executable..."

c++ -std=c++17 -O3 \
    -I"$PROJECT_DIR/include" \
    -I"$TORCH_INCLUDE_DIR" \
    -I"$TORCH_INCLUDE_DIR/torch/csrc/api/include" \
    -D_GLIBCXX_USE_CXX11_ABI=0 \
    "$PROJECT_DIR/examples/demo_linter.cpp" \
    "$BUILD_DIR/libstability_linter.dylib" \
    -L"$TORCH_LIB_DIR" \
    -ltorch -ltorch_cpu -lc10 \
    -Wl,-rpath,"$TORCH_LIB_DIR" \
    -Wl,-rpath,"$BUILD_DIR" \
    -o "$BUILD_DIR/demo_linter"

echo "✓ Demo executable created: $BUILD_DIR/demo_linter"
echo ""

echo "=== Build Complete ==="
echo ""
echo "Run tests with:  $BUILD_DIR/test_linter"
echo "Run demo with:   $BUILD_DIR/demo_linter"
echo ""
