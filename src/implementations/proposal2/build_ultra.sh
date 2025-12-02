#!/bin/bash

# Enhanced build script for Proposal #2
# Builds all new advanced sheaf theory components

set -e  # Exit on error

echo "=================================="
echo "Building HNF Proposal #2: Enhanced"
echo "Advanced Sheaf Cohomology System"
echo "=================================="
echo ""

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Clean previous build
if [ -d "build_ultra" ]; then
    echo -e "${YELLOW}Cleaning previous build...${NC}"
    rm -rf build_ultra
fi

# Create build directory
mkdir -p build_ultra
cd build_ultra

echo -e "${GREEN}Configuring CMake...${NC}"

# Set Eigen path (adjust if needed)
EIGEN_PATH="${PWD}/../eigen-3.4.0"

# Try to find libtorch
LIBTORCH_PATH=""

# First try Python's torch
PYTHON_TORCH=$(python3 -c "import torch; print(torch.__path__[0])" 2>/dev/null || echo "")
if [ -n "$PYTHON_TORCH" ] && [ -d "$PYTHON_TORCH" ]; then
    LIBTORCH_PATH="$PYTHON_TORCH"
    echo "Found PyTorch via Python: $LIBTORCH_PATH"
elif [ -d "/opt/homebrew/opt/pytorch/lib" ]; then
    LIBTORCH_PATH="/opt/homebrew/opt/pytorch"
elif [ -d "$HOME/libtorch" ]; then
    LIBTORCH_PATH="$HOME/libtorch"
elif [ -d "/usr/local/libtorch" ]; then
    LIBTORCH_PATH="/usr/local/libtorch"
fi

if [ -z "$LIBTORCH_PATH" ]; then
    echo -e "${RED}ERROR: Could not find libtorch${NC}"
    echo "Please install PyTorch:"
    echo "  pip3 install torch"
    exit 1
fi

echo "Using libtorch from: $LIBTORCH_PATH"
echo "Using Eigen from: $EIGEN_PATH"

# Configure with CMake
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH="$LIBTORCH_PATH" \
    -DEIGEN3_INCLUDE_DIR="$EIGEN_PATH" \
    -DCMAKE_CXX_FLAGS="-O3 -march=native -std=c++17" \
    || { echo -e "${RED}CMake configuration failed${NC}"; exit 1; }

echo -e "${GREEN}Building...${NC}"
make -j$(sysctl -n hw.ncpu) || { echo -e "${RED}Build failed${NC}"; exit 1; }

echo ""
echo -e "${GREEN}=================================="
echo "Build completed successfully!"
echo "==================================${NC}"
echo ""
echo "Executables created:"
echo "  • test_sheaf_cohomology (original tests)"
echo "  • test_advanced_sheaf (NEW advanced sheaf theory tests)"
echo "  • mnist_precision_demo (original MNIST demo)"
echo "  • impossible_without_sheaf (NEW impossibility demonstration)"
if [ -f "comprehensive_mnist_demo" ]; then
    echo "  • comprehensive_mnist_demo (with Z3 support)"
fi
echo ""
echo "To run tests:"
echo "  cd build_ultra"
echo "  ./test_advanced_sheaf"
echo "  ./impossible_without_sheaf"
echo ""
