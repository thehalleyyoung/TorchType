#!/bin/bash

# Build script for HNF Proposal #2: Sheaf Cohomology Mixed-Precision

set -e

echo "================================================"
echo "Building HNF Proposal #2: Sheaf Cohomology"
echo "================================================"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Get PyTorch paths
echo -e "${YELLOW}Finding PyTorch installation...${NC}"
TORCH_PREFIX=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)" 2>/dev/null)
if [ -z "$TORCH_PREFIX" ]; then
    echo -e "${RED}Error: PyTorch not found. Please install PyTorch first.${NC}"
    exit 1
fi
echo -e "${GREEN}Found PyTorch at: $TORCH_PREFIX${NC}"

# Check for Eigen
echo -e "${YELLOW}Checking for Eigen3...${NC}"
if [ -d "/opt/homebrew/include/eigen3" ]; then
    EIGEN_DIR="/opt/homebrew/include/eigen3"
    echo -e "${GREEN}Found Eigen3 at: $EIGEN_DIR${NC}"
elif [ -d "/usr/include/eigen3" ]; then
    EIGEN_DIR="/usr/include/eigen3"
    echo -e "${GREEN}Found Eigen3 at: $EIGEN_DIR${NC}"
elif [ -d "/usr/local/include/eigen3" ]; then
    EIGEN_DIR="/usr/local/include/eigen3"
    echo -e "${GREEN}Found Eigen3 at: $EIGEN_DIR${NC}"
else
    echo -e "${YELLOW}Eigen3 not found system-wide. Will download header-only version...${NC}"
    
    # Download Eigen (header-only)
    if [ ! -d "eigen-3.4.0" ]; then
        echo "Downloading Eigen 3.4.0..."
        curl -L https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz -o eigen.tar.gz
        tar -xzf eigen.tar.gz
        rm eigen.tar.gz
    fi
    EIGEN_DIR="$(pwd)/eigen-3.4.0"
    echo -e "${GREEN}Using downloaded Eigen at: $EIGEN_DIR${NC}"
fi

# Create build directory
mkdir -p build
cd build

echo -e "${YELLOW}Configuring with CMake...${NC}"

# Configure
cmake .. \
    -DCMAKE_PREFIX_PATH="$TORCH_PREFIX" \
    -DEIGEN3_INCLUDE_DIR="$EIGEN_DIR" \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CXX_STANDARD=17

echo -e "${YELLOW}Building...${NC}"

# Build
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo -e "${GREEN}================================================${NC}"
echo -e "${GREEN}Build completed successfully!${NC}"
echo -e "${GREEN}================================================${NC}"

echo ""
echo "Run tests with:"
echo "  cd build && ./test_sheaf_cohomology"
echo ""
echo "Run MNIST demo with:"
echo "  cd build && ./mnist_precision_demo"
echo ""
