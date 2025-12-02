#!/bin/bash
# Comprehensive runner for Proposal #3

set -e  # Exit on error

echo "=================================================="
echo "  HNF Proposal #3: Attention Stability Analysis"
echo "  Complete Test & Demo Suite"
echo "=================================================="
echo ""

# Setup
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_DIR"

# Get LibTorch path
TORCH_LIB="$(python3 -c 'import torch, os; print(os.path.join(torch.__path__[0], "lib"))')"
export DYLD_LIBRARY_PATH="$TORCH_LIB:$DYLD_LIBRARY_PATH"

# Build if needed
if [ ! -f "build/test_attention" ]; then
    echo "Building project..."
    mkdir -p build
    cd build
    CMAKE_PREFIX_PATH="$(python3 -c 'import torch; print(torch.utils.cmake_prefix_path)')" cmake ..
    cmake --build . --parallel 4
    cd ..
    echo "Build complete!"
    echo ""
fi

# Run tests
echo "=================================================="
echo "  Running Comprehensive Tests (15 tests)"
echo "=================================================="
echo ""
./build/test_attention
echo ""

# Run demo
echo ""
echo "=================================================="
echo "  Running Vision Transformer Demonstration"
echo "=================================================="
echo ""
./build/vit_demo
echo ""

echo "=================================================="
echo "  All tests and demos complete!"
echo "=================================================="
echo ""
echo "Documentation:"
echo "  - README: src/implementations/proposal3/README.md"
echo "  - Summary: implementations/PROPOSAL3_SUMMARY.md"
echo "  - How-To: implementations/PROPOSAL3_HOWTO_DEMO.md"
echo "  - Results: implementations/PROPOSAL3_RESULTS.md"
echo ""
