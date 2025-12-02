#!/bin/bash

# Build script for HNF Proposal #4: Stability-Preserving Graph Rewriter

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════════╗"
echo "║  Building HNF Proposal #4: Graph Rewriter                      ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Create build directory
BUILD_DIR="build"
if [ -d "$BUILD_DIR" ]; then
    echo "→ Cleaning existing build directory..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo "→ Running CMake..."
cmake .. -DCMAKE_BUILD_TYPE=Release

echo ""
echo "→ Compiling..."
make -j$(sysctl -n hw.ncpu 2>/dev/null || nproc 2>/dev/null || echo 4)

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                    BUILD SUCCESSFUL                             ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""
echo "Built executables:"
echo "  • $BUILD_DIR/test_proposal4         - Comprehensive test suite"
echo "  • $BUILD_DIR/transformer_demo       - Transformer optimization demo"
echo ""
echo "Run tests with:"
echo "  ./build/test_proposal4"
echo ""
echo "Run demo with:"
echo "  ./build/transformer_demo"
echo ""
