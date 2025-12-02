#!/bin/bash

# Enhanced build script for Proposal 10 with all new features

set -e  # Exit on error

echo "════════════════════════════════════════════════════════════════════"
echo "  Building HNF Stability Linter - Enhanced Implementation"
echo "  Proposal #10: Numerical Stability Linter for Transformer Code"
echo "════════════════════════════════════════════════════════════════════"
echo ""

# Configuration
LIBTORCH_PREFIX="/opt/homebrew/Cellar/pytorch/2.1.0_4"
if [ ! -d "$LIBTORCH_PREFIX" ]; then
    echo "⚠️  Warning: LibTorch not found at $LIBTORCH_PREFIX"
    echo "   Looking for alternative locations..."
    
    for alt in "/usr/local/libtorch" "/opt/libtorch" "$HOME/libtorch"; do
        if [ -d "$alt" ]; then
            LIBTORCH_PREFIX="$alt"
            echo "   Found LibTorch at: $LIBTORCH_PREFIX"
            break
        fi
    done
fi

OUTPUT_DIR="output"
SRC_DIR="src"
INCLUDE_DIR="include"
EXAMPLES_DIR="examples"
TESTS_DIR="tests"

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Compiler flags
CXX_FLAGS="-std=c++17 -Wall -Wextra -O2 -g"
INCLUDE_FLAGS="-I$INCLUDE_DIR -I$LIBTORCH_PREFIX/include -I$LIBTORCH_PREFIX/include/torch/csrc/api/include"
LINK_FLAGS="-L$LIBTORCH_PREFIX/lib -ltorch -lc10"

# macOS specific flags
if [[ "$OSTYPE" == "darwin"* ]]; then
    LINK_FLAGS="$LINK_FLAGS -Wl,-rpath,$LIBTORCH_PREFIX/lib"
fi

echo "Compiler Configuration:"
echo "  CXX: $(which c++)"
echo "  C++ Standard: C++17"
echo "  LibTorch: $LIBTORCH_PREFIX"
echo "  Output: $OUTPUT_DIR/"
echo ""

# Step 1: Compile core library sources
echo "──────────────────────────────────────────────────────────────────"
echo "Step 1: Compiling Core Library"
echo "──────────────────────────────────────────────────────────────────"

compile_source() {
    local src_file=$1
    local obj_file=$2
    
    echo "  Compiling $(basename $src_file)..."
    c++ $CXX_FLAGS $INCLUDE_FLAGS -c "$src_file" -o "$obj_file"
}

compile_source "$SRC_DIR/stability_linter.cpp" "$OUTPUT_DIR/stability_linter.o"
compile_source "$SRC_DIR/patterns.cpp" "$OUTPUT_DIR/patterns.o"

echo ""

# Step 2: Compile enhanced components
echo "──────────────────────────────────────────────────────────────────"
echo "Step 2: Compiling Enhanced Components"
echo "──────────────────────────────────────────────────────────────────"

if [ -f "$SRC_DIR/transformer_analyzer.cpp" ]; then
    compile_source "$SRC_DIR/transformer_analyzer.cpp" "$OUTPUT_DIR/transformer_analyzer.o"
    TRANSFORMER_OBJ="$OUTPUT_DIR/transformer_analyzer.o"
else
    echo "  ⚠️  transformer_analyzer.cpp not found (optional)"
    TRANSFORMER_OBJ=""
fi

if [ -f "$SRC_DIR/precision_sheaf.cpp" ]; then
    compile_source "$SRC_DIR/precision_sheaf.cpp" "$OUTPUT_DIR/precision_sheaf.o"
    SHEAF_OBJ="$OUTPUT_DIR/precision_sheaf.o"
else
    echo "  ⚠️  precision_sheaf.cpp not found (optional)"
    SHEAF_OBJ=""
fi

echo ""

# Step 3: Create shared library
echo "──────────────────────────────────────────────────────────────────"
echo "Step 3: Creating Shared Library"
echo "──────────────────────────────────────────────────────────────────"

LIB_OBJECTS="$OUTPUT_DIR/stability_linter.o $OUTPUT_DIR/patterns.o"
if [ -n "$TRANSFORMER_OBJ" ]; then
    LIB_OBJECTS="$LIB_OBJECTS $TRANSFORMER_OBJ"
fi
if [ -n "$SHEAF_OBJ" ]; then
    LIB_OBJECTS="$LIB_OBJECTS $SHEAF_OBJ"
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    LIB_NAME="libstability_linter.dylib"
else
    LIB_NAME="libstability_linter.so"
fi

echo "  Creating $LIB_NAME..."
c++ $CXX_FLAGS -shared $LIB_OBJECTS -o "$OUTPUT_DIR/$LIB_NAME" $LINK_FLAGS

echo "  ✓ Created $OUTPUT_DIR/$LIB_NAME"
echo ""

# Step 4: Compile test suite
echo "──────────────────────────────────────────────────────────────────"
echo "Step 4: Compiling Test Suite"
echo "──────────────────────────────────────────────────────────────────"

if [ -f "$TESTS_DIR/test_linter.cpp" ]; then
    echo "  Compiling test_linter..."
    c++ $CXX_FLAGS $INCLUDE_FLAGS "$TESTS_DIR/test_linter.cpp" \
        -o "$OUTPUT_DIR/test_linter" \
        -L"$OUTPUT_DIR" -lstability_linter $LINK_FLAGS
    echo "  ✓ Created $OUTPUT_DIR/test_linter"
else
    echo "  ⚠️  test_linter.cpp not found"
fi

echo ""

# Step 5: Compile examples
echo "──────────────────────────────────────────────────────────────────"
echo "Step 5: Compiling Examples"
echo "──────────────────────────────────────────────────────────────────"

if [ -f "$EXAMPLES_DIR/demo_linter.cpp" ]; then
    echo "  Compiling demo_linter..."
    c++ $CXX_FLAGS $INCLUDE_FLAGS "$EXAMPLES_DIR/demo_linter.cpp" \
        -o "$OUTPUT_DIR/demo_linter" \
        -L"$OUTPUT_DIR" -lstability_linter $LINK_FLAGS
    echo "  ✓ Created $OUTPUT_DIR/demo_linter"
fi

if [ -f "$EXAMPLES_DIR/comprehensive_demo.cpp" ]; then
    echo "  Compiling comprehensive_demo..."
    c++ $CXX_FLAGS $INCLUDE_FLAGS "$EXAMPLES_DIR/comprehensive_demo.cpp" \
        -o "$OUTPUT_DIR/comprehensive_demo" \
        -L"$OUTPUT_DIR" -lstability_linter $LINK_FLAGS
    echo "  ✓ Created $OUTPUT_DIR/comprehensive_demo"
fi

echo ""

# Step 6: Build summary
echo "════════════════════════════════════════════════════════════════════"
echo "  Build Complete!"
echo "════════════════════════════════════════════════════════════════════"
echo ""
echo "Created artifacts:"
echo "  📚 Library:  $OUTPUT_DIR/$LIB_NAME"
echo "  🧪 Tests:    $OUTPUT_DIR/test_linter"
echo "  🎯 Demo:     $OUTPUT_DIR/demo_linter"

if [ -f "$OUTPUT_DIR/comprehensive_demo" ]; then
    echo "  ⭐ Enhanced: $OUTPUT_DIR/comprehensive_demo"
fi

echo ""
echo "Quick Start:"
echo "  Run tests:          ./$OUTPUT_DIR/test_linter"
echo "  Run basic demo:     ./$OUTPUT_DIR/demo_linter"
if [ -f "$OUTPUT_DIR/comprehensive_demo" ]; then
    echo "  Run full demo:      ./$OUTPUT_DIR/comprehensive_demo"
fi
echo ""

# Optional: Run tests automatically
read -p "Run tests now? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "════════════════════════════════════════════════════════════════════"
    echo "  Running Test Suite"
    echo "════════════════════════════════════════════════════════════════════"
    echo ""
    
    export DYLD_LIBRARY_PATH="$OUTPUT_DIR:$LIBTORCH_PREFIX/lib:$DYLD_LIBRARY_PATH"
    export LD_LIBRARY_PATH="$OUTPUT_DIR:$LIBTORCH_PREFIX/lib:$LD_LIBRARY_PATH"
    
    ./$OUTPUT_DIR/test_linter
fi

echo ""
echo "════════════════════════════════════════════════════════════════════"
