#!/bin/bash
# OpenROX Compilation Script for ARM64
# This script compiles OpenROX and the odometry module for ARM architecture

set -e  # Exit on error

OPENROX_DIR=$1 
HDVO_DIR=$2 

echo "=========================================="
echo "Building OpenROX for ARM64"
echo "=========================================="

# Build OpenROX library
mkdir -p "$OPENROX_DIR/build"
cd "$OPENROX_DIR/build"
echo "Cleaning build directory..."
rm -rf *

echo "Running CMake..."
cmake ..

echo "Building OpenROX library..."
make -j$(nproc)

echo "✓ OpenROX library built successfully!"
ls -lh libopenrox.so

# Build odometry module
echo ""
echo "=========================================="
echo "Building rox_odometry_module"
echo "=========================================="

cd "$OPENROX_DIR/examples"

echo "Compiling rox_odometry_module.c..."
gcc -c -fPIC rox_odometry_module.c \
    -I"$OPENROX_DIR/sources" \
    -I"$OPENROX_DIR/build" \
    -o rox_odometry_module.o

echo "Creating shared library..."
gcc -shared -o librox_odometry_module.so rox_odometry_module.o \
    -L"$OPENROX_DIR/build" \
    -lopenrox \
    -Wl,-rpath,'$ORIGIN' \
    -lm -lpthread

echo "✓ rox_odometry_module built successfully!"
ls -lh librox_odometry_module.so

# Copy to HDVO directory
echo ""
echo "=========================================="
echo "Copying libraries to HDVO directory"
echo "=========================================="

cp "$OPENROX_DIR/build/libopenrox.so" "$HDVO_DIR/"
cp librox_odometry_module.so "$HDVO_DIR/rox_odometry_module.so"

echo "✓ Libraries copied to $HDVO_DIR"
ls -lh "$HDVO_DIR"/*.so

echo ""
echo "=========================================="
echo "Build completed successfully!"
echo "=========================================="
echo "OpenROX library: $HDVO_DIR/libopenrox.so"
echo "Odometry module: $HDVO_DIR/rox_odometry_module.so"
echo ""
echo "You can now run your HDVO tests."