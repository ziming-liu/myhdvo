#!/bin/bash
# Build Docker image for HDVO on Thor

set -e

echo "======================================================"
echo "Building HDVO Docker Image for NVIDIA Jetson Thor"
echo "======================================================"
echo "Base Image: nvcr.io/nvidia/pytorch:25.08-py3"
echo "Target Image: hdvo:thor"
echo ""
echo "This build process will:"
echo "  1. Install system dependencies (X11, OpenGL, etc.)"
echo "  2. Install mmcv-full==1.7.2"
echo "  3. Install Python packages (albumentations, mmengine, etc.)"
echo "  4. Install all requirements from requirements_thor.txt"
echo ""
echo "Estimated build time: 15-30 minutes"
echo "======================================================"
echo ""

# Check if Dockerfile exists
if [ ! -f "Dockerfile.thor" ]; then
    echo "Error: Dockerfile.thor not found!"
    echo "Please run this script from the HDVO root directory."
    exit 1
fi

# Check if requirements_thor.txt exists
if [ ! -f "requirements_thor.txt" ]; then
    echo "Error: requirements_thor.txt not found!"
    echo "Please ensure requirements_thor.txt is in the current directory."
    exit 1
fi

# Build the Docker image
echo "Starting Docker build..."
echo ""

sudo docker build \
    --progress=plain \
    -f Dockerfile.thor \
    -t hdvo:thor \
    .

echo ""
echo "======================================================"
echo "Build completed successfully!"
echo "======================================================"
echo ""
echo "Image name: hdvo:thor"
echo ""
echo "To run the container:"
echo "  ./run_docker_thor_custom.sh"
echo ""
echo "Or manually:"
echo '  sudo docker run --rm -it \'
echo '    --network=host \'
echo '    --runtime nvidia \'
echo '    --gpus all \'
echo '    -v $PWD:/workspace \'
echo '    -w /workspace \'
echo '    hdvo:thor'
echo ""
echo "To verify installation:"
echo '  sudo docker run --rm --gpus all hdvo:thor python3 -c "import torch; print(torch.cuda.is_available())"'
echo ""
echo "======================================================"
