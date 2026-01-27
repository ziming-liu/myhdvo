#!/bin/bash
# Activation script for HDVO environment on Thor (JetPack 7.0)

# Set CUDA paths
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu/nvidia:/usr/lib/aarch64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate hdvo

echo "========================================"
echo "HDVO Environment Activated (Thor)"
echo "========================================"
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo ""
echo "Testing PyTorch..."
python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')" 2>&1 | head -10
echo "========================================"
