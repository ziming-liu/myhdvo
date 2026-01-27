#!/bin/bash
# Verify PyTorch installation in Docker container

echo "======================================================"
echo "Testing PyTorch in Docker Container"
echo "======================================================"

sudo docker run --rm \
  --network=host \
  --runtime nvidia \
  --gpus all \
  -v $PWD:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:25.08-py3 \
  python3 << 'PYEOF'
import torch
import sys

print("\n" + "="*60)
print("PyTorch Environment Information")
print("="*60)
print(f"PyTorch version:     {torch.__version__}")
print(f"Python version:      {sys.version.split()[0]}")
print(f"CUDA available:      {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:        {torch.version.cuda}")
    print(f"cuDNN version:       {torch.backends.cudnn.version()}")
    print(f"GPU count:           {torch.cuda.device_count()}")
    print(f"GPU name:            {torch.cuda.get_device_name(0)}")
    print(f"GPU capability:      {torch.cuda.get_device_capability(0)}")
    
    print("\n" + "="*60)
    print("Running GPU Test...")
    print("="*60)
    
    # Test GPU computation
    x = torch.rand(5000, 5000, device="cuda")
    y = torch.rand(5000, 5000, device="cuda")
    z = torch.matmul(x, y)
    
    print(f"✓ Matrix multiplication on GPU successful!")
    print(f"  Result sum: {z.sum().item():.2f}")
    print(f"  Tensor shape: {z.shape}")
    
    # Memory test
    print(f"\nGPU Memory:")
    print(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB")
    print(f"  Reserved:  {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB")
else:
    print("\n⚠ WARNING: CUDA is not available!")
    print("Check GPU drivers and runtime configuration.")

print("="*60)
print("Test Complete!")
print("="*60 + "\n")
PYEOF

echo ""
echo "======================================================"
echo "Docker PyTorch verification complete!"
echo "======================================================"
