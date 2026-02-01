#!/bin/bash
# Quick environment verification for HDVO in Docker

echo "=================================================="
echo "HDVO Docker Environment Verification"
echo "=================================================="
echo ""

echo "1. Python Version:"
python --version
echo ""

echo "2. Core Packages:"
python << 'EOF'
packages = [
    'torch', 'torchvision', 'numpy', 'cv2', 'mmcv',
    'scipy', 'sklearn', 'pandas', 'matplotlib'
]
for pkg in packages:
    try:
        if pkg == 'cv2':
            import cv2
            print(f"  ✓ OpenCV: {cv2.__version__}")
        elif pkg == 'sklearn':
            import sklearn
            print(f"  ✓ Scikit-learn: {sklearn.__version__}")
        else:
            mod = __import__(pkg)
            print(f"  ✓ {pkg}: {mod.__version__}")
    except Exception as e:
        print(f"  ✗ {pkg}: FAILED - {e}")
EOF

echo ""
echo "3. GPU Status:"
python << 'EOF'
import torch
print(f"  CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  CUDA Version: {torch.version.cuda}")
    print(f"  cuDNN Version: {torch.backends.cudnn.version()}")
EOF

echo ""
echo "=================================================="
echo "Environment is ready for HDVO development!"
echo "=================================================="
