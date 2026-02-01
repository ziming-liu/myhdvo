#!/bin/bash
# Install HDVO dependencies in Docker container
# This script should be run INSIDE the Docker container

echo "=========================================="
echo "Installing HDVO Dependencies in Container"
echo "=========================================="
echo ""

# Set proxy (if needed)
# export http_proxy=http://rb-proxy-unix-apac.bosch.com:8080
# export https_proxy=http://rb-proxy-unix-apac.bosch.com:8080

echo "Checking pre-installed packages..."
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'Not installed')"
echo "NumPy: $(python -c 'import numpy; print(numpy.__version__)' 2>/dev/null || echo 'Not installed')"
echo "OpenCV: $(python -c 'import cv2; print(cv2.__version__)' 2>/dev/null || echo 'Not installed')"
echo ""

echo "Installing packages without network dependency..."
pip install --no-deps addict==2.4.0 || true
pip install --no-deps pyyaml==6.0 || true
pip install --no-deps fire==0.5.0 || true
pip install --no-deps terminaltables==3.1.10 || true
pip install --no-deps prettytable==3.9.0 || true

echo ""
echo "Installing packages that need dependencies..."
pip install einops timm scikit-learn pandas scikit-image imageio albumentations yapf pytest || echo "Some packages failed, continuing..."

echo ""
echo "Installing mmcv-full from local source..."
cd /workspace/mmcv-full-1.6.0
MMCV_WITH_OPS=1 pip install -e . || echo "Warning: mmcv-full installation failed"

echo ""
echo "Installing hdvo package..."
cd /workspace
pip install -e . || echo "Warning: hdvo installation failed"

echo ""
echo "=========================================="
echo "Installation Summary"
echo "=========================================="
python << 'EOF'
import sys
packages = [
    'torch', 'torchvision', 'numpy', 'scipy', 'matplotlib',
    'cv2', 'PIL', 'tqdm', 'addict', 'yaml', 'fire',
    'einops', 'timm', 'sklearn', 'pandas', 'mmcv', 'mmengine'
]

print("\nPackage Status:")
print("-" * 50)
for pkg in packages:
    try:
        if pkg == 'cv2':
            import cv2 as module
        elif pkg == 'PIL':
            import PIL as module
        elif pkg == 'yaml':
            import yaml as module
        elif pkg == 'sklearn':
            import sklearn as module
        else:
            module = __import__(pkg)
        version = getattr(module, '__version__', 'installed')
        print(f"✓ {pkg:20s} : {version}")
    except ImportError:
        print(f"✗ {pkg:20s} : NOT INSTALLED")
print("-" * 50)
EOF

echo ""
echo "Done! You can now use HDVO in this container."
