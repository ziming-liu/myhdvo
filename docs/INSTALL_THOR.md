# Installation Guide for Thor Device

## Environment Setup for Thor (JetPack 7.0)

Reference: [eLinux Jetson AI Stack - JetPack 7.0](https://elinux.org/Jetson/L4T/Jetson_AI_Stack#JetPack_7.0)

### System Information

- **Device**: NVIDIA Jetson Thor
- **JetPack**: 7.0 (L4T 38.2)
- **CUDA**: 13.0
- **cuDNN**: 9.12
- **TensorRT**: 10.13
- **Python**: 3.12 (for JetPack 7.0)
- **Compute Capability**: sm_110

### Check JetPack Version

```bash
# Check your JetPack version first
apt show nvidia-jetpack
```

Expected output: `Version: 7.0` or similar (L4T 38.2)

### GPU Verification

```bash
# Check GPU with nvidia-smi
watch -n 1 nvidia-smi

# Or use jtop (community version for Thor)
git clone https://github.com/eous/jetson_stats.git
cd jetson_stats/
sudo pip3 install . --break-system-packages
sudo systemctl start jtop.service
# Please logout or reboot
jtop
```

## PyTorch Installation Methods

JetPack 7.0 supports two installation methods:

### Method 1: Native Installation (Recommended for Development)

**Important**: JetPack 7.0 uses **Python 3.12**, not Python 3.10!

#### Step 1: Install NVPL (Required for PyTorch)

```bash
# Download and install NVPL
wget https://developer.download.nvidia.com/compute/nvpl/25.5/local_installers/nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb
sudo dpkg -i nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb
sudo cp /var/nvpl-local-repo-ubuntu2404-25.5/nvpl-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install nvpl
rm nvpl-local-repo-ubuntu2404-25.5_1.0-1_arm64.deb
```

#### Step 2: Create Python Virtual Environment

```bash
# Install venv
sudo apt install -y python3.12-venv

# Create virtual environment
python3 -m venv ~/hdvo_env
source ~/hdvo_env/bin/activate
```

#### Step 3: Install PyTorch and TorchVision

```bash
# Download PyTorch 2.9.0 wheel (CUDA 13.0, Python 3.12)
wget https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/a96/474a2e6e0f0a3/torch-2.9.0.dev20250827+cu130.g378edb0-cp312-cp312-linux_aarch64.whl

# Download TorchVision 0.24.0 wheel
wget https://pypi.jetson-ai-lab.io/sbsa/cu130/+f/c26/fb4d05c0d694c/torchvision-0.24.0-cp312-cp312-linux_aarch64.whl

# Install PyTorch
pip3 install torch-2.9.0.dev20250827+cu130.g378edb0-cp312-cp312-linux_aarch64.whl

# Install TorchVision
pip3 install torchvision-0.24.0-cp312-cp312-linux_aarch64.whl

# Clean up
rm torch-2.9.0.dev20250827+cu130.g378edb0-cp312-cp312-linux_aarch64.whl
rm torchvision-0.24.0-cp312-cp312-linux_aarch64.whl
```

#### Step 4: Verify Installation

```bash
python3 << EOF
import torch
import torchvision
print("PyTorch version:", torch.__version__)
print("TorchVision version:", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA version:", torch.version.cuda)
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.rand(1000, 1000, device="cuda")
    print("Tensor sum (GPU test):", x.sum().item())
EOF
```

Expected output:
```
PyTorch version: 2.9.0.dev20250827+cu130.g378edb0
TorchVision version: 0.24.0
CUDA available: True
CUDA version: 13.0
GPU name: NVIDIA Thor
Tensor sum (GPU test): 499xxx.x
```

### Method 2: Docker Container Installation

#### Step 1: Install Docker (if not already installed)

```bash
# Download and run Docker installation script
wget https://raw.githubusercontent.com/AastaNV/JetPack_7.0/refs/heads/main/install_docker_Jetpack7.0.sh
chmod +x install_docker_Jetpack7.0.sh
sudo ./install_docker_Jetpack7.0.sh
```

#### Step 2: Run PyTorch Container

```bash
# Run NVIDIA PyTorch container (25.08)
sudo docker run --rm -it \
  --network=host \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility,video,graphics \
  --runtime nvidia \
  --privileged \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /etc/X11:/etc/X11 \
  --device /dev/nvhost-vic \
  -v /dev:/dev \
  -v $PWD:/workspace \
  -w /workspace \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  nvcr.io/nvidia/pytorch:25.08-py3
```

#### Step 3: Verify in Container

```bash
# Inside the container, verify PyTorch
python3 << 'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.rand(10000, 10000, device="cuda")
    print("Tensor sum:", x.sum().item())
EOF
```

**Container Details**:
- Image: `nvcr.io/nvidia/pytorch:25.08-py3`
- OS: Ubuntu 24.04
- Python: 3.12
- PyTorch: 2.8.0a0+34c6371
- TorchVision: 0.23.0

## Install HDVO Dependencies

### Option A: In Native Environment

```bash
# Activate virtual environment
source ~/hdvo_env/bin/activate

# Navigate to HDVO directory
cd /home/bosch_driving/hdvo

# Install basic dependencies
pip install --upgrade pip
pip install numpy opencv-python matplotlib

# Install requirements (skip incompatible packages)
cat requirements_clean.txt | while read package; do
  if [ -n "$package" ] && [[ ! "$package" =~ ^#.* ]]; then
    echo "Installing: $package"
    pip install "$package" || echo "Warning: Failed to install $package, skipping..."
  fi
done
```

### Option B: In Docker Container

```bash
# Inside the container
cd /workspace

# Install additional dependencies
pip install -r requirements_clean.txt

```


### InstallSuccessfully installed mmcv-full-1.7.2

**Note**: MMCV 1.6.0 was built for older PyTorch versions. You may need to adapt or use a compatible version.

```bash
cd /home/bosch_driving/hdvo/mmcv-full-1.6.0

# Try to install (may require modifications for PyTorch 2.9)
python setup.py develop

# If it fails, you may need to:
# 1. Use a newer MMCV version compatible with PyTorch 2.9
# 2. Or install MMCV from PyPI
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch2.0.0/index.html
```

### Install OpenCV with CUDA (Optional, Native Installation)

For better performance with CUDA acceleration:

```bash
# Download OpenCV installation script for JetPack 7.0
wget https://raw.githubusercontent.com/AastaNV/JetPack_7.0/refs/heads/main/install_opencv4.12.0_Jetpack7.0.sh
sudo chmod +x install_opencv4.12.0_Jetpack7.0.sh
./install_opencv4.12.0_Jetpack7.0.sh

# Verify OpenCV CUDA support
opencv_version --verbose | grep NVIDIA
python -c "import cv2; print(cv2.getBuildInformation())" | grep NVIDIA
```

Expected output should show:
```
NVIDIA CUDA:                   YES (ver 13.0, CUFFT CUBLAS)
NVIDIA GPU arch:             110
```

## Performance Optimization

### Enable Max Performance Mode

```bash
# Set maximum power mode
sudo nvpmodel -m 0

# Lock clocks to maximum frequency
sudo jetson_clocks
```

### Monitor GPU Usage

```bash
# Monitor with nvidia-smi
watch -n 1 nvidia-smi

# Monitor system stats
sudo tegrastats

# Use jtop (if installed)
jtop
```

### GPU Stress Test (Optional)

```bash
# Clone CUDA samples
git clone -b v13.0 https://github.com/NVIDIA/cuda-samples
cd cuda-samples/

# Download and apply stress test patch
wget https://raw.githubusercontent.com/AastaNV/JetPack_7.0/refs/heads/main/patch/matrixMulCUBLAS_stree.patch
git apply matrixMulCUBLAS_stree.patch

# Build and test
cd Samples/4_CUDA_Libraries/matrixMulCUBLAS/
cmake . && make
./matrixMulCUBLAS

# You should be able to reach ~112W with tegrastats
```

## Additional AI Libraries (Optional)

### Install vLLM (for LLM inference)

```bash
# Use vLLM container
sudo docker run --ipc=host --net=host --gpus=all --runtime=nvidia \
  --privileged -it --rm -u 0:0 --name=vllm \
  nvcr.io/nvidia/tritonserver:25.08-vllm-python-py3

# Inside container, install venv
apt update && apt install -y python3.12-venv
python3 -m venv env --system-site-packages && source env/bin/activate
```

### Python Package Index for Jetson

Additional pre-built packages for JetPack 7.0+Thor:
- https://pypi.jetson-ai-lab.io/

## Troubleshooting

### CUDA Toolkit Issues

```bash
# Check CUDA installation
nvcc --version

# Should show CUDA 13.0
# CUDA should be in: /usr/local/cuda-13.0/

# Set environment variables if needed
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

### Docker Permission Issues

If you get permission errors with Docker (Exit Code: 126):

```bash
# Add user to docker group
sudo usermod -aG docker $USER

# Reboot or logout/login
sudo reboot

# Verify Docker installation
docker run --rm hello-world
```

### Memory Issues

Thor has 128GB RAM, but monitor usage during installation:

```bash
# Monitor memory
watch -n 1 free -h

# Or use htop
sudo apt install htop
htop
```

### Python Version Mismatch

**Important**: JetPack 7.0 requires Python 3.12, not 3.10 or 3.8!

```bash
# Check Python version
python3 --version

# Should output: Python 3.12.x

# If using conda with Python 3.10, this won't work for JetPack 7.0
# Use system Python 3.12 instead with venv
```

### MMCV Build Issues

If MMCV fails to build:

```bash
# Install build dependencies
sudo apt-get update
sudo apt-get install -y build-essential python3-dev ninja-build

# Install compatible MMCV version
pip install mmcv-full -f https://download.openmmlab.com/mmcv/dist/index.html
```

### Wheel Compatibility Issues

If wheel files fail to install:

```bash
# Check platform compatibility
python3 << EOF
import sysconfig
print("Platform:", sysconfig.get_platform())
print("Python version:", sysconfig.get_python_version())
EOF

# For JetPack 7.0 Thor, you need:
# - cp312 (CPython 3.12)
# - linux_aarch64 (ARM64 Linux)
```

### Next Steps

After successful installation, refer to:
- [Inference Guide](INFERENCE_QUICKSTART.md)
- [Training Guide](train.md)
- [Dataset Preparation](prepare_dataset.md)

---

**Device Info**: Thor (JetPack 7.0)  
**Python**: 3.10  
**PyTorch**: 2.5.0+ (compatible with JP 7.0)  
**MMCV**: 1.6.0
