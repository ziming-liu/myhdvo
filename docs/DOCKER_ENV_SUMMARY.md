# HDVO Environment Setup - Docker Container (Thor JetPack 7.0)

## ✅ Successfully Installed Components

### Core Framework
- **PyTorch**: 2.8.0a0+34c6371d24.nv25.08 (CUDA 13.0)
- **TorchVision**: 0.23.0a0+428a54c9
- **MMCV-Full**: 1.6.0 (installed from source)

### Compute & ML Libraries
- **NumPy**: 1.26.4 (downgraded for compatibility)
- **SciPy**: 1.15.3
- **Scikit-learn**: 1.6.1
- **Pandas**: 2.2.3

### Computer Vision
- **OpenCV**: 4.13.0.90 (opencv-python)
- **Pillow**: 11.3.0
- **Albumentations**: 2.0.8
- **Scikit-image**: 0.26.0
- **ImageIO**: 2.37.2

### Model Utilities
- **TIMM**: 1.0.24
- **Einops**: 0.8.1

### Training & Logging
- **TensorBoard**: 2.16.2
- **tqdm**: 4.67.1
- **Matplotlib**: 3.10.5

### Utilities
- **PyYAML**: 6.0.2
- **Fire**: 0.5.0
- **Addict**: 2.4.0
- **YAPF**: 0.43.0
- **pytest**: 8.1.1
- **PrettyTable**: 3.9.0
- **TerminalTables**: 3.1.10

## 🎯 GPU Configuration

```python
import torch
print(f"CUDA Available: {torch.cuda.is_available()}")  # True
print(f"GPU: {torch.cuda.get_device_name(0)}")        # NVIDIA Thor
print(f"CUDA Version: {torch.version.cuda}")          # 13.0
```

## 📦 Container Information

- **Base Image**: nvcr.io/nvidia/pytorch:25.08-py3
- **OS**: Ubuntu 24.04
- **Python**: 3.12
- **Workspace**: `/workspace` (mounted from `/home/bosch_driving/hdvo`)

## 🚀 Usage

### Start Container
```bash
cd /home/bosch_driving/hdvo
./run_docker_thor.sh
```

### Inside Container
All HDVO code is available at `/workspace`:
```bash
cd /workspace
python tools/test.py  # Your HDVO scripts
```

### Verify Environment
```bash
python -c "import torch, mmcv; print(f'PyTorch: {torch.__version__}, MMCV: {mmcv.__version__}')"
```

## ⚠️ Known Issues & Solutions

### 1. NumPy Version Conflict
**Issue**: Some packages require NumPy 2.x, but PyTorch in container requires <2.0

**Solution**: NumPy 1.26.4 is installed (compatible with PyTorch)

### 2. HDVO Package Installation Failed
**Issue**: `pip install -e .` requires torch to be imported during setup

**Solution**: Install dependencies manually as shown above, or run:
```bash
cd /workspace
pip install --no-build-isolation -e .
```

### 3. Network Access in Container
**Issue**: External network may not work without proxy

**Solution**: Use pre-installed packages or configure proxy:
```bash
export http_proxy=http://rb-proxy-unix-apac.bosch.com:8080
export https_proxy=http://rb-proxy-unix-apac.bosch.com:8080
```

## 📝 Files Created

1. **run_docker_thor.sh** - Start Docker container with all configs
2. **install_hdvo_deps_in_docker.sh** - Install HDVO dependencies
3. **docs/INSTALL_THOR.md** - Complete installation guide
4. **docs/DOCKER_ENV_SUMMARY.md** - This file

## 🔧 Additional Setup (Optional)

### Install Remaining HDVO Requirements
```bash
cd /workspace
pip install mmengine lightning thop setuptools-scm ultralytics
```

### Install from requirements_inference.txt
```bash
pip install -r requirements_inference.txt
```

## 🎉 Environment Ready!

Your HDVO development environment in Docker is ready to use with:
- Full GPU support (NVIDIA Thor)
- PyTorch 2.8.0 with CUDA 13.0
- MMCV-Full 1.6.0
- All major dependencies installed

Start developing with:
```bash
cd /workspace
python your_script.py
```
