# Docker Build and Run Instructions for Thor

## Build the Docker Image

```bash
# Navigate to the HDVO directory
cd /home/bosch_driving/hdvo

# Build the Docker image (this may take 15-30 minutes)
sudo docker build -f Dockerfile.thor -t hdvo:thor .
```

## Run the Container

### Option 1: Interactive Shell (Development)

```bash
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
  hdvo:thor
```

### Option 2: With Named Container (Persistent)

```bash
sudo docker run -it \
  --name hdvo-thor \
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
  hdvo:thor
```

### Option 3: Using the Script

```bash
# Use the provided run script (modify it to use hdvo:thor instead)
./run_docker_thor.sh
```

## Verify Installation Inside Container

Once inside the container, run:

```bash
# Verify PyTorch and CUDA
python3 << 'EOF'
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

# Verify MMCV
python3 -c "import mmcv; print('MMCV version:', mmcv.__version__)"

# Verify other packages
python3 << 'EOF'
import albumentations
import skimage
import mmengine
import timm
import ruamel.yaml
print("All packages imported successfully!")
EOF
```

## Important Notes

### Volume Mounting
- The `-v $PWD:/workspace` mounts your current directory to `/workspace` in the container
- Any changes made in `/workspace` will persist on your host machine
- To mount a different directory: `-v /path/to/your/data:/workspace`

### GPU Access
- `--gpus all` enables access to all GPUs
- `--runtime nvidia` uses NVIDIA Container Runtime
- Verify GPU access with: `nvidia-smi` inside the container

### Known Issues and Fixes

#### 1. Torch Checkpoint Loading Warning
If you encounter warnings about `weights_only=True`, modify the checkpoint loading function:

```bash
# Inside container
vim /usr/local/lib/python3.12/dist-packages/mmcv/runner/checkpoint.py

# Change torch.load() calls to include weights_only=False
# torch.load(filename, map_location=map_location, weights_only=False)
```

#### 2. Numpy Compatibility
The Dockerfile installs `numpy==1.26.0` for compatibility. If you encounter issues:

```bash
pip install --force-reinstall numpy==1.26.0
```

## Saving Your Container State

### Method 1: Commit Changes
```bash
# In another terminal (while container is running)
sudo docker commit hdvo-thor hdvo:thor-updated

# Your changes are now saved in hdvo:thor-updated
```

### Method 2: Export Image
```bash
# Save image to tar file
sudo docker save hdvo:thor -o hdvo-thor.tar

# Load on another machine
sudo docker load -i hdvo-thor.tar
```

## Container Management

```bash
# List running containers
sudo docker ps

# List all containers (including stopped)
sudo docker ps -a

# Start a stopped container
sudo docker start hdvo-thor

# Attach to a running container
sudo docker attach hdvo-thor

# Execute command in running container
sudo docker exec -it hdvo-thor bash

# Stop container
sudo docker stop hdvo-thor

# Remove container
sudo docker rm hdvo-thor

# Remove image
sudo docker rmi hdvo:thor
```

## Performance Tips

Before running the container, maximize Thor performance:

```bash
# Set maximum power mode
sudo nvpmodel -m 0

# Lock clocks to maximum frequency
sudo jetson_clocks

# Monitor GPU (in separate terminal)
watch -n 1 nvidia-smi
```

## Troubleshooting

### Docker Permission Error
```bash
sudo usermod -aG docker $USER
# Then logout/login or reboot
```

### CUDA Not Available
```bash
# Verify nvidia-docker runtime is installed
docker run --rm --gpus all nvidia/cuda:13.0-base nvidia-smi
```

### Out of Memory During Build
```bash
# Build with limited concurrent jobs
sudo docker build --build-arg MAKEFLAGS="-j4" -f Dockerfile.thor -t hdvo:thor .
```

### Package Installation Failures
```bash
# Build with verbose output
sudo docker build --progress=plain --no-cache -f Dockerfile.thor -t hdvo:thor .
```

## Next Steps

After successful container launch:
1. Verify all installations (see "Verify Installation" section above)
2. Prepare datasets: See [prepare_dataset.md](prepare_dataset.md)
3. Run inference: See [INFERENCE_QUICKSTART.md](INFERENCE_QUICKSTART.md)
4. Training: See [train.md](train.md)

---
**Device**: NVIDIA Jetson Thor (JetPack 7.0)  
**Base Image**: nvcr.io/nvidia/pytorch:25.08-py3  
**CUDA**: 13.0 | **Python**: 3.12 | **PyTorch**: 2.8.0


# other problmes

If you have safe checkpoints loading error. 

pls modify torch load checkpoint from local function in the following file, give an additional input `weights_only=False`

```bash
vim /usr/local/lib/python3.12/dist-packages/mmcv/runner/checkpoint.py
```
