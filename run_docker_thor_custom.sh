#!/bin/bash
# Run HDVO Docker container (custom built image for Thor)

echo "======================================================"
echo "Starting HDVO Container for NVIDIA Jetson Thor"
echo "======================================================"
echo "Container: jetson_thor_hdvo:py312_torch26"
echo "Workspace: $PWD -> /workspace"
echo "Python: 3.12 | PyTorch: 2.8.0 | CUDA: 13.0"
echo ""
echo "To verify PyTorch in container, run:"
echo "  python3 -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "To exit container: type 'exit'"
echo "======================================================"
echo ""

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
  jetson_thor_hdvo:py312_torch26
