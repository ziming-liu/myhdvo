# Dockerfile for HDVO on NVIDIA Jetson Thor (JetPack 7.0)
# Base image: PyTorch 25.08 with Python 3.12, CUDA 13.0
FROM nvcr.io/nvidia/pytorch:25.08-py3

# Set working directory
WORKDIR /workspace

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Update package lists and install system dependencies
RUN apt update && apt install -y \
    libxcb1 \
    libx11-6 \
    libxrender1 \
    libxrandr2 \
    libxinerama1 \
    libxi6 \
    libgl1 \
    python3-tk \
    && rm -rf /var/lib/apt/lists/*

# Install MMCV-Full
RUN pip install --no-cache-dir mmcv-full==1.7.2

# Install core Python packages
RUN pip install --no-cache-dir \
    albumentations \
    scikit-image \
    mmengine \
    timm \
    ruamel.yaml

# Install specific NumPy version (required for compatibility)
RUN pip install --no-cache-dir numpy==1.26.0

# Copy requirements file
COPY requirements_thor.txt /workspace/requirements_thor.txt

# Install all dependencies from requirements_thor.txt
RUN pip install --no-cache-dir -r requirements_thor.txt

# Copy the entire HDVO project
COPY . /workspace/

# Set the default command
CMD ["/bin/bash"]

# Metadata
LABEL maintainer="HDVO Team"
LABEL description="HDVO Development Environment for NVIDIA Jetson Thor with JetPack 7.0"
LABEL cuda.version="13.0"
LABEL python.version="3.12"
LABEL pytorch.version="2.8.0"
