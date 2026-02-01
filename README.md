# HDVO: Hybrid Dense Direct Visual Odometry

<div align="center">

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1+-orange.svg)](https://pytorch.org/)

</div>

<div align="center">
  <img src="docs/imgs/inria.png" height="80"/>
</div>

<br>

Official PyTorch implementation of **HDVO** from [ACENTAURI team, INRIA](https://team.inria.fr/acentauri/).

---

## 🔥 News

- **[2026-03]** Initial code release

## ✨ Highlights

- **Hybrid Architecture**: Combines data-driven deep learning with model-based geometric constraints for robust visual odometry
- **Unified Framework**: Joint stereo depth estimation and visual odometry in a single framework
- **State-of-the-art Performance**: Achieves competitive results on KITTI Odometry and Virtual KITTI 2 benchmarks
- **Flexible Design**: Modular architecture allows easy integration and customization

## 📋 Table of Contents

- [Installation](#-installation)
- [Dataset Preparation](#-dataset-preparation)
- [Model Zoo](#-model-zoo)
- [Getting Started](#-getting-started)
- [Training](#-training)
- [Evaluation](#-evaluation)
- [Citation](#-citation)
- [License](#-license)
- [Acknowledgements](#-acknowledgements)
- [Contact](#-contact)

## 🛠️ Installation

### Prerequisites

- Python == 3.10
- PyTorch == 2.0.1
- CUDA == 11.8 (for GPU support)

### Environment Setup


```bash
# Clone the repository
git clone https://github.com/ziming-liu/hdvo.git
cd hdvo

# Create conda environment
conda create -n hdvo python=3.8
conda activate hdvo

# Install dependencies
cat requirements.txt | while read package; do    if [ -n "$package" ] && [[ ! "$package" =~ ^#.* ]]; then      echo "Installing: $package";     pip install "$package" || echo "Skip failed packages: $package";   fi; done

# Install mmcv 
cd mmcv-full-1.6.0 && python setup.py develop 

# Install the package hdvo
python install -e .

# Install openrox 
bash build_openrox.sh  $OPENROX_DIR  $HDVO_DIR

# or use compiled openrox file
$HDVO_DIR=/your_path
export LD_LIBRARY_PATH=$HDVO_DIR/openrox/cmake:$LD_LIBRARY_PATH

# rox_odometry_module.so has been included in this repo
```

For Nvidia Jetson Orion and Thor, pls switch to corresponding git branches.

## 📦 Dataset Preparation

We support the following datasets:

- **KITTI Odometry**: For visual odometry training and evaluation
- **Virtual KITTI 2**: For training and testing

Annotations for `KITTI Odometry` has already existed under `annotations/kittiodometry`.

Please refer to [prepare_dataset.md](docs/prepare_dataset.md) for detailed instructions on downloading and organizing datasets.

## 🎯 Model Zoo

### Stereo Depth Estimation

| Model | Dataset | Config | Checkpoint |
|-------|---------|--------|------------|
| HDVO | KITTI Odometry | [config](configs/hdvo) | [ckp](pretrained/iter_40000.pth) |
| HDVO | Virtual KITTI 2 | Coming soon | Coming soon
<!-- ### Quick Demo

```python
# Coming soon
``` -->

### Inference

For detailed inference instructions, please see [inference.md](docs/inference.md).


```bash
torchrun --standalone --nnodes=1 --nproc_per_node=1 --master_port=12860 \
  tools/test.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
   work_dirs/stereohdvo_posesup_coex_kittiodom_huberloss/iter_40000.pth   \
    --launcher pytorch  --test_seq_id 09 
```

### Distributed Training

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=$NUM_GPU  \
    tools/train.py configs/hdvo/stereohdvo_posesup_coex_kittiodom_huberloss.py \
      --launcher pytorch 
```

## 📄 License

This project is released under the [Apache 2.0 license](LICENSE).

## 🙏 Acknowledgements

This codebase is built upon several excellent open-source projects:

- [MMAction](https://github.com/open-mmlab/mmaction) - Framework structure and utilities
- [OpenRox](https://github.com/robocortex/openrox) - Computer vision algorithms

We thank the authors for their great work and open-source contributions.

## 📧 Contact

For questions and discussions, please contact:

- **Ziming Liu**: liuziming.email@gmail.com

You can also open an issue in this repository for bug reports and feature requests.

---

<div align="center">
Made with ❤️ by ACENTAURI team @ INRIA
</div> 