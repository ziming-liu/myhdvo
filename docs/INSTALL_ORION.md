

create python env 

```bash
conda create -n hdvo python==3.8 

```

install pytorch jetson version, refer to [install-pytorch-jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html#overview__section_orin)


`v512` is the jetpack version, which is required.  

```bash
 export TORCH_INSTALL=https://developer.download.nvidia.cn/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl


 python3 -m pip install --upgrade pip; python3 -m pip install numpy==1.26.1; python3 -m pip install --no-cache $TORCH_INSTALL


```
install mmcv-full 1.6.0

```bash
cd mmcv-full-1.6.0 && python setup.py develop 
```

install other packages 

```bash
 source $(conda info --base)/etc/profile.d/conda.sh && conda activate hdvo && cat requirements_clean.txt | while read package; do    if [ -n "$package" ] && [[ ! "$package" =~ ^#.* ]]; then      echo "Installing: $package";     pip install "$package" || echo "Skip failed packages: $package";   fi; done

```

install torchvision

```bash
git clone https://github.com/pytorch/vision.git
cd vision

git checkout tags/v0.19.0

pip install -e .
```

