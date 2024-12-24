<!--
 * @Developer: ACENTAURI team, INRIA institute
 * @Author: Ziming Liu
 * @Date: 2024-02-02 15:07:46
 * @LastEditors: Ziming Liu
 * @LastEditTime: 2024-02-11 12:39:11
-->
# Running environment

This codebase is only tested on Nvidia A40, 2080Ti GPUs with Linux 3.10.0-693.17.1.el7.x86_64 centos 7.4.1708. 

# software

Install these software
- Conda (any version)
- CUDA 11.1 (other version is not promised)
- cudnn (version aligned with cuda 11.1)
- cmake
- gcc


# recover python envrionment

Use Conda to recover the environment with this file. 

```
 conda env create -f docs/hdvo_env02022024.yml
```

activate python environment 

```
source activate  xxx 

source deactivate xxx 
```

# install this code as a python library

run 

```

python setup.py develop 

```

