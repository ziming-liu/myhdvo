from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension,CUDAExtension

# root forlder
include_dirs = os.path.dirname(os.path.abspath(__file__))
# source C++ 
source_cpu = glob.glob(os.path.join(include_dirs, 'esm_grid_sample', '*.cpp'))
#source_cpu = glob.glob(os.path.join(include_dirs, 'esm_grid_sample', '*.cu'))

setup(
    name='esm_grid_sample',  # import in python
    version="0.1",
    ext_modules=[
        CppExtension('esm_grid_sample_cpp', sources=[os.path.join(include_dirs, 'esm_grid_sample', './esm_grid_sample.cpp'),], include_dirs=[include_dirs]),
        CUDAExtension('esm_grid_sample_cuda', [
            os.path.join(include_dirs, 'esm_grid_sample', './esm_grid_sample_cuda.cpp'),
            os.path.join(include_dirs, 'esm_grid_sample', './esm_grid_sample_kernel.cu'),
        ])
    ],

    cmdclass={
        'build_ext': BuildExtension
    }
)

# moudule load gcc/7.2
# module load cuda 
