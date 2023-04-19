'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-04-18 17:26:24
LastEditors: Ziming Liu
LastEditTime: 2023-04-18 23:16:21
'''

from setuptools import setup
import os
import glob
from torch.utils.cpp_extension import BuildExtension, CppExtension,CUDAExtension

# root forlder
include_dirs = os.path.dirname(os.path.abspath(__file__))
print(os.path.join(include_dirs, 'build_image_volume', './build_image_volume_cuda.cpp'))

setup(
    name='build_image_volume_cuda',  # import in python
    version="0.1",
    ext_modules=[
        CUDAExtension('build_image_volume_cuda', [
            os.path.join( 'build_image_volume', 'build_image_volume_cuda.cpp'),
            os.path.join( 'build_image_volume', 'build_image_volume_kernel.cu'),
        ])
    ],

    cmdclass={
        'build_ext': BuildExtension#.with_options(use_ninja=False)
    }
)
