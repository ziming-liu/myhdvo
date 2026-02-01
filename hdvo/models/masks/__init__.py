"""
Mask modules for HDVO.

This module contains different types of mask generators for filtering
out unreliable regions in optical flow and depth estimation.

Author: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-05-07
Last Modified: 2024-01-29
"""
from .homo_mask_fast import *
from .homo_mask_sam import *
from .homo_mask_tinysam import *
from .stc_mask_fast import *
from .stc_mask_sam import *
from .stc_mask_sam2 import *

__all__ = [
    'HomoMaskFast',
    'HomoMaskSAM',
    'HomoMaskTinySAM',
    'STCMaskFast',
    'STCMaskFast2',
    'STCMaskSAM',
    'STCMaskSAM2',
]