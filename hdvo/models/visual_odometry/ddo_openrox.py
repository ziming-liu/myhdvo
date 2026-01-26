'''
Author: Ziming Liu
Date: 2022-06-24 15:25:59
LastEditors: Ziming Liu
LastEditTime: 2024-02-02 21:12:00
Description: Python function to load OPENROX Dense direct odometry with openrox library. (C code) 
'''
from ctypes import *
import numpy as np
import torch
import os
from ..registry import VISUAL_ODOMETRY
from ...core import vis_depth_tensor
""" 
# these are the data structor used in OPENROX code. 
ROX_TYPE_UCHAR, ROX_TYPE_SCHAR, ROX_TYPE_USHORT, ROX_TYPE_SSHORT, ROX_TYPE_UINT, \
    ROX_TYPE_FLOAT, ROX_TYPE_SINT, ROX_TYPE_ULONG, ROX_TYPE_SLINT, ROX_TYPE_DOUBLE = \
        4, 6, 8, 10, 16, 17, 18, 32, 34, 33

class Rox_Array_Struct(Structure):
    _fields_ = [ ("system_ptr", c_void_p),
                    ("base_ptr",c_void_p),
                    ("datatype", c_int),
                    ("nbblocks", c_uint),
                    ("length", c_size_t),
                    ("datatype", c_int),
                    ("reference_count",c_uint),
                    ("owndata", c_bool)
                ]

class Rox_Array2D_Struct(Structure):
    _fields_ = [("cols", c_int), # num of cols
                ("rows", c_int), # num of rows
                ("step", c_int), # num of bytes used per row , including padding 
                ("nbblocks", c_uint), # num of blocks, (what's block???)
                ("align_shift", c_uint), # 
                ("data", Rox_Array_Struct),
                ("rows_ptr", POINTER(c_void_p)),
                ("block_ptr", POINTER(POINTER(c_void_p)))]
"""
@VISUAL_ODOMETRY.register_module()
class DirectVO_OpenRox:
    def __init__(self, so_file_path=None, ifmask=1, disp_log=0, ifrobust=0, **kwargs ):
        '''
        description: ifmask: control use custom mask or not;
                    disp_log: if print DDO running log. 
        return: {*}
        '''        
        if so_file_path is None:
            # Get the project root directory (3 levels up from this file)
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
            so_file_path = os.path.join(project_root, "rox_odometry_module.so")
        print("loading openrox ddo library from ", so_file_path)
        self.cdll = cdll.LoadLibrary(so_file_path)  
        self.ifmask=ifmask
        self.disp_log = disp_log
        self.ifrobust = ifrobust
     
    def get_pose(self, Ir_uchar, Z_uchar, Ic_uchar, imask_uchar, cTr_given, Kuchar,):
        '''
        description: you have to use contiguous() for tensor data to make sure the memory allocation is contiguous. 
                    we also make a copy() again to make sure it's right if you didn't do it before this function. 
        return: {*}
        '''        
        self.cdll.ddo.restype =  POINTER(c_double)
        cols = Ir_uchar.shape[1]
        rows = Ir_uchar.shape[0] 
        #print(f"h {rows} w {cols}, matrix \n {imask_uchar}")
        # vis_depth_tensor(torch.FloatTensor(imask_uchar).unsqueeze(0), "/home/ziliu/vis/openroxvo", "imaskunchar")
        # to make sure the memory is continous. we do deep copy(). 
        Ir_uchar, Z_uchar, Ic_uchar, imask_uchar,\
             cTr_given, Kuchar = np.asarray(Ir_uchar, np.double).reshape(-1).copy().ctypes.data_as(POINTER(c_ubyte)), np.asarray(Z_uchar, np.double).reshape(-1).copy().ctypes.data_as(POINTER(c_ubyte)),\
                 np.asarray(Ic_uchar, np.double).reshape(-1).copy().ctypes.data_as(POINTER(c_ubyte)), np.asarray(imask_uchar, np.double).reshape(-1).copy().ctypes.data_as(POINTER(c_ubyte)),\
                      np.asarray(cTr_given[:4,:4], np.double).copy().reshape(-1).ctypes.data_as(POINTER(c_ubyte)), np.asarray(Kuchar[:3,:3].copy(), np.double).reshape(-1).ctypes.data_as(POINTER(c_ubyte))
        cTr_est = self.cdll.ddo(Ir_uchar, Z_uchar, Ic_uchar, imask_uchar, cTr_given, Kuchar, rows, cols, self.ifmask, self.disp_log, self.ifrobust) # give POINTER and cols, rows
        cTr_est =  np.asarray([ cTr_est[i] for i in range(16)]).reshape(4,4)
        #print("string ctr>> ",cTr_est)
        return cTr_est

