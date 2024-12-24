'''
Author: Ziming Liu
Date: 2021-08-31 15:55:44
LastEditors: Ziming Liu
LastEditTime: 2023-06-03 19:12:20
Description: ...
Dependent packages: don't need any extral dependency
'''
from torch.autograd import Function
import torch

import esm_grid_sample_cpp
import esm_grid_sample_cuda

class ESMGridSampleCUDAFunction(Function):

    @staticmethod
    def forward(ctx, source_depth, pr, gt_map):
        dw =  esm_grid_sample_cuda.forward(source_depth, pr, 0,0,True) # interploate_type(but will not work, todo), padding_type(zeros), aligh_corner
        #interploate = torch.LongTensor(interploate)
        #padding = torch.LongTensor(padding)
        ctx.save_for_backward(source_depth, pr, gt_map)
        return dw

    @staticmethod
    def backward(ctx, gradOutput):
        grad_ = esm_grid_sample_cuda.backward(gradOutput, *ctx.saved_variables,0,0,True)
        return  grad_[0], grad_[1]


class ESMGridSampleFunction(Function):

    @staticmethod
    def forward(ctx, source_depth, pr, gt_map):
        dw =  esm_grid_sample_cpp.forward(source_depth, pr,0,0,False)
        ctx.save_for_backward(source_depth, pr, gt_map)
        return dw

    @staticmethod
    def backward(ctx, gradOutput):
        grad_ = esm_grid_sample_cpp.backward(gradOutput, *ctx.saved_variables,0,0,False)
        return  grad_[0], grad_[1]


class ESMGridSample(torch.nn.Module):

    def __init__(self):
        super(ESMGridSample, self).__init__()

    def forward(self, source_depth, pr, gt_map):
        return ESMGridSampleFunction.apply(source_depth, pr, gt_map)


class ESMGridSampleCUDA(torch.nn.Module):

    def __init__(self):
        super(ESMGridSampleCUDA, self).__init__()

    def forward(self, source_depth, pr, gt_map, interploate=0, padding=0, bord_align=True):
        return ESMGridSampleCUDAFunction.apply(source_depth, pr, gt_map)

