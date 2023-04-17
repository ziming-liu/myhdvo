'''
Author: 
Date: 2022-07-07 23:22:12
LastEditors: Ziming
LastEditTime: 2022-07-07 23:59:34
Description: refer to https://github.com/DeepMotionAIResearch/DenseMatchingBenchmark 
Dependent packages: don't need any extral dependency
'''
import torch.nn as nn


class CostProcessor(nn.Module):

    def __init__(self):
        super(CostProcessor, self).__init__()

    def forward(self, *input):
        raise NotImplementedError
