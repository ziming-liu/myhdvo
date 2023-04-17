''''''
'''
@Author: Huangying Zhan (huangying.zhan.work@gmail.com)
@Date: 2020-03-01
@Copyright: Copyright (C) Huangying Zhan 2020. All rights reserved. Please refer to the license file.
@LastEditTime: 2020-05-28
@LastEditors: Huangying Zhan
@Description: 
'''

import torch
import torch.nn as nn

from .reprojection import Reprojection


class PixToFlow(nn.Module):
    """Layer to transform flow into camera pixel coordiantes
    """
    def __init__(self, batch_size, height, width):
        """Prepare regular grid (Nx2xHxW)
        """
        super(PixToFlow, self).__init__()

        self.batch_size = batch_size
        self.height = height
        self.width = width

        meshgrid = np.meshgrid(range(self.width), range(self.height), indexing='xy')
        self.id_coords = np.stack(meshgrid, axis=0).astype(np.float32)
        self.id_coords = nn.Parameter(torch.from_numpy(self.id_coords))

        self.pix_coords = torch.unsqueeze(self.id_coords, 0)
        self.pix_coords = self.pix_coords.repeat(batch_size, 1, 1, 1)
        self.pix_coords = nn.Parameter(self.pix_coords)

    def forward(self, pix_coords, normalized=False):
        """Forward pass

        Args:
            pix_coords (tensor, [NxHxWx2]): pixel coordinates (normalized)
            normalized (bool):  flow vector normalized by image size
        
        Returns:
            flow (tensor, [Nx2xHxW]): [x, y] flow vector
        """
        flow = pix_coords.permute(0, 3, 1, 2) - self.pix_coords
        if normalized:
            flow[:, 0] /= self.width - 1
            flow[:, 1] /= self.height - 1
        return flow


class RigidFlow(nn.Module):
    """Layer to compute rigid flow given depth and camera motion
    """
    def __init__(self, height, width):
        """
        Args:
            height (int): image height
            width (int): image width
        """
        super(RigidFlow, self).__init__()
        # basic configuration
        self.height = height
        self.width = width
        self.device = torch.device('cuda')

        # layer setup
        self.pix2flow = PixToFlow(1, self.height, self.width) 
        self.pix2flow.to(self.device)

        self.reprojection = Reprojection(self.height, self.width)

    def forward(self, depth, T, K, inv_K, normalized=True):
        """Forward pass
        
        Args:
            depth (tensor, [Nx1xHxW]): depth map 
            T (tensor, [Nx4x4]): transformation matrice
            inv_K (tensor, [Nx4x4]): inverse camera intrinsics
            K (tensor, [Nx4x4]): camera intrinsics
            normalized (bool): 
                
                - True: normalized to [-1, 1]
                - False: [0, W-1] and [0, H-1]

        Returns:
            flow (NxHxWx2): rigid flow
        """
        xy = self.reprojection(depth, T, K, inv_K, normalized)
        flow = self.pix2flow(xy)

        return flow

