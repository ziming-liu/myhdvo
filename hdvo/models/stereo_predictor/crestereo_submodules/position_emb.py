'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 15:21:41
LastEditors: Ziming Liu
LastEditTime: 2023-03-12 19:13:56
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.unsqueeze(torch.cumsum(torch.ones(max_shape), 0), 0)
        x_position = torch.unsqueeze(torch.cumsum(torch.ones(max_shape), 1), 0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2) * (-math.log(10000.0) / d_model // 2)
        )
        div_term = div_term.unsqueeze(1).unsqueeze(2)  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.pe = torch.unsqueeze(pe, 0)

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, : x.shape[2], : x.shape[3]].to(x.device)