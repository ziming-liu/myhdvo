'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-12 15:16:22
LastEditors: Ziming Liu
LastEditTime: 2023-03-12 19:46:55
'''
import torch
import torch.nn.functional as F
import numpy as np


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]

    img = F.grid_sample(img, coords, )

    if mask:
        mask = (
            (coords[:, :, :, 0:1] < 0)
            | (coords[:, :, :, 0:1] > W - 1)
            | (coords[:, :, :, 1:2] < 0)
            | (coords[:, :, :, 1:2] > H - 1)
        )
        mask = torch.logical_not(mask)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    x_grid, y_grid = np.meshgrid(np.arange(wd), np.arange(ht))
    y_grid, x_grid = torch.FloatTensor(y_grid).float(), torch.FloatTensor(
        x_grid).float()
    coords = torch.stack([x_grid, y_grid], dim=0)
    coords = torch.repeat_interleave(torch.unsqueeze(coords, dim=0), batch, dim=0)
    return coords


def manual_pad(x, pady, padx):
    if pady > 0:
        u = torch.repeat_interleave(x[:, :, 0:1, :], pady, dim=2)
        d = torch.repeat_interleave(x[:, :, -1:, :], pady, dim=2)
        x = torch.concat([u, x, d], dim=2)
    if padx > 0:
        l = torch.repeat_interleave(x[:, :, :, 0:1], padx, dim=3)
        r = torch.repeat_interleave(x[:, :, :, -1:], padx, dim=3)
        x = torch.concat([l, x, r], dim=3)
    return x