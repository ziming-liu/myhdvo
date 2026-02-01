"""
Fast homogeneous region masking module.

Author: Ziming Liu
Date: 2022-07-01
Last Modified: 2023-07-06
"""
import torch
import torch.nn as nn

from ..registry import MASKS

@MASKS.register_module()
class HomoMaskFast(nn.Module):
    """Fast homogeneous region mask generator using average and max pooling.
    
    Args:
        avg_kernel (tuple): Kernel size for average pooling. Default: (5, 5)
        max_kernel (tuple): Kernel size for max pooling. Default: (5, 5)
        threshold (float): Threshold for homogeneous region detection. Default: 2
    """
    
    def __init__(self, avg_kernel=(5, 5), max_kernel=(5, 5), threshold=2):
        super().__init__()
        self.avg_kernel = avg_kernel
        self.max_kernel = max_kernel
        self.threshold = threshold
        
        avg_pad = (int((avg_kernel[0] - 1) / 2), int((avg_kernel[1] - 1) / 2))
        max_pad = (int((max_kernel[0] - 1) / 2), int((max_kernel[1] - 1) / 2))
        
        self.avgpool = nn.AvgPool2d(
            kernel_size=avg_kernel, stride=(1, 1), padding=avg_pad
        ).cuda()
        self.maxpool = nn.MaxPool2d(
            kernel_size=max_kernel, stride=(1, 1), padding=max_pad
        ).cuda()
 

    def _each_homo_mask(self, x):
        """Generate homogeneous mask for input image.
        
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Binary mask of shape (B, 1, H, W)
        """
        b, c, h, w = x.shape
        
        # Apply double average pooling to smooth the image
        x_mean = self.avgpool(x)
        x_mean = self.avgpool(x_mean)
        
        # Compute absolute difference from mean
        x = torch.abs(x - x_mean)
        x = self.maxpool(x)
        
        # Create mask: 0 for homogeneous regions, 1 for non-homogeneous
        mask = torch.ones(x.shape, device=x.device)
        mask[x < self.threshold] = 0
        
        # Get maximum across channels
        mask_out, _ = torch.max(mask, 1, True)
        return mask_out


    def forward(self, x):
        """Forward pass to generate homogeneous mask.
        
        Args:
            x (Tensor): Input image tensor of shape (B, C, H, W)
            
        Returns:
            Tensor: Binary mask of shape (B, 1, H, W) where 0 indicates homogeneous regions
        """
        # Convert to grayscale if multi-channel
        if x.shape[1] > 1:
            x = x.mean(1, True)
        assert x.shape[1] == 1, "Input must be grayscale image"
        
        return self._each_homo_mask(x).detach()
    
    def loss(self, losses, mask):
        """Apply mask to loss values.
        
        Args:
            losses (dict or list): Loss dictionary or list of loss dictionaries
            mask (Tensor or list): Mask tensor or list of mask tensors
            
        Returns:
            list: Masked loss dictionaries
        """
        if not isinstance(losses, (list, tuple)):
            losses = [losses]
        if not isinstance(mask, (list, tuple)):
            mask = [mask for _ in range(len(losses))]
        
        for i in range(len(losses)):
            for k in losses[i].keys():
                losses[i][k] = losses[i][k] * mask[i].float()
        
        return losses
        

        
        