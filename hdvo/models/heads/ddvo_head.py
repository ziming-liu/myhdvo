"""
DDVO (Direct Deep Visual Odometry) Pose Head.

This module implements a pose estimation head using DDVO for visual odometry
with photometric and structural losses.

Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-07-06
Last Modified: 2024-02-07
"""

import cv2
import numpy as np
import torch
import torch.nn as nn

from ..builder import build_loss, build_visual_odometry
from ..registry import HEADS
from ..utils.temporal_warping import temporal_warp_core

#from ..visual_odometry.pose_transform import *

 
# epsilon for testing whether a number is close to zero
_EPS = torch.finfo(float).eps * 4.0

# axis sequences for Euler angles
_NEXT_AXIS = [1, 2, 0, 1]

# map axes strings to/from tuples of inner axis, parity, repetition, frame
_AXES2TUPLE = {
    'sxyz': (0, 0, 0, 0), 'sxyx': (0, 0, 1, 0), 'sxzy': (0, 1, 0, 0),
    'sxzx': (0, 1, 1, 0), 'syzx': (1, 0, 0, 0), 'syzy': (1, 0, 1, 0),
    'syxz': (1, 1, 0, 0), 'syxy': (1, 1, 1, 0), 'szxy': (2, 0, 0, 0),
    'szxz': (2, 0, 1, 0), 'szyx': (2, 1, 0, 0), 'szyz': (2, 1, 1, 0),
    'rzyx': (0, 0, 0, 1), 'rxyx': (0, 0, 1, 1), 'ryzx': (0, 1, 0, 1),
    'rxzx': (0, 1, 1, 1), 'rxzy': (1, 0, 0, 1), 'ryzy': (1, 0, 1, 1),
    'rzxy': (1, 1, 0, 1), 'ryxy': (1, 1, 1, 1), 'ryxz': (2, 0, 0, 1),
    'rzxz': (2, 0, 1, 1), 'rxyz': (2, 1, 0, 1), 'rzyz': (2, 1, 1, 1)}

_TUPLE2AXES = dict((v, k) for k, v in _AXES2TUPLE.items())


def euler_from_matrix(matrix, axes='szxy'):
    """Extract Euler angles from rotation matrix.
    
    Args:
        matrix: 4x4 or 3x3 transformation/rotation matrix.
        axes (str): Axis specification string. Defaults to 'szxy'.
        
    Returns:
        Tensor: Euler angles [ax, ay, az].
        
    Raises:
        ValueError: If axes string is invalid.
    """
    try:
        firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    except (AttributeError, KeyError):
        raise ValueError("Invalid axes value")

    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]

    #M = torch.tensor(matrix, dtype=torch.float64)[:3, :3]
    M = matrix[:3, :3].type(torch.float64)
    if repetition:
        sy = torch.sqrt(M[i, j] * M[i, j] + M[i, k] * M[i, k])
        if sy > _EPS:
            ax = torch.atan2(M[i, j], M[i, k])
            ay = torch.atan2(sy, M[i, i])
            az = torch.atan2(M[j, i], -M[k, i])
        else:
            ax = torch.atan2(-M[j, k], M[j, j])
            ay = torch.atan2(sy, M[i, i])
            az = torch.tensor(0.0)
    else:
        cy = torch.sqrt(M[i, i] * M[i, i] + M[j, i] * M[j, i])
        if cy > _EPS:
            ax = torch.atan2(M[k, j], M[k, k])
            ay = torch.atan2(-M[k, i], cy)
            az = torch.atan2(M[j, i], M[i, i])
        else:
            ax = torch.atan2(-M[j, k], M[j, j])
            ay = torch.atan2(-M[k, i], cy)
            az = torch.tensor(0.0)

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return torch.FloatTensor([ax, ay, az]).type_as(matrix)

 
    
@HEADS.register_module()
class PoseDDVOHead(nn.Module):
    """Direct Deep Visual Odometry pose estimation head.
    
    This head uses DDVO for pose estimation with photometric and structural
    consistency losses through temporal warping.
    
    Args:
        ddvo (dict): DDVO configuration.
        photo_loss (dict, optional): Photometric loss configuration.
        struct_loss (dict, optional): Structural loss configuration.
        loss_weights (list): Weights for multi-scale losses. 
            Defaults to [1, 1.25, 1.5, 1.75, 2.0].
        grid_sample_type (str): Grid sampling type. Defaults to 'pytorch'.
        padding_mode (str): Padding mode for warping. Defaults to 'zeros'.
    """
    
    def __init__(self, 
                 ddvo,
                 photo_loss = None,
                 struct_loss = None,
                 loss_weights = [1, 1.25, 1.5, 1.75, 2.0],
                 grid_sample_type="pytorch", 
                 padding_mode="zeros",  ):
        super(PoseDDVOHead, self).__init__()
        self.loss_weights = loss_weights
        self.grid_sample_type = grid_sample_type
        self.padding_mode = padding_mode
        if photo_loss is not None: 
            self.photo_loss = build_loss(photo_loss)
        else:
            self.photo_loss = None
        if struct_loss is not None:
            self.struct_loss = build_loss(struct_loss)
        else:
            self.struct_loss = None
        self.ddvo = build_visual_odometry(ddvo)
        self.initpose = np.eye(4)
          
    
    def forward(self, source_img, target_depth, sTt, K, target_img=None, target_mask=None, test_mode=False):
        assert not isinstance(source_img, (list,tuple))
        assert not isinstance(target_depth, (list,tuple))
        assert not isinstance(target_img, (list,tuple))

        if not test_mode:
            return self.forward_train(source_img, target_depth, sTt, K, target_img, target_mask)
        else:
            return self.forward_test(source_img, target_depth, sTt, K, target_img, target_mask)


    def forward_train(self, source_img, target_depth, sTt, K, target_img, target_mask):
        warped = temporal_warp_core(source_img, target_depth, sTt, K, torch.linalg.inv(K), gt_map=target_img)
        valid_mask = (warped != 0).all(dim=1, keepdim=True).float().detach()
        
        # # Temporary visualization for debugging warping operation
        # self._visualize_warping_debug(source_img, target_img, warped, valid_mask=valid_mask)
        
        loss = self.loss(warped, target_img)
        loss = {k: (v * valid_mask).mean() for k, v in loss.items()}
        if target_mask is not None:
            loss = {k: (v * target_mask).mean() for k, v in loss.items()}
       
        return loss
    
    def forward_test(self, source_img, target_depth, init_sTt, K, target_img, target_mask):
        #self.ddvo.max_iters = 100 # this results a bug. change the max_iters in the ddvo, the next training iter will have the wrong max_iters
        if target_mask is None:
            target_mask = torch.ones_like(target_depth)
        Ir, Zr, Ic,  K, imask , cTr = target_img, target_depth, source_img, K, target_mask, init_sTt

        b,_,h,w = Ir.shape
        est_poses = []
        Ir, Ic = Ir.detach().permute(0,2,3,1).cpu().numpy(), Ic.detach().permute(0,2,3,1).cpu().numpy()
        Zr = Zr.detach().cpu().numpy()
        target_mask = target_mask.detach().cpu().numpy()
         
        for idx in range(b):
            est_pose = self.ddvo.get_pose(cv2.cvtColor(Ir[idx], cv2.COLOR_RGB2GRAY).reshape(h,w), Zr[idx].reshape(h,w), cv2.cvtColor(Ic[idx], cv2.COLOR_RGB2GRAY).reshape(h,w),\
                                        target_mask[idx].reshape(h,w), self.initpose, K.detach().cpu().numpy())
            self.initpose = est_pose.copy()
            est_poses.append(torch.FloatTensor(est_pose).to(source_img.device))
        est_poses = torch.stack(est_poses)
        return est_poses

    def loss(self, warped_ref, ref_img):
        loss = {}
        ddvo_photoloss = self.photo_loss(warped_ref, ref_img)
        loss[f"ddvo_photoloss"] = ddvo_photoloss #* (warped_ref[i]!=0).float().detach()
        ddvo_structloss = self.struct_loss(warped_ref, ref_img)
        loss[f"ddvo_structloss"] = ddvo_structloss #* (warped_ref[i]!=0).float().detach()
        
        return loss
    
    def _visualize_warping_debug(self, source_img, target_img, warped, valid_mask=None, save_dir='./debug_warp_vis'):
        """
        Temporary visualization function to check image warping operation.
        Saves source, target, and warped images side by side for inspection.
        
        Args:
            source_img: Source image tensor [B, C, H, W]
            target_img: Target/reference image tensor [B, C, H, W]
            warped: Warped image tensor [B, C, H, W]
            valid_mask: Optional valid mask [B, 1, H, W]
            save_dir: Directory to save visualization images
        """
        import os
        import matplotlib.pyplot as plt
        from datetime import datetime
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Only visualize the first sample in batch to avoid too many images
        batch_idx = 0
        
        # Convert tensors to numpy arrays and normalize to [0, 1]
        def tensor_to_numpy(img_tensor):
            img = img_tensor[batch_idx].detach().cpu()
            # Normalize to [0, 1] range
            img = (img - img.min()) / (img.max() - img.min() + 1e-8)
            # Convert from CxHxW to HxWxC
            if img.shape[0] == 3:
                img = img.permute(1, 2, 0).numpy()
            else:
                img = img.squeeze().numpy()
            return img
        
        source_np = tensor_to_numpy(source_img)
        target_np = tensor_to_numpy(target_img)
        warped_np = tensor_to_numpy(warped)
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Plot source image
        axes[0, 0].imshow(source_np)
        axes[0, 0].set_title('Source Image', fontsize=12)
        axes[0, 0].axis('off')
        
        # Plot target image
        axes[0, 1].imshow(target_np)
        axes[0, 1].set_title('Target Image', fontsize=12)
        axes[0, 1].axis('off')
        
        # Plot warped image
        axes[0, 2].imshow(warped_np)
        axes[0, 2].set_title('Warped Image', fontsize=12)
        axes[0, 2].axis('off')
        
        # Plot difference: target - warped
        diff = np.abs(target_np - warped_np)
        axes[1, 0].imshow(diff)
        axes[1, 0].set_title('Abs Difference (Target - Warped)', fontsize=12)
        axes[1, 0].axis('off')
        
        # Plot valid mask if provided
        if valid_mask is not None:
            mask_np = valid_mask[batch_idx].detach().cpu().squeeze().numpy()
            axes[1, 1].imshow(mask_np, cmap='gray')
            axes[1, 1].set_title('Valid Mask', fontsize=12)
        else:
            axes[1, 1].text(0.5, 0.5, 'No Valid Mask', ha='center', va='center')
            axes[1, 1].set_title('Valid Mask', fontsize=12)
        axes[1, 1].axis('off')
        
        # Plot overlay: blend target and warped
        if len(target_np.shape) == 3:
            overlay = 0.5 * target_np + 0.5 * warped_np
            axes[1, 2].imshow(overlay)
            axes[1, 2].set_title('Overlay (0.5*Target + 0.5*Warped)', fontsize=12)
        else:
            axes[1, 2].text(0.5, 0.5, 'Cannot overlay grayscale', ha='center', va='center')
            axes[1, 2].set_title('Overlay', fontsize=12)
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        
        # Save with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        save_path = os.path.join(save_dir, f'warp_debug_{timestamp}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[DEBUG] Warping visualization saved to: {save_path}") 
