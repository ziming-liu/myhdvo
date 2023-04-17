'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-04-12 10:11:32
LastEditors: Ziming Liu
LastEditTime: 2023-04-12 11:15:18
'''
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from zimingdepth.models.stereo_predictor.pixelnet import PixelNet
import torch

if __name__ == '__main__':

        
    
    max_disp = 192
    model = PixelNet(sample_rate=4, 
            max_disp=max_disp,
            backbone=dict(
                type='ResNet3dSlowOnly',
                in_channels=6,
                depth=18,
                pretrained='torchvision://resnet18',
                lateral=False,
                out_indices=(0,1,2,3, ),
                conv1_kernel=(1, 7, 7),
                conv1_stride_t=1,
                pool1_stride_t=1,
                inflate=(0, 0, 1, 1),
                norm_eval=False),
            neck=dict(type="FPN3d",
                in_channels=[256//4,512//4,1024//4,2048//4],
                out_channels=256,
                num_outs=4,
                out_indices=(0, ),
                start_level=0,
                end_level=-1,
                add_extra_convs=False,
                extra_convs_on_inputs=False,
                relu_before_extra_convs=False,
                no_norm_on_lateral=False,
                conv_cfg=dict(type='Conv3d'),
                #norm_cfg=dict(type='BN2d', requires_grad=True),
                #act_cfg=dict(type='ReLU', inplace=True),
                upsample_cfg=dict(mode='trilinear'),),
            losses=dict(
                    type="DispL1Loss",
                    start_disp=0,
                    # the maximum disparity of disparity search range
                    max_disp=max_disp,
                    # weight for l1_loss with regard to other loss type
                    weight=1,
                    # weights for different scale loss
                    weights=(1.0, 0.7, 0.5),
                    sparse=False,
                    set_range=True,
                    random_mask = False, 
                    random_mask_ratio = 0.5,
                ),)
    load_checkpoint(model, "work_dirs/pixelnet18_small_b8/iter_25000.pth")
    model = model.cpu().eval()

    dummy_input = (torch.randn(1, 3, 256, 512), torch.randn(1, 3, 256, 512))
    print(type(dummy_input))
    torch.onnx.export(model, dummy_input, "work_dirs/stereoone.onnx", opset_version=11, do_constant_folding=False)


