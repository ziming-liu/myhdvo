import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from .crestereo_submodules.update import BasicUpdateBlock
from .crestereo_submodules.extractor import BasicEncoder
from .crestereo_submodules.corr import AGCL

from .crestereo_submodules import PositionEncodingSine, LocalFeatureTransformer
from ..registry import STEREO_PREDICTOR
from abc import ABCMeta, abstractmethod
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import _load_checkpoint, load_checkpoint
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from ...utils import get_root_logger
from mmcv.runner import auto_fp16
import warnings
import torch
import torch.distributed as dist
import torch.nn as nn
from mmcv.runner import auto_fp16
from .. import builder
import warnings


@STEREO_PREDICTOR.register_module()
class CREStereo(nn.Module):
    def __init__(self, max_disp=192, mixed_precision=False, test_mode=False, **kwargs):
        super(CREStereo, self).__init__()
        self.max_disp = max_disp
        self.max_flow = max_disp
        self.mixed_precision = mixed_precision
        self.test_mode = test_mode

        self.hidden_dim = 128
        self.context_dim = 128
        self.dropout = 0

        # feature network and update block
        self.fnet = BasicEncoder(
            output_dim=256, norm_fn="instance", dropout=self.dropout
        )
        self.update_block = BasicUpdateBlock(
            hidden_dim=self.hidden_dim, cor_planes=4 * 9, mask_size=4
        )

        # loftr
        self.self_att_fn = LocalFeatureTransformer(
            d_model=256, nhead=8, layer_names=["self"] * 1, attention="linear"
        )
        self.cross_att_fn = LocalFeatureTransformer(
            d_model=256, nhead=8, layer_names=["cross"] * 1, attention="linear"
        )

        # adaptive search
        self.search_num = 9
        self.conv_offset_16 = nn.Conv2d(
            256, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.conv_offset_8 = nn.Conv2d(
            256, self.search_num * 2, kernel_size=3, stride=1, padding=1
        )
        self.range_16 = 1
        self.range_8 = 1

        self.timer = dict(sum_time=0, count=0, avg_time=0, f0_time=0, fps=0)

        self.freeze_bn()

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                nn.eval()

    
    def convex_upsample(self, flow, mask, rate=4):
        """[H/rate, W/rate, 2] -> [H, W, 2]"""
        N, _, H, W = flow.shape
        mask = mask.reshape(N, 1, 9, rate, rate, H, W)
        mask = F.softmax(mask, 2)

        #up_flow = self.unfold(rate * flow, [3, 3], padding=1)
        up_flow = F.unfold(rate * flow, [3, 3], padding=1)
        up_flow = up_flow.reshape(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, 2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return torch.reshape(up_flow, (N, 2, rate * H, rate * W))

    def zero_init(self, fmap):
        N, C, H, W = fmap.shape
        _x = torch.zeros([N, 1, H, W]).float()
        _y = torch.zeros([N, 1, H, W]).float()
        zero_flow = torch.cat([_x, _y], dim=1).to(fmap.device)
        return zero_flow

    def _forward(self, image1, image2, iters=10, flow_init=None):
        self.freeze_bn()

        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        hdim = self.hidden_dim
        cdim = self.context_dim

        # feature network
        fmap1, fmap2 = self.fnet([image1, image2])

        fmap1 = fmap1.float()
        fmap2 = fmap2.float()


        # 1/4 -> 1/8
        # feature
        fmap1_dw8 = F.avg_pool2d(fmap1, 2, stride=2)
        fmap2_dw8 = F.avg_pool2d(fmap2, 2, stride=2)

        # offset
        offset_dw8 = self.conv_offset_8(fmap1_dw8)
        offset_dw8 = self.range_8 * (torch.sigmoid(offset_dw8) - 0.5) * 2.0

        # context
        net, inp = torch.split(fmap1, [hdim, fmap1.shape[1]-hdim], dim=1)
        net = F.tanh(net)
        inp = F.relu(inp)
        net_dw8 = F.avg_pool2d(net, 2, stride=2)
        inp_dw8 = F.avg_pool2d(inp, 2, stride=2)

        # 1/4 -> 1/16
        # feature
        fmap1_dw16 = F.avg_pool2d(fmap1, 4, stride=4)
        fmap2_dw16 = F.avg_pool2d(fmap2, 4, stride=4)
        offset_dw16 = self.conv_offset_16(fmap1_dw16)
        offset_dw16 = self.range_16 * (torch.sigmoid(offset_dw16) - 0.5) * 2.0

        # context
        net_dw16 = F.avg_pool2d(net, 4, stride=4)
        inp_dw16 = F.avg_pool2d(inp, 4, stride=4)

        # positional encoding and self-attention
        pos_encoding_fn_small = PositionEncodingSine(
            d_model=256, max_shape=(image1.shape[2] // 16, image1.shape[3] // 16)
        )
        # 'n c h w -> n (h w) c'
        x_tmp = pos_encoding_fn_small(fmap1_dw16)
        fmap1_dw16 = torch.reshape(
            x_tmp.permute(0, 2, 3, 1),
            (x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1]),
        )
        # 'n c h w -> n (h w) c'
        x_tmp = pos_encoding_fn_small(fmap2_dw16)
        fmap2_dw16 = torch.reshape(
            x_tmp.permute(0, 2, 3, 1),
            (x_tmp.shape[0], x_tmp.shape[2] * x_tmp.shape[3], x_tmp.shape[1]),
        )

        fmap1_dw16, fmap2_dw16 = self.self_att_fn(fmap1_dw16, fmap2_dw16)
        fmap1_dw16, fmap2_dw16 = [
                torch.reshape(x, (x.shape[0], \
                                  image1.shape[2] // 16, -1, x.shape[2])).permute(0, 3, 1, 2)
            
            for x in [fmap1_dw16, fmap2_dw16]
        ]

        corr_fn = AGCL(fmap1, fmap2)
        corr_fn_dw8 = AGCL(fmap1_dw8, fmap2_dw8)
        corr_fn_att_dw16 = AGCL(fmap1_dw16, fmap2_dw16, att=self.cross_att_fn)

        # Cascaded refinement (1/16 + 1/8 + 1/4)
        predictions = []
        flow = None
        flow_up = None
        if flow_init is not None:
            scale = fmap1.shape[2] / flow_init.shape[2]
            flow = -scale * F.interpolate(
                flow_init,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )
        else:
            # zero initialization
            flow_dw16 = self.zero_init(fmap1_dw16)

            # Recurrent Update Module
            # RUM: 1/16
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw16 = flow_dw16.detach()
                out_corrs = corr_fn_att_dw16(
                    flow_dw16, offset_dw16, small_patch=small_patch
                )

                net_dw16, up_mask, delta_flow = self.update_block(
                    net_dw16, inp_dw16, out_corrs, flow_dw16
                )

                flow_dw16 = flow_dw16 + delta_flow
                flow = self.convex_upsample(flow_dw16, up_mask, rate=4)
                flow_up = -4 * F.interpolate(
                    flow,
                    size=(4 * flow.shape[2], 4 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)
            
            scale = fmap1_dw8.shape[2] / flow.shape[2]
            flow_dw8 = -scale * F.interpolate(
                flow,
                size=(fmap1_dw8.shape[2], fmap1_dw8.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

            # RUM: 1/8
            for itr in range(iters // 2):
                if itr % 2 == 0:
                    small_patch = False
                else:
                    small_patch = True

                flow_dw8 = flow_dw8.detach()
                out_corrs = corr_fn_dw8(flow_dw8, offset_dw8, small_patch=small_patch)

                net_dw8, up_mask, delta_flow = self.update_block(
                    net_dw8, inp_dw8, out_corrs, flow_dw8
                )

                flow_dw8 = flow_dw8 + delta_flow
                flow = self.convex_upsample(flow_dw8, up_mask, rate=4)
                flow_up = -2 * F.interpolate(
                    flow,
                    size=(2 * flow.shape[2], 2 * flow.shape[3]),
                    mode="bilinear",
                    align_corners=True,
                )
                predictions.append(flow_up)

            scale = fmap1.shape[2] / flow.shape[2]
            flow = -scale * F.interpolate(
                flow,
                size=(fmap1.shape[2], fmap1.shape[3]),
                mode="bilinear",
                align_corners=True,
            )

        # RUM: 1/4
        for itr in range(iters):
            if itr % 2 == 0:
                small_patch = False
            else:
                small_patch = True

            flow = flow.detach()
            out_corrs = corr_fn(flow, None, small_patch=small_patch, iter_mode=True)

            net, up_mask, delta_flow = self.update_block(net, inp, out_corrs, flow)

            flow = flow + delta_flow
            flow_up = -self.convex_upsample(flow, up_mask, rate=4)
            predictions.append(flow_up)

        #if self.test_mode:
        #    return flow_up

        return predictions
    


    def sequence_loss(self, flow_preds, flow_gt, valid, gamma=0.8):

        n_predictions = len(flow_preds)
        flow_loss = 0.0
        flow_losses = dict()
        for i in range(n_predictions):
            i_weight = gamma ** (n_predictions - i - 1)
            i_loss = torch.abs(flow_preds[i] - flow_gt)
            flow_losses[f"flow_loss_{i}"] = i_weight * (valid * i_loss).mean()

        return flow_losses

        
    def forward_train(self, left_image, right_image, **kwargs):

        flow_predictions = self._forward(left_image, right_image)
        gt_disp = kwargs["left_disps"]
        gt_flow = torch.cat([gt_disp, gt_disp * 0], 1).reshape(flow_predictions[-1].shape)

        mask = (gt_disp < self.max_disp) & (gt_disp > 0)
        
        loss = self.sequence_loss(
                        flow_predictions, gt_flow, mask, gamma=0.8
                    )


        return loss

    def forward_test(self, left_image ,right_image, **kwargs):
        t0 = time.time()
        outputs = self._forward(left_image, right_image, iters=20)
        torch.cuda.synchronize()
        
        outs = [[],[],[],[],[],[]]
        outs[0] = outputs[-1][:,0,:,:].detach().cpu().numpy()
        outs[1] = kwargs["left_disps"].detach().cpu().numpy()
        t1 = time.time()
        self.timer["sum_time"] += t1 - t0
        self.timer["count"] += 1
        if self.timer["f0_time"] == 0:
            self.timer["f0_time"] = t1 - t0
        else:
            self.timer["avg_time"] = (self.timer["sum_time"]-self.timer["f0_time"]) / (self.timer["count"]-1)
            self.timer["fps"] = 1 / self.timer["avg_time"]
            print("avg_time: ", self.timer["avg_time"])
            print("fps: ", self.timer["fps"])
            
        return  outs


    @staticmethod
    def _parse_losses(losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor
                which may be a weighted sum of all losses, log_vars contains
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items())
                #   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def forward(self, left_imgs, right_imgs =None, return_loss=True, **kwargs):
        """Define the computation performed at every call."""
        #self.start_index = kwargs['start_index']
        #self.seq_length =  kwargs['seq_len']
        if kwargs.get('gradcam', False):
            del kwargs['gradcam']
            return self.forward_gradcam(left_imgs, right_imgs, **kwargs)
        if return_loss:
            #if pose is None:
            #    return self.forward_train(left_imgs, right_imgs, **kwargs)
            #    #raise ValueError('Label should not be None.')
            return self.forward_train(left_imgs, right_imgs, **kwargs)

        return self.forward_test(left_imgs, right_imgs, **kwargs)

 

    def train_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data_batch (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``,
                ``num_samples``.
                ``loss`` is a tensor for back propagation, which can be a
                weighted sum of multiple losses.
                ``log_vars`` contains all the variables to be sent to the
                logger.
                ``num_samples`` indicates the batch size (when the model is
                DDP, it means the batch size on each GPU), which is used for
                averaging the logs.
        """
        left_imgs, right_imgs = data_batch['left_imgs'], data_batch['right_imgs']
        #label = data_batch['pose']
        
        aux_info = {}
        # if aux info is not define in config, there 
        #for item in self.aux_info:
        #    assert item in data_batch
        #    aux_info[item] = data_batch[item]
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        keys.remove('right_imgs')
        #keys.remove('pose')
        for item in keys:
            aux_info[item] = data_batch[item]
        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)
        
        # print gradient of network
        #for name, parms in self.named_parameters():
        #    print('-->name:', name, '-->grad_requirs:',parms.requires_grad,  )
        #    print(' -->grad_value: \n {}'.format(parms.grad))

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs

    def val_step(self, data_batch, optimizer, **kwargs):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        left_imgs, right_imgs = data_batch['left_imgs'], data_batch['right_imgs']
        #label = data_batch['pose']

        aux_info = {}
        #for item in self.aux_info:
        #    aux_info[item] = data_batch[item]
        keys = list(data_batch.keys())
        keys.remove('left_imgs')
        keys.remove('right_imgs')
        #keys.remove('pose')
        for item in keys:
            aux_info[item] = data_batch[item]

        losses = self(left_imgs, right_imgs, return_loss=True, **aux_info)

        loss, log_vars = self._parse_losses(losses)
        
        # print gradient of network
        #for name, parms in self.named_parameters():
        #    print('-->name:', name, '-->grad_requirs:',parms.requires_grad,  )
        #    print(' -->grad_value: \n {}'.format(parms.grad))

        outputs = dict(
            loss=loss,
            log_vars=log_vars,
            num_samples=len(next(iter(data_batch.values()))))

        return outputs
    
