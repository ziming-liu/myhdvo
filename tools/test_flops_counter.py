import argparse
import os
import os.path as osp
from selectors import EpollSelector
import warnings
import numpy as np
import copy
import matplotlib.pyplot as plt
import time
import pandas as pd
import random
import cv2
import matplotlib as mp
import sys
#sys.path.append("../")
#from kitti_odom_eval.kitti_odometry import KittiEvalOdom
#from KITTI_odometry_evaluation_tool.evaluation import kittiOdomEval
mp.use("pdf")

#cmap = plt.cm.viridis
plasma = plt.get_cmap('magma')

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.runner import LogBuffer

from zimingdepth.apis import multi_gpu_test, single_gpu_test
from zimingdepth.datasets import build_dataloader, build_dataset
from zimingdepth.models import build_model
from zimingdepth.utils import collect_env, get_root_logger, register_module_hooks
#from eval_odometry import dr_api,dr_api2

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument(
        '--out',
        default=None,
        help='output result file in pkl/yaml/json format')
    parser.add_argument(
        '--fuse-conv-bn',
        action='store_true',
        help='Whether to fuse conv and bn, this will slightly increase'
        'the inference speed')
    parser.add_argument(
        '--eval',
        type=str,
        nargs='+',
        help='evaluation metrics, which depends on the dataset, e.g.,'
        ' "top_k_accuracy", "mean_class_accuracy" for video dataset')
    parser.add_argument(
        '--tasks',
        type=list,
        default=["depth"],
        nargs='+',
        help='evaluation task, "depth", "odometry" ')
    parser.add_argument(
        '--gpu-collect',
        action='store_true',
        help='whether to use gpu to collect results')
    parser.add_argument(
        '--tmpdir',
        help='tmp directory used for collecting results from multiple '
        'workers, available when gpu-collect is not specified')
    parser.add_argument(
        '--options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function (deprecate), '
        'change to --eval-options instead.')
    parser.add_argument(
        '--eval-options',
        nargs='+',
        action=DictAction,
        default={},
        help='custom options for evaluation, the key-value pair in xxx=yyy '
        'format will be kwargs for dataset.evaluate() function')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    parser.add_argument(
        '--average-clips',
        choices=['score', 'prob', None],
        default=None,
        help='average type when averaging test clips')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument(
        '--vis',
        default=True,
        help="visualize the testing results of the visual odometry"
    )
    parser.add_argument(
        '--load_pred_depth',
        type=str,
        default=None,
        help='if load already predicted depths or estimate again.'
    )
    parser.add_argument(
        '--gpu_collect',
        default=False,
        help="collect multi gpu test tensors"
    )
    parser.add_argument(
        '--iftransparent',
        default=False,
        help="if save the masks as transparent png"
    )
    parser.add_argument(
        '--test_seq_id',
        type=str,
        default=None,
        help=" give the test sequences of the visual odometry"
    )
    parser.add_argument(
        '--test_range',
        type=str,
        default=None,
        help=" give the test sequences of the visual odometry"
    )
    parser.add_argument(
        '--input_size',
        type=str,
        default=None,
        help=" input image size (H, W)"
    )
    parser.add_argument(
        '--max_depth',
        type=int,
        default=655,
        help=" The max depth values in the prediction map"
    )
    parser.add_argument(
        '--output_pkl',
        type=str,
        default=None,
        help=" directly load output .pkl for testing"
    )
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.options and args.eval_options:
        raise ValueError(
            '--options and --eval-options cannot be both '
            'specified, --options is deprecated in favor of --eval-options')
    if args.options:
        warnings.warn('--options is deprecated in favor of --eval-options')
        args.eval_options = args.options
    return args


def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)

    cfg.merge_from_dict(args.cfg_options)
    #logger = get_root_logger()

    # dump config
    #cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)

    # log some basic info
    #logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config: {cfg.pretty_text}')


    cfg.data.videos_per_gpu = 1
    if_still_dataset = 0
    if cfg.dataset_type == "KittiDepthStereoDataset" or cfg.dataset_type == "CityspacesDataset" or cfg.dataset_type == "KittiStereoMatchingDataset" or cfg.dataset_type == "SceneFlowDataset":
        if_still_dataset=1
    # give the test sequence ID 
    if args.test_seq_id is not None and if_still_dataset==0:
        logger.info("# update the test seq id to -{}-  ".format(str(args.test_seq_id)))
        cfg.test_seq_id = args.test_seq_id
        if cfg.dataset_type == "VKITTI2StereoDataset":
            cfg.data.test.test_seq_id = "Scene"+args.test_seq_id
        else:
            cfg.data.test.test_seq_id = args.test_seq_id
    if cfg.dataset_type == "KittiDepthStereoDataset" or cfg.dataset_type == "CityspacesDataset" or cfg.dataset_type == "KittiStereoMatchingDataset" or cfg.dataset_type == "SceneFlowDataset":
        test_seq_id = "99"
    #assert  "test_seq_id" in cfg, "you forgot to specific the test sequence"
    else:
        test_seq_id = cfg.test_seq_id

    if args.load_pred_depth is not None:
        print("directly load perdicted depths")
        path_list = args.load_pred_depth.strip().split(',')
        pl_cfg = cfg.data.test.pipeline[-2:]
        for idx, pl in enumerate(pl_cfg):
            if len(path_list) == 1:
                pl['keys'].append('left_pred_depths')
            if len(path_list) == 2:
                pl['keys'].append('left_pred_depths')
                pl['keys'].append('right_pred_depths')
        #assert len(path_list) < 2
        # set end id, end_id == numsample -1. because we didin't save the last sample of the pred_depths, because of the test input  data uses 2 imgs. 
        if "end_id" not in cfg.data.test:
            cfg.data.test['end_id'] = -1
        test_seq_id_for_load_depth = test_seq_id
        if cfg.dataset_type == "VKITTI2StereoDataset":
            test_seq_id_for_load_depth = args.test_seq_id
        if len(path_list) == 1:
            cfg.data.test['pred_depth_dir_left'] =  osp.join(cfg.work_dir, f"{path_list[0]}_pred_depths_left_{cfg.dataset_type}_seq"+ test_seq_id_for_load_depth)
        if len(path_list) == 2:
            cfg.data.test['pred_depth_dir_left'] =  osp.join(cfg.work_dir, f"{path_list[0]}_pred_depths_left_{cfg.dataset_type}_seq"+ test_seq_id_for_load_depth)
            cfg.data.test['pred_depth_dir_right'] =  osp.join(cfg.work_dir, f"{path_list[0]}_pred_depths_right_{cfg.dataset_type}_seq"+ test_seq_id_for_load_depth)
    print(cfg.data.test)
    if args.test_range is not None:   
        logger.info("# update the test range to -{}-  ".format(str(args.test_range)))
        cfg.data.test.test_range = (int(args.test_range.split(',')[0]), int(args.test_range.split(',')[1]))
    # Load output_config from cfg
    output_config = cfg.get('output_config', {})
    if args.out:
        # Overwrite output_config from args.out
        output_config = Config._merge_a_into_b(
            dict(out=args.out), output_config)

    # Load eval_config from cfg
    eval_config = cfg.get('eval_config', {})
    if args.eval:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(metrics=args.eval), eval_config)
    if args.tasks:
        # Overwrite eval_config from args.eval
        eval_config = Config._merge_a_into_b(
            dict(tasks=args.tasks), eval_config)
    if args.vis:
        # Overwrite eval_config from args.vis
        eval_config = Config._merge_a_into_b(
            dict(vis=args.vis), eval_config)
    
    if args.eval_options:
        # Add options from args.eval_options
        eval_config = Config._merge_a_into_b(args.eval_options, eval_config)

    assert output_config or eval_config, \
        ('Please specify at least one operation (save or eval the '
         'results) with the argument "--out" or "--eval"')

    dataset_type = cfg.data.test.type
    if output_config.get('out', None):
        out = output_config['out']
        # make sure the dirname of the output path exists
        mmcv.mkdir_or_exist(osp.dirname(out))
        _, suffix = osp.splitext(out)
        if dataset_type == 'AVADataset':
            assert suffix[1:] == 'csv', ('For AVADataset, the format of the '
                                         'output file should be csv')
        else:
            assert suffix[1:] in file_handlers, (
                'The format of the output '
                'file should be json, pickle or yaml')

    # set cudnn benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    cfg.data.test.test_mode = True

    if cfg.model.get('test_cfg') is None and cfg.get('test_cfg') is None:
        cfg.model.setdefault('test_cfg',
                             dict(average_clips=args.average_clips))
    else:
        # You can set average_clips during testing, it will override the
        # original settting
        if args.average_clips is not None:
            if cfg.model.get('test_cfg') is not None:
                cfg.model.test_cfg.average_clips = args.average_clips
            else:
                cfg.test_cfg.average_clips = args.average_clips

    # init distributed env first, since logger depends on the dist info.
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
    # log some basic info
    logger.info(f'Distributed training: {distributed}')
    
    # The flag is used to register module's hooks
    cfg.setdefault('module_hooks', [])

    # build the dataloader
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    dataloader_setting = dict(
        videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    #register_module_hooks(model.backbone, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    #if args.checkpoint.split('.')[-1] != "pth": 
     #   print("## didn't load checkpoint")
    #else:
    #    print("## loading checkpoint: {}".format(args.checkpoint))
    #    load_checkpoint(model, args.checkpoint, map_location='cpu')
        
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)

    #from ptflops import get_model_complexity_info
    with torch.cuda.device(0):
        model = model.cuda()
        #model.eval()
        """ 
        def input_constructor(stereo_input):
            input_tensor = torch.ones(*stereo_input).float().cuda()
            print(input_tensor.device)
            return {"left_imgs": input_tensor, "right_imgs": input_tensor, "left_depths": input_tensor,  "focal": torch.FloatTensor([277]).cuda(), "baseline":torch.FloatTensor([0.5]).cuda() }
        
        macs, params = get_model_complexity_info(model, (1, 3, int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1])), as_strings=True,
                                                input_constructor = input_constructor,
                                                print_per_layer_stat=True, verbose=True,)
                                            
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        """ 
        """
        from thop import profile
        input = torch.ones(1, 3, int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1])).float().cuda()
        focal = torch.FloatTensor([277]).cuda()
        baseline = torch.FloatTensor([0.5]).cuda()
        inputs = {"left_imgs": input, "right_imgs": input, "return_loss":False, "left_depths": input, "focal": torch.FloatTensor([277]).cuda(), "baseline":torch.FloatTensor([0.5]).cuda() }
        macs, params = profile(model, inputs=inputs)
        from thop import clever_format
        macs, params = clever_format([macs, params], "%.3f")
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))
        """

        from torchsummaryX import summary
        input_tensor = torch.ones(1, 3, int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1])).float().cuda()
        print(input_tensor.device)
        left_depth =  torch.ones(1, 1, int(args.input_size.split(',')[0]), int(args.input_size.split(',')[1])).float().cuda()
        ks = {   "return_loss":False, "left_depths": left_depth,  "left_disps": left_depth,  "focal": torch.FloatTensor([277]).cuda(), "baseline":torch.FloatTensor([0.5]).cuda() }
        
        summary(model, input_tensor, input_tensor, **ks)


if __name__ == '__main__':
    main()
