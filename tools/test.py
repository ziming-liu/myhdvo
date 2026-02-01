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
from torch.nn import functional as F

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

from hdvo.apis import multi_gpu_test, single_gpu_test
from hdvo.datasets import build_dataloader, build_dataset
from hdvo.models import build_model
from hdvo.utils import collect_env, get_root_logger, register_module_hooks


# ============================================================================
# Result Processing and Evaluation Helper Functions
# ============================================================================

def save_depth_results(outputs, args, cfg, test_seq_id, logger):
    """
    Save depth prediction and ground truth depth maps.
    
    Args:
        outputs: Model outputs containing [pred_depths, gt_depths, ...]
        args: Command line arguments
        cfg: Config object
        test_seq_id: Current test sequence ID
        logger: Logger instance
    """
    if args.load_pred_depth is not None or len(outputs[0]) == 0:
        return
    
    from outputs_proc.save_load_depth import save_depth_maps
    checkpoint_name = args.checkpoint.split('/')[-1].split('.')[0]
    
    # Process and save predicted left depth maps
    pred_depth = outputs[0].copy()
    pred_depth = [item[0] for item in pred_depth]  # Extract left depths
    
    # Clip invalid depth values
    for i in range(len(pred_depth)):
        pred_depth[i][pred_depth[i] > args.max_depth] = args.max_depth
        pred_depth[i][np.isnan(pred_depth[i])] = args.max_depth
        pred_depth[i][np.isinf(pred_depth[i])] = args.max_depth
    
    if args.save_depth:
        logger.info(f"Saving left depth maps for {test_seq_id}")
        save_depth_maps(
            cfg.dataset_type, test_seq_id, pred_depth, cfg.work_dir,
            checkpoint_name, stereo_view="left",
            min_depth=1, max_depth=args.max_depth, first_frame_id=0
        )
        
        # Save ground truth left depth maps
        if len(outputs[1]) > 0:
            gt_depth = [item[0] for item in outputs[1].copy()]
            logger.info(f"Saving GT left depth maps for {test_seq_id}")
            save_depth_maps(
                cfg.dataset_type, test_seq_id, gt_depth, cfg.work_dir,
                checkpoint_name, stereo_view="left", ifgtdepth=True,
                min_depth=1, max_depth=args.max_depth, first_frame_id=0
            )
        
        # Save right depth maps if available
        if len(outputs[0][0]) == 2:
            pred_depth_right = [item[1] for item in outputs[0].copy()]
            logger.info(f"Saving right depth maps for {test_seq_id}")
            save_depth_maps(
                cfg.dataset_type, test_seq_id, pred_depth_right, cfg.work_dir,
                checkpoint_name, stereo_view="right",
                min_depth=1, max_depth=6000, first_frame_id=0
            )
        
        logger.info("Done saving depth maps")


def save_mask_results(outputs, args, cfg, test_seq_id, logger):
    """
    Save predicted and ground truth mask maps.
    
    Args:
        outputs: Model outputs containing [..., pred_masks, gt_masks, ...]
        args: Command line arguments
        cfg: Config object
        test_seq_id: Current test sequence ID
        logger: Logger instance
    """
    from outputs_proc.save_load_mask import save_mask_maps
    checkpoint_name = args.checkpoint.split('/')[-1].split('.')[0]
    
    # Save predicted masks
    if len(outputs[2]) > 0:
        # Determine mask types based on output shape
        if outputs[2][0].shape[0] == 6:
            pred_mask_types = [
                "left_temporal_multi_masks", "left_homo_mask",
                "left_stc_t_mask", "left_stc_s_mask",
                "left_temporal_edge_mask", "left_stereo_edge_mask"
            ]
        else:
            pred_mask_types = [
                "left_temporal_multi_masks",
                "right_temporal_multi_masks"
            ]
        
        num_mask_types = len(outputs[2][0]) if isinstance(outputs[2][0], list) else outputs[2][0].shape[0]
        
        for mask_idx in range(num_mask_types):
            mask_type = pred_mask_types[mask_idx]
            logger.info(f"Saving {mask_type} for {test_seq_id}")
            pred_mask = [item[mask_idx] for item in outputs[2]]
            save_mask_maps(
                cfg.dataset_type, mask_type, test_seq_id, pred_mask,
                cfg.work_dir, checkpoint_name, first_frame_id=0,
                iftransparent=args.iftransparent
            )
        logger.info("Done saving predicted masks")
    
    # Save ground truth masks
    if len(outputs[3]) > 0:
        gt_mask_types = [
            "gt_left_temporal_multi_masks",
            "gt_right_temporal_multi_masks"
        ]
        for mask_idx in range(len(outputs[3][0])):
            mask_type = gt_mask_types[mask_idx]
            logger.info(f"Saving {mask_type} for {test_seq_id}")
            gt_mask = [item[mask_idx] for item in outputs[3]]
            save_mask_maps(
                cfg.dataset_type, mask_type, test_seq_id, gt_mask,
                cfg.work_dir, checkpoint_name, first_frame_id=0
            )
        logger.info("Done saving GT masks")


def save_sequence_results(outputs, args, cfg, test_seq_id, logger):
    """
    Save all results for a single test sequence.
    
    Args:
        outputs: Model outputs containing [pred_depths, gt_depths, pred_masks, gt_masks, ...]
        args: Command line arguments
        cfg: Config object
        test_seq_id: Current test sequence ID
        logger: Logger instance
    """
    rank, _ = get_dist_info()
    
    if rank != 0:
        return
    
    logger.info(f"Saving results for sequence {test_seq_id}")
    
    # Save output pickle if requested
    if args.save_pkl:
        result_filename = f'test_outputs_{test_seq_id}.pkl'
        result_path = osp.join(cfg.work_dir, result_filename)
        logger.info(f'Writing results to {result_path}')
        mmcv.dump(outputs, result_path)
    
    # Determine evaluation tasks
    if "eval_tasks" not in cfg:
        eval_tasks = ["depth"] if len(outputs[0]) > 0 else []
        cfg.eval_tasks = eval_tasks
    
    # Save depth maps
    save_depth_results(outputs, args, cfg, test_seq_id, logger)
    
    # Save mask maps
    save_mask_results(outputs, args, cfg, test_seq_id, logger)
    
    logger.info(f"Finished saving results for sequence {test_seq_id}")


def evaluate_sequence(outputs, dataset, args, cfg, test_seq_id, logger, eval_config):
    """
    Evaluate results for a single test sequence using independent evaluators.
    
    Args:
        outputs: Model outputs [pred_depths, gt_depths, pred_masks, gt_masks, pred_poses, gt_poses]
        dataset: Dataset instance
        args: Command line arguments
        cfg: Config object
        test_seq_id: Current test sequence ID
        logger: Logger instance
        eval_config: Evaluation configuration
        
    Returns:
        eval_results: Evaluation metrics dictionary (or None if no evaluation)
    """
    from hdvo.evaluator import DepthEvaluator, PoseEvaluator
    
    rank, _ = get_dist_info()
    
    if rank != 0:
        return None
    
    if args.no_gt:
        logger.info(f"Skipping evaluation for sequence {test_seq_id} (no ground truth)")
        return None
    
    logger.info(f"Evaluating sequence {test_seq_id}...")
    
    eval_results = {}
    
    # ====================
    # Depth Evaluation
    # ====================
    try:
        if int(test_seq_id) <= 10:
            all_pred_depth = outputs[0]
            all_gt_depth = outputs[1]
            all_multimasks = outputs[2] if len(outputs) > 2 else []
            
            # Get sequence directory from dataset
            seq_dir = dataset.seq_dir if hasattr(dataset, 'seq_dir') else None
            gt_path = cfg.data.test.get('kitti_rawdata_path', None)
            
            # Initialize depth evaluator
            depth_evaluator = DepthEvaluator(
                min_depth=1.0,
                max_depth=80.0,
                garg_crop=True,
                eigen_crop=False,
                test_kbcrop=False
            )
            
            # Evaluate depth
            if len(all_gt_depth) > 0:
                eval_results['depth'] = depth_evaluator.evaluate(
                    all_pred_depth, 
                    all_gt_depth=all_gt_depth,
                    all_multimasks=all_multimasks,
                    seq_dir=seq_dir,
                    logger=logger
                )
            else:
                eval_results['depth'] = depth_evaluator.evaluate(
                    all_pred_depth,
                    all_gt_depth=None,
                    all_multimasks=all_multimasks,
                    seq_dir=seq_dir,
                    gt_path=gt_path,
                    logger=logger
                )
                
    except Exception as e:
        logger.error(f"Depth evaluation failed: {e}")
        import traceback
        traceback.print_exc()
    
    # ====================
    # Pose Evaluation
    # ====================
    if len(outputs) > 4 and len(outputs[4]) > 0:
        try:
            pred_relative_poses = outputs[4]
            gt_relative_poses = outputs[5] if len(outputs) > 5 else None
            
            # Initialize pose evaluator
            pose_evaluator = PoseEvaluator(
                work_dir=cfg.work_dir,
                test_seq_id=test_seq_id,
                dataset_type=cfg.dataset_type,
                apply_sim3_alignment=True
            )
            
            # Evaluate poses
            eval_results['pose'] = pose_evaluator.evaluate(
                pred_relative_poses,
                gt_relative_poses,
                logger=logger
            )
            
        except Exception as e:
            logger.error(f"Pose evaluation failed: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info(f"Completed evaluation for sequence {test_seq_id}\n")
    
    return eval_results


# ============================================================================
# Command Line Argument Parser
# ============================================================================

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
        'EPE 3PE D1')
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
        '--save_depth',
        action='store_true',
        help="if save prediction to the disk"
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
        default="99",
        help=" give the test sequences of the visual odometry"
    )
    parser.add_argument(
        '--test_range',
        type=str,
        default=None,
        help=" give the test sequences of the visual odometry"
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
    parser.add_argument(
        '--save_pkl',
        action='store_true',
        help=" if to save output .pkl for testing"
    )
    parser.add_argument(
        '--no_gt',
        action='store_true',
        help=" if to save output .pkl for testing"
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
    
    # Check if test config has multiple sequences
    base_test_cfg = cfg.data.test
    if isinstance(base_test_cfg, dict) and 'test_sequences' in base_test_cfg:
        # Multiple sequences in test_sequences list
        test_sequences = base_test_cfg.test_sequences
        logger.info(f'Testing on {len(test_sequences)} sequence(s) with shared config')
    elif isinstance(base_test_cfg, list):
        # Legacy format: list of complete configs
        test_sequences = [{'_full_config': cfg} for cfg in base_test_cfg]
        logger.info(f'Testing on {len(test_sequences)} sequence(s) with individual configs')
    else:
        # Single sequence
        test_sequences = [{}]
        logger.info('Testing on single sequence')
    
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

    # build the model and load checkpoint (only once)
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    #register_module_hooks(model.backbone, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint.split('.')[-1] != "pth": 
        print("## didn't load checkpoint")
    else:
        print("## loading checkpoint: {}".format(args.checkpoint))
        load_checkpoint(model, args.checkpoint, map_location='cpu')
        
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    
    # Wrap model for distributed/single GPU testing (only once)
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
    else:
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False)
    
    # Loop through each test sequence
    for test_idx, seq_info in enumerate(test_sequences):
        logger.info(f'\n{"="*60}')
        logger.info(f'Testing sequence {test_idx+1}/{len(test_sequences)}: {seq_info.get("test_seq_id", f"seq_{test_idx}")}')
        logger.info(f'{"="*60}\n')
        args.test_seq_id = seq_info.get('test_seq_id', cfg.get('test_seq_id', f"seq_{test_idx}"))
        # Build test config for current sequence
        if '_full_config' in seq_info:
            # Legacy format: use full config
            current_test_cfg = copy.deepcopy(seq_info['_full_config'])
        else:
            # New format: merge sequence info into base config
            current_test_cfg = copy.deepcopy(base_test_cfg)
            # Remove test_sequences from config to avoid confusion
            if 'test_sequences' in current_test_cfg:
                del current_test_cfg['test_sequences']
            # Update with sequence-specific info
            current_test_cfg.update(seq_info)
        
        current_test_cfg.test_mode = True
        
        # Override test_seq_id if provided in command line args
        if args.test_seq_id is not None and if_still_dataset==0:
            logger.info("# update the test seq id to -{}-  ".format(str(args.test_seq_id)))
            cfg.test_seq_id = args.test_seq_id
            if cfg.dataset_type == "VKITTI2StereoDataset" or cfg.dataset_type == "VKitti2Dataset":
                current_test_cfg.test_seq_id = "Scene"+args.test_seq_id
            else:
                current_test_cfg.test_seq_id = args.test_seq_id
        
        test_seq_id = current_test_cfg.get('test_seq_id', cfg.get('test_seq_id', '99'))
        logger.info(f'Test sequence ID: {test_seq_id}')
        
        if args.load_pred_depth is not None:
            print("directly load predicted depths")
            path_list = args.load_pred_depth.strip().split(',')
            pl_cfg = current_test_cfg.pipeline[-2:]
            for idx, pl in enumerate(pl_cfg):
                if len(path_list) == 1:
                    pl['keys'].append('left_pred_depths')
                if len(path_list) == 2:
                    pl['keys'].append('left_pred_depths')
                    pl['keys'].append('right_pred_depths')
            
            if "end_id" not in current_test_cfg:
                current_test_cfg['end_id'] = -1
            test_seq_id_for_load_depth = test_seq_id
            if cfg.dataset_type == "VKITTI2StereoDataset" or cfg.dataset_type == "VKitti2Dataset":
                test_seq_id_for_load_depth = args.test_seq_id if args.test_seq_id is not None else test_seq_id
            if len(path_list) == 1:
                current_test_cfg['pred_depth_dir_left'] =  osp.join(cfg.work_dir, f"{path_list[0]}_pred_depths_left_{cfg.dataset_type}_seq"+ test_seq_id_for_load_depth)
            if len(path_list) == 2:
                current_test_cfg['pred_depth_dir_left'] =  osp.join(cfg.work_dir, f"{path_list[0]}_pred_depths_left_{cfg.dataset_type}_seq"+ test_seq_id_for_load_depth)
                current_test_cfg['pred_depth_dir_right'] =  osp.join(cfg.work_dir, f"{path_list[0]}_pred_depths_right_{cfg.dataset_type}_seq"+ test_seq_id_for_load_depth)
        
        logger.info(f"Current test config keys: {list(current_test_cfg.keys())}")
        
        if args.test_range is not None:   
            logger.info("# update the test range to -{}-  ".format(str(args.test_range)))
            current_test_cfg.test_range = (int(args.test_range.split(',')[0]), int(args.test_range.split(',')[1]))
        
        # build the dataloader for current sequence
        dataset = build_dataset(current_test_cfg, dict(test_mode=True))
        dataloader_setting = dict(
            videos_per_gpu=cfg.data.get('videos_per_gpu', 1),
            workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
            dist=distributed,
            shuffle=False)
        dataloader_setting = dict(dataloader_setting,
                                  **cfg.data.get('test_dataloader', {}))
        data_loader = build_dataloader(dataset, **dataloader_setting)
        
        logger.info(f'Dataset size: {len(dataset)} samples')
 
        if args.output_pkl is not None:
            outputs = mmcv.load(args.output_pkl)
        else:
            outputs = single_gpu_test(model, data_loader) if not distributed else multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect)
            try:
                if hasattr(model, 'module') and hasattr(model.module, 'directvo_timer'):
                    print("\n #### direct vo running sum_time:{}, avg time: {} over {} frames ".format(
                        model.module.directvo_timer["sum_time"], 
                        model.module.directvo_timer["avg_time"], 
                        model.module.directvo_timer["frames"]))
            except:
                print("no directvo_timer")
        
        # Get model outputs
        logger.info(f"Sequence {test_seq_id} - {len(outputs)} output groups")
        """
        outputs format: [pred_depths, gt_depths, pred_masks, gt_masks, pred_poses, frame_dir]
        """
        
        # Save results
        save_sequence_results(outputs, args, cfg, test_seq_id, logger)
        
        # Evaluate results
        eval_results = evaluate_sequence(
            outputs, dataset, args, cfg, test_seq_id, logger, eval_config
        )
        
        # End of loop for current test sequence
        logger.info(f'Finished testing sequence {test_seq_id}\n')
    
    # All sequences tested
    logger.info(f'\n{"="*60}')
    logger.info(f'Completed testing all {len(test_sequences)} sequence(s)')
    logger.info(f'{"="*60}\n')
 






if __name__ == '__main__':
    main()
