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
import time
import matplotlib as mp
import sys
from torch.nn import functional as F
import torch.optim as optim
from outputs_proc.save_load_mask import save_mask_maps
from outputs_proc.save_load_depth import save_depth_maps

sys.path.append("../")
from kitti_odom_eval.kitti_odometry import KittiEvalOdom
from KITTI_odometry_evaluation_tool.evaluation import kittiOdomEval
mp.use("pdf")

#cmap = plt.cm.viridis
plasma = plt.get_cmap('magma')

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.cnn import fuse_conv_bn
from mmcv.fileio.io import file_handlers
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint,save_checkpoint
from mmcv.runner.fp16_utils import wrap_fp16_model
from mmcv.runner import LogBuffer

from zimingdepth.apis import multi_gpu_test, single_gpu_test
from zimingdepth.datasets import build_dataloader, build_dataset
from zimingdepth.models import build_model
from zimingdepth.utils import collect_env, get_root_logger, register_module_hooks
from eval_odometry import dr_api,dr_api2

def parse_args():
    parser = argparse.ArgumentParser(
        description='MMAction2 test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', default=None, help='checkpoint file')
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
        '--save_depth',
        default=False,
        help="if save prediction to the disk"
    )
    parser.add_argument(
        '--test_time_training',
        default=False,
        help="test time training"
    )
    parser.add_argument(
        '--seq_optim',
        default=False,
        help="test time training"
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
        '--max_depth',
        type=int,
        default=655,
        help=" The max depth values in the prediction map"
    )
    parser.add_argument(
        '--bs',
        type=int,
        default=1,
        help=" The batch size "
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help=" The learning rate for test time training"
    )
    parser.add_argument(
        '--output_pkl',
        type=str,
        default=None,
        help=" directly load output .pkl for testing"
    )
    parser.add_argument(
        '--saving_name',
        type=str,
        default="Default",
        help=" saving_name for depth dirs."
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
    if not os.path.exists(cfg.work_dir):
        os.makedirs(cfg.work_dir)
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
        videos_per_gpu= args.bs, #cfg.data.get('videos_per_gpu', 1),
        workers_per_gpu=cfg.data.get('workers_per_gpu', 1),
        dist=distributed,
        shuffle=False)
    dataloader_setting = dict(dataloader_setting,
                              **cfg.data.get('test_dataloader', {}))
    data_loader = build_dataloader(dataset, **dataloader_setting)

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    
    base_model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))
    #register_module_hooks(model.backbone, cfg.module_hooks)

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        wrap_fp16_model(model)
    if args.checkpoint is None or args.checkpoint.split('.')[-1] != "pth": 
        print("## didn't load checkpoint")
    else:
        print("## loading checkpoint: {}".format(args.checkpoint))
        load_checkpoint(model, args.checkpoint, map_location='cpu')
    
    #save_checkpoint(model, "./temp.pth")
    #print("## saved loaded model parameters into ./temp.pth")
    if args.fuse_conv_bn:
        model = fuse_conv_bn(model)
    """
    if os.path.exists(args.out):
        rank, _ = get_dist_info()
        if rank == 0:
            outputs = mmcv.load(args.out)
            if eval_config:
                eval_res = dataset.evaluate(outputs, **eval_config)
                for name, val in eval_res.items():
                    print(f'{name}: {val:.04f}')
            if args.vis:
                vis_traj(dataset.video_infos, outputs, cfg, True)
        exit()
    """
    if args.test_time_training:
        optimizer = optim.AdamW(model.parameters(), args.lr)

    model = MMDataParallel(model, device_ids=[0])
    base_model = MMDataParallel(base_model, device_ids=[0])
    #model.eval()
    #results = []
    results = [[] for _ in range(10)] # depth, pose, seq_dir
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for idx, data in enumerate(data_loader):
        # firstly train onece with unsupervised loss, SPR loss. 
        #print(data["left_imgs"].shape)
        if args.test_time_training:
            #if not args.seq_optim:
            #    load_checkpoint(model, "./temp.pth",)
            model.train()
            optimizer.zero_grad()
            out = model.train_step(data, optimizer)
            losses = out["loss"]
            #l = sum(_value for _key, _value in losses.items())
            #l = torch.sum(l)
            losses.backward()
            optimizer.step() 

        model.eval()
        with torch.no_grad():
            result = model.val_step(data, None)
        num_output = len(result)
        for k in range(num_output):
            results[k].extend(result[k])
        #results.extend(result)

        if args.test_time_training:
            if not args.seq_optim:
                #load_checkpoint(model, "./temp.pth",)
                model.load_state_dict(copy.deepcopy(base_model.state_dict()))
                optimizer = optim.AdamW(model.parameters(), args.lr)
            elif idx % int(args.seq_optim)==0:
                #print(f"load dict at s {idx}")
                model.load_state_dict(copy.deepcopy(base_model.state_dict()))
                optimizer = optim.AdamW(model.parameters(), args.lr)

        # use the first key as main key to calculate the batch size
        batch_size = len(next(iter(data.values())))
        for _ in range(batch_size):
            prog_bar.update()
    outputs = results

    rank, _ = get_dist_info()
    
    num_outputs = len(outputs)
    """
    outputs = [[], [], [], [], [] ] # [pred_depths], [gt_depths], [pred_masks], [gt_masks], [pred_poses], frame_dir
    """

    if "eval_tasks" in cfg:
        task_flag = True
        eval_tasks = cfg.eval_tasks
    else:
        task_flag = False
        eval_tasks = []
        eval_tasks.append("depth")
        #eval_tasks.append("pose")
        cfg.eval_tasks = eval_tasks
    

    # evaluate the results of depth estimation
    #if 'depth' in cfg.eval_tasks and cfg.work_dir is not None and 
    if rank == 0:
        # save results to cfg.work_dir
        print(f"---save testouputs into {cfg.work_dir}>> 'test_outputs.pkl'")
        result_path = osp.join(cfg.work_dir, 'test_outputs.pkl')
        logger.info('\nwriting depth results to {}'.format(result_path))
        mmcv.dump(outputs, result_path)  
        print("#done")
        
        if args.checkpoint.split('.')[-1]=='pth':
            args.saving_name = args.checkpoint.split('/')[-1].split('.')[0]
        print("saving name ", args.saving_name)
        print(args.load_pred_depth)

        if len(outputs[0]) == 0:
            eval_tasks = []
        if args.load_pred_depth is None and len(outputs[0])>0: # if we use existing depth, we don;t need to save again.
            from outputs_proc.save_load_depth import save_depth_maps
            pred_depth = outputs[0].copy()
            pred_depth = [ item[0] for item in pred_depth] # left depths
            for i in range(len(pred_depth)):
                pred_depth[i][pred_depth[i]>args.max_depth] = args.max_depth
                pred_depth[i][np.isnan(pred_depth[i])] = args.max_depth
                pred_depth[i][np.isinf(pred_depth[i])] = args.max_depth
            if args.save_depth:
                print(f"--- save left depth maps into {cfg.work_dir}")
                save_depth_maps(cfg.dataset_type, test_seq_id, pred_depth, cfg.work_dir, \
                    args.saving_name, stereo_view="left", \
                        min_depth =  1, max_depth=args.max_depth,first_frame_id=0  )
            
            gt_depth = outputs[1].copy()
            gt_depth = [ item[0] for item in gt_depth] # gt left depths
            if args.save_depth:
                print(f"--- save GT left depth maps into {cfg.work_dir}")
                save_depth_maps(cfg.dataset_type, test_seq_id, gt_depth, cfg.work_dir, \
                    args.saving_name, stereo_view="left", ifgtdepth=True, \
                        min_depth =  1, max_depth=args.max_depth,first_frame_id=0  )

            print("#done")
            if len(outputs[0][0])==2 and args.save_depth: # left+right pred depths
                print(f"--- save right depth maps into {cfg.work_dir}")
                pred_depth = outputs[0].copy()
                pred_depth = [item[1] for item in pred_depth] # right depths
                save_depth_maps(cfg.dataset_type, test_seq_id, pred_depth, cfg.work_dir, \
                args.saving_name, stereo_view="right", \
                    min_depth =  1, max_depth=6000,first_frame_id=0  )
                print("#done")

        from outputs_proc.save_load_mask import save_mask_maps
        
        # save pred masks (multi)
        print("output2 pred_mask: ",len(outputs[3]) )
        if len(outputs[2])>0: # pred mask is not [] 
            if outputs[2][0].shape[0]==6: # visualize all masks
                pred_mask_types = ["left_temporal_multi_masks","left_homo_mask",
                                "left_stc_t_mask", "left_stc_s_mask",
                                "left_temporal_edge_mask", "left_stereo_edge_mask" ]
            else:
                pred_mask_types = ["left_temporal_multi_masks","right_temporal_multi_masks", ]
            if isinstance(outputs[2][0], list):
                num_mask_types = len(outputs[2][0])
            else:
                num_mask_types = outputs[2][0].shape[0]
            for mask_idx in range(num_mask_types):
                mask_type = pred_mask_types[mask_idx]
                print(f"--- save {mask_type} maps into {cfg.work_dir}")
                pred_mask = [item[mask_idx] for item in outputs[2]]
                save_mask_maps(cfg.dataset_type, mask_type, test_seq_id, pred_mask, cfg.work_dir, \
                    args.saving_name, first_frame_id=0, iftransparent=args.iftransparent )
            print("#done")
        # save GT mask (for vkitti2)
        print("output3 GTmask: ",len(outputs[3]) )
        if len(outputs[3])>0: # pred mask is not [] 
            gt_mask_types =  ["gt_left_temporal_multi_masks","gt_right_temporal_multi_masks", ]
            for mask_idx in range(len(outputs[3][0])):
                mask_type = gt_mask_types[mask_idx]
                print(f"--- save {mask_type} maps into {cfg.work_dir}")
                gt_mask = [item[mask_idx] for item in outputs[3]]
                save_mask_maps(cfg.dataset_type, mask_type, test_seq_id, gt_mask, cfg.work_dir, \
                    args.saving_name, first_frame_id=0 )
                print("#done")

        def pose_relative2absolute(relative_poses, first_abs_pose=np.eye(4)):
            '''
            description: transfrom relative camera pose into absolute camera pose
            parameter: {reltive_poses: [list], camera poses from the reference to the current, i.e. rTc}
            return: {absolute_poses: [list]}
            '''            
            abs_poses = [first_abs_pose, ]
            num = len(relative_poses)
            print(f"FUNC pose_relative2absolute: \n given {num} relative poses, init first camera pose with \n____\n {first_abs_pose}. \n ___")
            ref_abs_pose = first_abs_pose
            for p_idx in range(num):
                cTr = relative_poses[p_idx]
                cur_abs_pose = np.matmul(ref_abs_pose, np.linalg.inv(cTr))
                theta = -0.5*np.pi
                y_trans_ = np.reshape(np.array([np.cos(theta), 0, -np.sin(theta), 0,1,0, np.sin(theta), 0, np.cos(theta)]), (3,3))
                y_trans = np.eye(4)
                y_trans[:3,:3] = y_trans_
                theta = 0.5*np.pi
                z_trans_ =np.reshape(np.array([np.cos(theta), np.sin(theta), 0, -np.sin(theta), np.cos(theta), 0, 0, 0, 1]), (3,3))
                z_trans = np.eye(4)
                z_trans[:3,:3] = z_trans_
                #cur_abs_pose = np.matmul(cur_abs_pose, np.linalg.inv(y_trans))
                #cur_abs_pose = np.matmul(cur_abs_pose, np.linalg.inv(z_trans))
                
                abs_poses.append(cur_abs_pose)
                ref_abs_pose = cur_abs_pose
            print(f"FUNC pose_relative2absolute: output {len(abs_poses)} absolute poses.")
            return abs_poses
        if len(outputs[4])>0 and (cfg.dataset_type == "VKITTI2StereoDataset" or cfg.dataset_type == "KittiOdometryDataset" or cfg.dataset_type=="EurocMavDataset" or cfg.dataset_type=="MidAirDataset"):
            # save pred pose into .txt with kitti format
            print("save pred pose into .txt with kitti format")
            pred_relative_pose_list = outputs[4].copy()
            pred_abs_pose_list = pose_relative2absolute(pred_relative_pose_list)
            time_str = '_'.join( time.asctime(time.localtime()).split(' ') )
            time_str = '_'.join( time_str.split(':') )
            result_dir = os.path.join(cfg.work_dir, f"pred_poses_{test_seq_id}_"+time_str+str(random.randrange(10000,19999)))
            pred_pose_path = os.path.join(result_dir, "{}.txt".format(test_seq_id))
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)
            with open(pred_pose_path, 'w') as f:
                for p_idx in range(len(pred_abs_pose_list)):
                    str_pose =  [  str(pp) for pp in pred_abs_pose_list[p_idx].reshape(-1)[:12].tolist()]
                    f.write(' '.join(str_pose)+'\n')
            # save gt pose into .txt with kitti format
            gt_relative_pose_list = outputs[5].copy()
            gt_abs_pose_list = pose_relative2absolute(gt_relative_pose_list)
            gt_pose_path = os.path.join(cfg.work_dir, "gt_poses", "{}.txt".format(test_seq_id))
            gt_dir = os.path.join(cfg.work_dir, "gt_poses")
            if not os.path.exists(gt_dir):
                os.makedirs(gt_dir)
            with open(gt_pose_path, 'w') as f:
                for p_idx in range(len(gt_abs_pose_list)):
                    str_pose =  [  str(pp) for pp in gt_abs_pose_list[p_idx].reshape(-1)[:12].tolist()]
                    f.write(' '.join(str_pose)+'\n')

            # evaluate pose estimation 
            
            eval_tool = KittiEvalOdom()
            if cfg.dataset_type == "KittiOdometryDataset":
                gt_dir = "kitti_odom_eval/dataset/kitti_odom/gt_poses"
            elif cfg.dataset_type == "VKITTI2StereoDataset":
                gt_dir = gt_dir
            elif cfg.dataset_type == "EurocMavDataset":
                gt_dir = gt_dir
            
            print("Evaluate result in  {}".format(result_dir))
            print(test_seq_id)

            if cfg.dataset_type != "EurocMavDataset" and cfg.dataset_type != "MidAirDataset" :
                eval_tool.eval(
                        gt_dir,
                        result_dir,
                        alignment=None, # ['scale', 'scale_7dof', '7dof', '6dof'],
                        seqs=[str(test_seq_id)], # e.g. 09,09ep2,09ep3,09ep4 10,10ep2",
                        plot_keys=[str(test_seq_id)+str(time.time())] # + args.checkpoint.split('.')[-2].split('/')[-1]  #  e.g. 10epoch1 10epoch2 ",
                        )
            
            # eval tool2
            dict_tool2 = {"gt_dir":gt_dir, "result_dir": result_dir, "eva_seqs": f"{test_seq_id}_pred",\
                "toCameraCoord": False}
            print(dict_tool2)
            pose_eval = kittiOdomEval(dict_tool2)
            pose_eval.eval(toCameraCoord=dict_tool2['toCameraCoord'])   # set the value according to the predicted results


        # if there are right depths map output use this
        #pred_depth = outputs[2]
        #save_depth_maps(pred_depth, "right")
        ###############################
        # perform dense vo module 
        ##############################
        #if int(test_seq_id)>10 and "pose" in eval_tasks:
        #    eval_tasks.remove("pose")
        """     
        if "pose" in eval_tasks:
            if cfg.dataset_type == "KittiOdometryDataset":
                
                type_dataset = "kittiodometry"
                image_path = dataset.data_prefix
                depth_path = depth_dir
                pred_pose_dir = os.path.join(cfg.work_dir, "pred_poses")
                gt_pose_dir = os.path.join(cfg.work_dir, "gt_poses")
                start_id = -1 
                end_id = num_samples
                dr_api2(type_dataset,image_path, depth_path, pred_pose_dir, gt_pose_dir,test_seq_id, start_id,end_id,0  )
                eval_config["pred_pose_dir"] = pred_pose_dir
                eval_config["gt_pose_dir"] = gt_pose_dir
                eval_config["test_seq_id"] = test_seq_id
            if cfg.dataset_type == "VKITTI2StereoDataset":
                type_dataset = "vkitti2"
                image_path = os.path.join(dataset.data_prefix, test_seq_id, "15-deg-left/") 
                depth_path = depth_dir
                pred_pose_dir = os.path.join(cfg.work_dir, "pred_poses")
                gt_pose_dir = os.path.join(cfg.work_dir, "gt_poses")
                start_id = -1 
                end_id = num_samples
                dr_api2(type_dataset,image_path, depth_path, pred_pose_dir, gt_pose_dir,test_seq_id[-2:], start_id,end_id,0  )
                eval_config["pred_pose_dir"] = pred_pose_dir
                eval_config["gt_pose_dir"] = gt_pose_dir
                eval_config["test_seq_id"] = test_seq_id
        """
        #if "pose_eval_dataset" in cfg:
        #    dr_api(cfg["pose_eval_dataset"], cfg["rgb_path"], depth_dir, 
         #       os.path.join(cfg["work_dir"],"pred_pose_seq",test_seq_id),os.path.join(cfg["work_dir"],"gt_pose_seq",test_seq_id), test_seq_id)
        if eval_config:
            eval_config["cfg"] = cfg.copy()
            eval_depth_res, eval_pose_res = dataset.evaluate(outputs, eval_tasks=eval_tasks, **eval_config)
            for name, val in eval_depth_res.items():
                logger.info(f'{name}: {val:.04f}')
            for name, val in eval_pose_res.items():
                logger.info(f'{name}: {val:.04f}')
        print("end evaluation! ")
        if args.vis:
            pass
            #vis_traj(dataset.video_infos, outputs, cfg, True)
            #vis_traj(gt_answers, outputs, cfg, True)
        # remove the internal tmp files .pkl in .dist_test
        import shutil
        if os.path.exists("./.dist_test"):
            shutil.rmtree("./.dist_test")

 
def eval_pose(gt_dir, result_dir, alignment, test_seq_id, ):
    """
    python tools/test_with_training.py  eval_pose  work_dirs/kittiodom_base2_resnet2_mono2stereo_sup_unsup/gt_poses/  work_dirs/kittiodom_base2_resnet2_mono2stereo_sup_unsup/pred_poses_10_Thu_Feb_16_12_20_34_2023/  '6dof'   10
    """
    eval_tool = KittiEvalOdom()
    eval_tool.eval(
            gt_dir,
            result_dir,
            alignment, # ['scale', 'scale_7dof', '7dof', '6dof'],
            [str(test_seq_id)],#=[str(test_seq_id)], # e.g. 09,09ep2,09ep3,09ep4 10,10ep2",
            plot_keys=[str(test_seq_id)+"plot"] # + args.checkpoint.split('.')[-2].split('/')[-1]  #  e.g. 10epoch1 10epoch2 ",
            )

if __name__ == '__main__':
    import fire
    #fire.Fire()
    main()
    
