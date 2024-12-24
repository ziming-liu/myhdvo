# Copyright (C) Huangying Zhan 2019. All rights reserved.

import argparse

from kitti_odometry import KittiEvalOdom

parser = argparse.ArgumentParser(description='KITTI evaluation')
parser.add_argument('--result', type=str, required=True,
                    help="Result directory")
parser.add_argument('--align', type=str, 
                    choices=['scale', 'scale_7dof', '7dof', '6dof'],
                    default=None,
                    help="alignment type")
parser.add_argument('--seqs', 
                    nargs="+",
                    type=str, 
                    help="sequences to be evaluated, support multi-pose-results for a same sequence. e.g. 09,09ep2,09ep3,09ep4 10,10ep2",
                    default=None)
parser.add_argument('--plot_keys', 
                    nargs="+",
                    type=str, 
                    help="plot keys, models names. Because of the limitation of the code, recomend to test a same sequence each time, if you give the plot keys for one sequence. e.g. 10epoch1 10epoch2 ",
                    default=None)
args = parser.parse_args()

eval_tool = KittiEvalOdom()
gt_dir = "dataset/kitti_odom/gt_poses/"
result_dir = args.result

print("Evaluate result in  {}".format(result_dir))
eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
        plot_keys=args.plot_keys
        )
"""
continue_flag = input("Evaluate result in {}? [y/n]".format(result_dir))
if continue_flag == "y":
    eval_tool.eval(
        gt_dir,
        result_dir,
        alignment=args.align,
        seqs=args.seqs,
        )
else:
    print("Double check the path!")
 """