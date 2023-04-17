'''
Author: DenseMatchingBenchmark
Date: 2022-10-18 23:38:05
LastEditors: Ziming Liu
LastEditTime: 2023-03-27 16:40:05
Description:  drivingstereo Dataset contains:
    174437   training samples

Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_train.json
Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_test.json

Dependent packages: mmcv

'''
import os
import numpy as np
import argparse
import os.path as osp
import json
from tqdm import tqdm
from mmcv import mkdir_or_exist
import warnings
import cv2 

def gettrainingAnnotation(path, ):
    Metas = []
    smallest_h = 100000
    biggest_h = 0
    left_w = 1000
    right_w = 0
    scenes = ['train_left_image', ]
    for scn in scenes:
        print("scene>>  ", scn)
        seq_lists = os.listdir(os.path.join(path, scn))
        seq_lists = sorted(seq_lists)
        for seqi in seq_lists:
            imgs = os.listdir(os.path.join(path,scn,seqi))
            imgs = sorted(imgs)
            for imgi in tqdm(imgs):
                name = imgi.split('.')[0]
                meta = dict(
                    left_image_path=os.path.join(path, "train_left_image", seqi, f"{name}.jpg"),
                    right_image_path=os.path.join(path,  "train_right_image", seqi, f"{name}.jpg"),
                    left_disp_map_path=os.path.join(path, "train_disparity_map", seqi, f"{name}.png"),
                    right_disp_map_path=None,
                    calib_path = os.path.join(path, "calib/half-image-calib", seqi+".txt"),
                )
                disp = cv2.imread(meta["left_disp_map_path"], cv2.IMREAD_UNCHANGED).astype('float32') / 256
                H, W = disp.shape
                for hi in range(H):
                    if sum(disp[hi,:]) !=0:
                        smallest_h = hi
                        break
                for hi in range(H-1, -1, -1):
                    if sum(disp[hi,:]) !=0:
                        biggest_h = hi
                        break
                for wi in range(W):
                    if sum(disp[:,wi]) !=0:
                        left_w = wi
                        break
                for wi in range(W-1, -1, -1):
                    if sum(disp[:,wi]) !=0:
                        right_w = wi
                        break
                Metas.append(meta)
            print("smallest_h", smallest_h)
            print("biggest_h", biggest_h)
            print("left_w", left_w)
            print("right_w", right_w)
            print("crop H ",biggest_h-smallest_h)
            print("crop W ",right_w-left_w)

    return Metas

def gettestAnnotation(path, version="half"):
    Metas = []
    
    scenes = [f'test_left_image/left_image_{version}_size', ]
    for scn in scenes:
        print("scene>>  ", scn)
        seq_lists = os.listdir(os.path.join(path, scn))
        seqMetas = {}
        for seq_name in seq_lists:
            seqMetas[seq_name] = []
        seq_lists = sorted(seq_lists)
        for seqi in seq_lists:
            imgs = os.listdir(os.path.join(path,scn,seqi))
            imgs = sorted(imgs)
            for imgi in tqdm(imgs):
                name = imgi.split('.')[0]
                meta = dict(
                    left_image_path=os.path.join(path, f"test_left_image/left_image_{version}_size", seqi, f"{name}.jpg"),
                    right_image_path=os.path.join(path,  f"test_right_image/right_image_{version}_size", seqi, f"{name}.jpg"),
                    left_disp_map_path=os.path.join(path, f"test_disparity_map/disparity_map_{version}_size", seqi, f"{name}.png"),
                    right_disp_map_path=None,
                    calib_path = os.path.join(path, f"test-calib/{version}-image-calib", seqi+".txt"),
                )
                Metas.append(meta)
                seqMetas[seqi].append(meta)
 

    return Metas, seqMetas


def build_annoFile(root, save_annotation_root, version="half"):
    """
    Build annotation files for Scene Flow Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    mkdir_or_exist(save_annotation_root)

    trainMetas = gettrainingAnnotation(root,)
    testMetas, test_seqMetas = gettestAnnotation(root,  version)

    info_str = 'drivingstereo Dataset contains:\n' \
               '    {:5d}   training samples \n'.format(len(trainMetas))
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, f"drivingstereo_{version}_{name}" + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    make_json(name='train', metas=trainMetas)
    make_json(name='test', metas=testMetas)
    for seq_name, seq_meta in test_seqMetas.items():
        make_json(name=f'test_{seq_name}', metas=seq_meta)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="SceneFlow Data PreProcess.")
    parser.add_argument(
        "--data-root",
        default=None,
        help="root of data",
        type=str,
    )
    parser.add_argument(
        "--save-annotation-root",
        default='./',
        help="save root of generated annotation file",
        type=str,
    )
    parser.add_argument(
        "--version",
        default="half",
        help="dataset version",
        type=str
    )
    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.version)

 # python tools/dataset_tools/gen_drivingstereo_anns.py --data-root /home/ziliu/mydata/drivingstereo --save-annotation-root  /home/ziliu/mydata/drivingstereo --version half

"""
crestereo Dataset contains:
    174437   training samples

Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_train.json
Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_test.json
Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_test_2018-10-11-16-03-19.json
Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_test_2018-08-01-11-13-14.json
Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_test_2018-08-07-13-46-08.json
Save to /home/ziliu/mydata/drivingstereo/drivingstereo_half_test_2018-07-11-14-48-52.json

"""