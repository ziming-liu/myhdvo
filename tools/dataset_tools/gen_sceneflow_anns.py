'''
Author: DenseMatchingBenchmark
Date: 2022-10-18 23:38:05
LastEditors: Ziming Liu
LastEditTime: 2023-03-16 16:37:50
Description: this code is modified from github/DenseMatchingBenchmark
            script:  
                python  tools/dataset_tools/gen_sceneflow_anns.py --data-root   /home/ziliu/mydata/sceneflow  --save-annotation-root  /home/ziliu/mydata/sceneflow/annotations --data-type  clean
            print: 
                if don't consider the camera_data annotation txt. 
                SceneFlow Dataset contains:
                    35454   training samples
                    4370 validation samples
                Save to /home/ziliu/mydata/sceneflow/annotations/cleanpass_train.json
                Save to /home/ziliu/mydata/sceneflow/annotations/cleanpass_test.json
                if we load camera_data txt data, there are some lost samples whose ID==15, 
                e.g. tools/dataset_tools/gen_sceneflow_anns.py:60: UserWarning: /home/ziliu/mydata/sceneflow/flyingthings3d/camera_data/TEST/C/0064/camera_data.txt don't have ID 15 
                SceneFlow Dataset contains:
                    35397   training samples
                    4361 validation samples
                PS: the camera_data is extrinsics

                - we don't use monkaa and driving training set, only flying 3D is used, 
                  and test set is from flying3d also, as the previous works.
            mini version:
                SceneFlow Dataset contains:
                3546   training samples
                4370 validation samples
                Save to /home/ziliu/mydata/sceneflow/annotations/cleanpass_mini_train_all.json
                Save to /home/ziliu/mydata/sceneflow/annotations/cleanpass_mini_test.json

            only flying3d
                SceneFlow Dataset contains:
                22390   training samples
                4370 validation samples     
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

def getFlying3dMetas(root, Type, data_type='clean'):
    Metas = []

    imgDir = 'flyingthings3d/frames_' + data_type + 'pass'
    dispDir = 'flyingthings3d/disparity'
    camDir =  'flyingthings3d/camera_data'
    Parts = ['A', 'B', 'C']

    for Part in Parts:
        partDir = osp.join(root, dispDir, Type, Part)
        idxDirs = os.listdir(partDir)
        for idxDir in idxDirs:
            dispNames = os.listdir(osp.join(partDir, idxDir, 'left'))
            imgNames = ["{}.png".format(name.split('.')[0]) for name in dispNames]
            cam_wTc_txt = osp.join(root, camDir, Type, Part, idxDir, "camera_data.txt" )
            cam_wTc_left = dict()
            cam_wTc_right = dict()
            #print(cam_wTc_txt)
            with open(cam_wTc_txt, 'r') as Kf: 
                lines = Kf.readlines()
                for line_idx in range(len(lines)):
                    spt = lines[line_idx].strip().split(' ')
                    if spt[0] == "Frame":
                        left_wTc = lines[line_idx+1].strip().split(' ')
                        assert left_wTc[0]=='L'
                        cam_wTc_left[spt[1]] = " ".join(left_wTc[1:])
                        right_wTc = lines[line_idx+2].strip().split(' ')
                        assert right_wTc[0]=='R'
                        cam_wTc_right[spt[1]] = " ".join(right_wTc[1:])
                    else:
                        continue
            for imgName, dispName in zip(imgNames, dispNames):
                id = str(int(imgName.split('.')[0]))
                if id not in cam_wTc_left.keys():
                    warnings.warn(cam_wTc_txt +  " don't have ID " + id)
                    
                meta = dict(
                    left_image_path=osp.join(
                        imgDir, Type, Part, idxDir, 'left', imgName
                    ),
                    right_image_path=osp.join(
                        imgDir, Type, Part, idxDir, 'right', imgName
                    ),
                    left_disp_map_path=osp.join(
                        dispDir, Type, Part, idxDir, 'left', dispName
                    ),
                    right_disp_map_path=osp.join(
                        dispDir, Type, Part, idxDir, 'right', dispName
                    ),
                    left_wTc = cam_wTc_left[id] if id in cam_wTc_left.keys() else None,
                    right_wTc = cam_wTc_right[id] if id in cam_wTc_left.keys() else None,
                )
                Metas.append(meta)
    return Metas


def getMonkaaMetas(root, data_type='clean'):
    Metas = []

    imgDir = 'monkaa/frames_' + data_type + 'pass'
    dispDir = 'monkaa/disparity'
    camDir =  'monkaa/camera_data'

    sceneDirs = os.listdir(osp.join(root, dispDir))

    for sceneDir in sceneDirs:
        dispNames = os.listdir(osp.join(root, dispDir, sceneDir, 'left'))
        imgNames = ["{}.png".format(name.split('.')[0]) for name in dispNames]
        cam_wTc_txt = osp.join(root, camDir, sceneDir, "camera_data.txt" )
        cam_wTc_left = dict()
        cam_wTc_right = dict()
        with open(cam_wTc_txt, 'r') as Kf: 
            lines = Kf.readlines()
            for line_idx in range(len(lines)):
                spt = lines[line_idx].strip().split(' ')
                if spt[0] == "Frame":
                    left_wTc = lines[line_idx+1].strip().split(' ')
                    assert left_wTc[0]=='L'
                    cam_wTc_left[spt[1]] = " ".join(left_wTc[1:])
                    right_wTc = lines[line_idx+2].strip().split(' ')
                    assert right_wTc[0]=='R'
                    cam_wTc_right[spt[1]] = " ".join(right_wTc[1:])
                else:
                    continue
        for imgName, dispName in zip(imgNames, dispNames):
            meta = dict(
                left_image_path=osp.join(
                    imgDir, sceneDir, 'left', imgName
                ),
                right_image_path=osp.join(
                    imgDir, sceneDir, 'right', imgName
                ),
                left_disp_map_path=osp.join(
                    dispDir, sceneDir, 'left', dispName
                ),
                right_disp_map_path=osp.join(
                    dispDir, sceneDir, 'right', dispName
                ),
                left_wTc = cam_wTc_left[str(int(imgName.split('.')[0]))],
                right_wTc = cam_wTc_right[str(int(imgName.split('.')[0]))],
            )
            Metas.append(meta)
    return Metas


def getDrivingMetas(root, data_type='clean'):
    Metas = []

    imgDir = 'driving/frames_' + data_type + 'pass'
    dispDir = 'driving/disparity'
    camDir = 'driving/camera_data'
    focalLengthDirs = os.listdir(osp.join(root, dispDir))

    for focalLengthDir in focalLengthDirs:
        wardDirs = os.listdir(osp.join(root, dispDir, focalLengthDir))
        for wardDir in wardDirs:
            speedDirs = os.listdir(osp.join(root, dispDir, focalLengthDir, wardDir))
            for speedDir in speedDirs:
                dispNames = os.listdir(osp.join(root, dispDir, focalLengthDir, wardDir, speedDir, 'left'))
                imgNames = ["{}.png".format(name.split('.')[0]) for name in dispNames]
                cam_wTc_txt = osp.join(root, camDir, focalLengthDir, wardDir, speedDir, "camera_data.txt" )
                cam_wTc_left = dict()
                cam_wTc_right = dict()
                with open(cam_wTc_txt, 'r') as Kf: 
                    lines = Kf.readlines()
                    for line_idx in range(len(lines)):
                        spt = lines[line_idx].strip().split(' ')
                        if spt[0] == "Frame":
                            left_wTc = lines[line_idx+1].strip().split(' ')
                            assert left_wTc[0]=='L'
                            cam_wTc_left[spt[1]] = " ".join(left_wTc[1:])
                            right_wTc = lines[line_idx+2].strip().split(' ')
                            assert right_wTc[0]=='R'
                            cam_wTc_right[spt[1]] = " ".join(right_wTc[1:])
                        else:
                            continue
                for imgName, dispName in zip(imgNames, dispNames):
                    meta = dict(
                        left_image_path=osp.join(
                            imgDir, focalLengthDir, wardDir, speedDir, 'left', imgName
                        ),
                        right_image_path=osp.join(
                            imgDir, focalLengthDir, wardDir, speedDir, 'right', imgName
                        ),
                        left_disp_map_path=osp.join(
                            dispDir, focalLengthDir, wardDir, speedDir, 'left', dispName
                        ),
                        right_disp_map_path=osp.join(
                            dispDir, focalLengthDir, wardDir, speedDir, 'right', dispName
                        ),
                        left_wTc = cam_wTc_left[str(int(imgName.split('.')[0]))],
                        right_wTc = cam_wTc_right[str(int(imgName.split('.')[0]))],
                    )
                    Metas.append(meta)
    return Metas


def build_annoFile(root, save_annotation_root, data_type='clean',
 onlyflying3d=False, version="full"):
    """
    Build annotation files for Scene Flow Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    mkdir_or_exist(save_annotation_root)

    trainMetas = getFlying3dMetas(root, 'TRAIN', data_type)
    testMetas = getFlying3dMetas(root, 'TEST', data_type)

    if not onlyflying3d:
        trainMetas.extend(getMonkaaMetas(root, data_type))
        trainMetas.extend(getDrivingMetas(root, data_type))

    for meta in tqdm(trainMetas):
        for k, v in meta.items():
            if "path" in k:
                assert osp.exists(osp.join(root, v)), 'trainMetas:{} not exists'.format(v)
    
    if version == "mini":
        new_trainMetas = []
        for idx, meta in tqdm(enumerate(trainMetas)):
            if idx%10 == 0:
                new_trainMetas.append(meta)
        trainMetas = new_trainMetas
    
    if version == "mini":
        new_testMetas = []
        for idx, meta in tqdm(enumerate(testMetas)):
            if idx%10 == 0:
                new_testMetas.append(meta)
        testMetas = new_testMetas
    for meta in tqdm(testMetas):
        for k, v in meta.items():
            if "path" in k:
                assert osp.exists(osp.join(root, v)), 'testMetas: {} not exists'.format(v)

    info_str = 'SceneFlow Dataset contains:\n' \
               '    {:5d}   training samples \n' \
               '    {:5d} validation samples'.format(len(trainMetas), len(testMetas))
    print(info_str)

    def make_json(name, metas):
        if version=="mini":
            name = version + "_" + name
        filepath = osp.join(save_annotation_root, data_type + 'pass_' + name + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    if onlyflying3d:
        make_json(name='train_flying3d', metas=trainMetas)
        make_json(name='test', metas=testMetas)
    else:
        make_json(name='train_all', metas=trainMetas)
        make_json(name='test', metas=testMetas)


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
        "--data-type",
        default='clean',
        help="the type of data, (clean or final)pass",
        type=str,
    )
    parser.add_argument(
        "--onlyflying3d",
        default=False,
        help="in the training set, if only use flying3d subset",
         
    )
    parser.add_argument(
        "--version",
        default='full',
        help="full dataset, mini dataset",
        type=str,
    )
    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.data_type, args.onlyflying3d, 
                args.version)

 