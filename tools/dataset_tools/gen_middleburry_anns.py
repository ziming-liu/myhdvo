'''
Author: DenseMatchingBenchmark
Date: 2022-10-18 23:38:05
LastEditors: Ziming
LastEditTime: 2023-01-30 21:05:37
Description:  python tools/dataset_tools/gen_middleburry_anns.py --data-root /home/ziliu/mydata/middlebury --save-annotation-root  /home/ziliu/mydata/middlebury --version 2014
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

def getAnnotation(root, type, version):
    Metas = []
    assert version in ["eval3", "2014"]
    if version=="eval3":
        imgDir = f'MiddEval3/{type}F/'
        if type == "training":
            scenes = ['Adirondack' , 'ArtL' , 'Jadeplant' , 'Motorcycle' , 'MotorcycleE' , 'Piano'  ,'PianoL', 
            'Pipes' , 'Playroom' , 'Playtable' , 'PlaytableP' , 'Recycle' , 'Shelves' , 'Teddy' , 'Vintage']
        elif type=="test":
            scenes = ['Australia' , 'AustraliaP' , 'Bicycle2',  'Classroom2' , 'Classroom2E' , 'Computer' , 'Crusade' , 'CrusadeP', 
            'Djembe', 'DjembeL' , 'Hoops' , 'Livingroom' , 'Newkuba'  ,'Plants' , 'Staircase']
        for scn in scenes:
            path = osp.join(root, imgDir, scn)
            meta = dict(
                left_image_path=os.path.join(path, "im0.png"),
                right_image_path=os.path.join(path, "im1.png"),
                left_disp_map_path=os.path.join(path, "disp0GT.pfm"),
                right_disp_map_path=os.path.join(path, "disp1GT.pfm"),
                calib_path = os.path.join(path, "calib.txt"),
            )
            Metas.append(meta)


    elif version =="2014":
        
        if type=="training":
            imgDir = f"MiddleBury2014/"
            scenes = ['Adirondack-perfect' , 'Cable-perfect'  ,     'Flowers-perfect'  ,  'Motorcycle-perfect' , 'Playroom-perfect' ,
              'Shelves-perfect', 'Storage-perfect' , 'Umbrella-perfect' , 'Backpack-perfect'  ,  'Classroom1-perfect',  
              'Jadeplant-perfect',  'Piano-perfect',  'Playtable-perfect' , 'Shopvac-perfect' , 'Sword1-perfect' , 'Vintage-perfect',
'Bicycle1-perfect' ,   'Couch-perfect' , 'Mask-perfect' , 'Pipes-perfect' ,  'Recycle-perfect' ,   'Sticks-perfect' ,  'Sword2-perfect']
            for scn in scenes:
                path = osp.join(root, imgDir, scn)
                meta = dict(
                    left_image_path=os.path.join(path, "im0.png"),
                    right_image_path=os.path.join(path, "im1.png"),
                    left_disp_map_path=os.path.join(path, "disp0.pfm"),
                    right_disp_map_path=os.path.join(path, "disp1.pfm"),
                    calib_path = os.path.join(path, "calib.txt"),
                )
                Metas.append(meta)

        elif type == "test":
            imgDir = f'MiddEval3/testF/'
            scenes = ['Australia' , 'AustraliaP' , 'Bicycle2',  'Classroom2' , 'Classroom2E' , 'Computer' , 'Crusade' , 'CrusadeP', 
            'Djembe', 'DjembeL' , 'Hoops' , 'Livingroom' , 'Newkuba'  ,'Plants' , 'Staircase']
            for scn in scenes:
                path = osp.join(root, imgDir, scn)
                meta = dict(
                    left_image_path=os.path.join(path, "im0.png"),
                    right_image_path=os.path.join(path, "im1.png"),
                    left_disp_map_path=os.path.join(path, "disp0GT.pfm"),
                    right_disp_map_path=os.path.join(path, "disp1GT.pfm"),
                    calib_path = os.path.join(path, "calib.txt"),
                )
                Metas.append(meta)

    return Metas

""" 
def getAnnotation(root, type,):
    Metas = []
    imgDir = f'MiddEval3/{type}/'
    if type == "trainingF":
        scenes = ['Adirondack' , 'ArtL' , 'Jadeplant' , 'Motorcycle' , 'MotorcycleE' , 'Piano'  ,'PianoL', 
         'Pipes' , 'Playroom' , 'Playtable' , 'PlaytableP' , 'Recycle' , 'Shelves' , 'Teddy' , 'Vintage']
    elif type=="testF":
        scenes = ['Australia' , 'AustraliaP' , 'Bicycle2',  'Classroom2' , 'Classroom2E' , 'Computer' , 'Crusade' , 'CrusadeP', 
          'Djembe', 'DjembeL' , 'Hoops' , 'Livingroom' , 'Newkuba'  ,'Plants' , 'Staircase']
    
    for scn in scenes:
        path = osp.join(root, imgDir, scn)
        meta = dict(
            left_image_path=os.path.join(path, "im0.png"),
            right_image_path=os.path.join(path, "im1.png"),
            left_disp_map_path=os.path.join(path, "disp0GT.pfm"),
            right_disp_map_path=os.path.join(path, "disp1GT.pfm"),
            calib_path = os.path.join(path, "calib.txt"),
        )
        Metas.append(meta)
    return Metas
 """

def build_annoFile(root, save_annotation_root, version="2014"):
    """
    Build annotation files for Scene Flow Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    mkdir_or_exist(save_annotation_root)

    trainMetas = getAnnotation(root, 'training', version)
    testMetas = getAnnotation(root, 'test', version)

    info_str = 'MiddleBury Dataset contains:\n' \
               '    {:5d}   training samples \n' \
               '    {:5d}   test samples'.format(len(trainMetas), len(testMetas))
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, f"MiddleBury_{version}_{name}" + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    make_json(name='train', metas=trainMetas)
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
        "--version",
        default="2014",
        help="dataset version",
        type=str
    )
    args = parser.parse_args()
    build_annoFile(args.data_root, args.save_annotation_root, args.version)

 # python tools/dataset_tools/gen_middleburry_anns.py --data-root /home/ziliu/mydata/middlebury --save-annotation-root  /home/ziliu/mydata/middlebury --version 2014