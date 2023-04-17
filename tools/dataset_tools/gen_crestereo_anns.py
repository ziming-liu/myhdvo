'''
Author: DenseMatchingBenchmark
Date: 2022-10-18 23:38:05
LastEditors: Ziming
LastEditTime: 2023-01-30 22:25:19
Description:  
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
    
    imgDir = "stereo_trainset/crestereo"
    assert type=="training"
    scenes = ['hole' , 'reflective' , 'shapenet' , 'tree']
    for scn in scenes:
        print("scene>>  ", scn)
        path = osp.join(root, imgDir, scn)
        items = os.listdir(path)
        name_list = []
        for item in tqdm(items):
            if item.split('_')[0] not in name_list:
                name_list.append(item.split('_')[0])
        for name in tqdm(name_list):
            meta = dict(
                left_image_path=os.path.join(path, f"{name}_left.jpg"),
                right_image_path=os.path.join(path,  f"{name}_right.jpg"),
                left_disp_map_path=os.path.join(path, f"{name}_left.disp.png"),
                right_disp_map_path=os.path.join(path, f"{name}_right.disp.png"),
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

def build_annoFile(root, save_annotation_root, version="full"):
    """
    Build annotation files for Scene Flow Dataset.
    Args:
        root:
    """
    # check existence
    assert osp.exists(root), 'Path: {} not exists!'.format(root)
    mkdir_or_exist(save_annotation_root)

    trainMetas = getAnnotation(root, 'training', version)
    #testMetas = getAnnotation(root, 'test', version)

    info_str = 'crestereo Dataset contains:\n' \
               '    {:5d}   training samples \n'.format(len(trainMetas))
    print(info_str)

    def make_json(name, metas):
        filepath = osp.join(save_annotation_root, f"crestereo_{version}_{name}" + '.json')
        print('Save to {}'.format(filepath))
        with open(file=filepath, mode='w') as fp:
            json.dump(metas, fp=fp)

    make_json(name='train', metas=trainMetas)
    #make_json(name='test', metas=testMetas)


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

 # python tools/dataset_tools/gen_crestereo_anns.py --data-root /home/ziliu/mydata/crestereo --save-annotation-root  /home/ziliu/mydata/crestereo --version full
