'''
Developer: ACENTAURI team, INRIA institute
Author: Ziming Liu
Date: 2023-03-27 18:41:32
LastEditors: Ziming Liu
LastEditTime: 2023-03-27 21:13:31
'''
import numpy as np
import cv2
import os
from tqdm import tqdm


def kittidepth_depth_read(filename):
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    #depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    #assert(np.max(depth_png) > 255)
    depth_png = cv2.imread(filename, cv2.IMREAD_UNCHANGED).astype('float32')
    #print("d1 >> ", depth_png[200:300,500:700])
    #print("d2>> ", d2[200:300,500:700])
    depth = depth_png.astype(np.float32) / 256.
    depth[depth_png == 0] = -1
    #print("d2>> ", depth[200:300,500:700])
    return depth

def kitti_depth2disp(depth_path, disp_path, focal):
    """Convert kitti depth to disparity.
    Args:
        depth_path (str): Path to kitti depth.
        disp_path (str): Path to kitti disparity.
        max_depth (int): Maximum depth value.
    """
    baseline = (-3.395242e+02 / -7.215377e+02) - (4.485728e+01 / -7.215377e+02)
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED).astype('float32') / 256
    depth[depth==0] = -1
    disp = (focal*baseline) / depth
    disp[disp<0] = 0
    disp = (disp * 256).astype(np.uint16)
    cv2.imwrite(disp_path, disp, [cv2.IMWRITE_PNG_COMPRESSION, 0])

def main( annotation, root="/home/ziliu/mydata/kitti_raw_data", depth_root="/home/ziliu/mydata/kitti_raw_depth"):
    with open(annotation, 'r') as f:
        anns = f.readlines()
        for ann in tqdm(anns):
            ann = ann.split()
            if ann[1] == "None":
                continue
            depth_path = os.path.join(depth_root, ann[1])
            new_disp_path  = os.path.join(depth_root, ann[1].replace("depth", "disp"))
            assert depth_path != new_disp_path
            if not os.path.exists('/'.join(new_disp_path.split('/')[:-1])):
                os.makedirs('/'.join(new_disp_path.split('/')[:-1]))
            if not os.path.exists(new_disp_path):
                kitti_depth2disp(depth_path, new_disp_path, float(ann[2]))


if __name__ == '__main__':
    import fire
    fire.Fire()