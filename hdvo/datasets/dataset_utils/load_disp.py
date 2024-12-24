'''
Author: DenseMatchingBenchmark
Date: 2022-10-19 00:10:20
LastEditors: Ziming
LastEditTime: 2023-02-07 14:45:41
Description: this code provides the function to load disparity of SceneFlow dataset
Dependent packages: don't need any extral dependency
'''
import re
import numpy as np
import cv2 

def load_pfm(file_path):
    """
    load image in PFM type.
    Args:
        file_path string: file path(absolute)
    Returns:
        data (numpy.array): data of image in (Height, Width[, 3]) layout
        scale (float): scale of image
    """
    with open(file_path, encoding="ISO-8859-1") as fp:
        color = None
        width = None
        height = None
        scale = None
        endian = None

        # load file header and grab channels, if is 'PF' 3 channels else 1 channel(gray scale)
        header = fp.readline().rstrip()
        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
            
        # grab image dimensions
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', fp.readline())
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # grab image scale
        scale = float(fp.readline().rstrip())
        if scale < 0:  # little-endian
            endian = '<'
            scale = -scale
        else:
            endian = '>'  # big-endian

        # grab image data
        data = np.fromfile(fp, endian + 'f')
        shape = (height, width, 3) if color else (height, width)

        # reshape data to [Height, Width, Channels]
        data = np.reshape(data, shape)
        data = np.flipud(data)
        #print(scale)
        return data, scale

def read_pfm(pfm_file_path):
    with open(pfm_file_path, 'rb') as pfm_file:
        header = pfm_file.readline().decode().rstrip()
        channels = 3 if header == 'PF' else 1
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', pfm_file.readline().decode('utf-8'))
        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception("Malformed PFM header.")

        scale = float(pfm_file.readline().decode().rstrip())
        if scale < 0:
            endian = '<'  # littel endian
            scale = -scale
        else:
            endian = '>'  # big endian

        dispariy = np.fromfile(pfm_file, endian + 'f')

    img = np.reshape(dispariy, newshape=(height, width, channels))
    img = np.flipud(img).astype('uint8')
    #show(img, "disparity")
    return dispariy, scale #[(height, width, channels), scale]

# load utils
def load_scene_flow_disp(img_path):
    """load scene flow disparity image
    Args:
        img_path:
    Returns:
    """
    assert img_path.endswith('.pfm'), "scene flow disparity image must end with .pfm" \
                                      "but got {}".format(img_path)

    disp_img, scale = load_pfm(img_path)
    #if max(disp_img.reshape(-1)) > 300:
    #    print(f"disp gt {disp_img} \n scale {scale}")
    return disp_img

def load_crestereo_disp(disp_path):
    disp = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED)
    return disp.astype(np.float32) / 32

def load_middlebury_disp(disp_path):
    assert disp_path.endswith('.pfm')
    disp, _ = load_pfm(disp_path)
    return disp
