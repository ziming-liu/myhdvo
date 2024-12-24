import torch 
import torch.nn as nn
import torch.nn.functional as F


def rox_matse3_update_right(pose, solution_pose):
    algebra_se3 = rox_algse3_set_velocity(solution_pose)
    update_matse3 = rox_matse3_exponential_algse3(algebra_se3)
    return torch.mm(pose, update_matse3)


def rox_matse3_exponential_algse3( algebra_se3 ):
    algebra_se3 = algebra_se3.double()
    update_matse3 = torch.zeros((4,4), dtype=torch.float64, device=algebra_se3.device)
    th = 0.0
    ux = 0.0
    uy = 0.0
    uz = 1.0

    vx = 0.0
    vy = 0.0
    vz = 0.0

    th_inv = 0.0

    # limiting values l'Hospital

    sin_th = 0.0
    cos_th = 1.0

    # sin_th/th
    sin_th_inv = 1.0

    # (1.0-cos_th)/th
    cos_th_inv = 0.0

    # A = [0 -uz*th +uy*th vx +uz*th 0 -ux*th vy -uy*th +ux*th 0 vz 0 0 0 0]
    # R = I + sin_th*[u] + (1-cos_th)*[u]^2 
    # Q = I + (1-cos_th)*[u]/th + (1-sin_th/th)*[u]^2 
    # t = Q*v
    # T = [R t 0 0 0 1]

    vx = algebra_se3[0][3]
    vy = algebra_se3[1][3]
    vz = algebra_se3[2][3]

    th = torch.sqrt(algebra_se3[1][2] * algebra_se3[1][2] + algebra_se3[0][1] * algebra_se3[0][1] + algebra_se3[2][0] * algebra_se3[2][0])

    # avoid division through 0
    if (th > torch.finfo(torch.float64).eps): # double 
        sin_th = torch.sin(th)
        cos_th = torch.cos(th)

        th_inv = 1.0 / th
        sin_th_inv = sin_th * th_inv
        cos_th_inv = (1.0 - cos_th) * th_inv

        # Axis of rotation such that ux^2+uy^2+uz^2-1=0
        ux = algebra_se3[2][1] * th_inv
        uy = algebra_se3[0][2] * th_inv
        uz = algebra_se3[1][0] * th_inv
    

    # Rotation matrix
    update_matse3[0][0] = 1.0 + (1.0 - cos_th) * (ux * ux - 1.0) # 1.0 + (1.0 - cos_th) * (-uz * uz - uy * uy)
    update_matse3[0][1] = -sin_th * uz + (1.0 - cos_th) * uy * ux
    update_matse3[0][2] =  sin_th * uy + (1.0 - cos_th) * uz * ux

    update_matse3[1][0] =  sin_th * uz + (1.0 - cos_th) * uy * ux
    update_matse3[1][1] = 1.0 + (1.0 - cos_th) * (uy * uy - 1.0) # 1.0 + (1.0 - cos_th) * (-uz * uz - ux * ux)
    update_matse3[1][2] = -sin_th * ux + (1.0 - cos_th) * uz * uy

    update_matse3[2][0] = -sin_th * uy + (1.0 - cos_th) * uz * ux
    update_matse3[2][1] =  sin_th * ux + (1.0 - cos_th) * uz * uy
    update_matse3[2][2] = 1.0 + (1.0 - cos_th) * (uz * uz - 1.0) # 1.0 + (1.0 - cos_th) * (-uy * uy - ux * ux)

    #matse3[0][3] = (1.0 + (1.0 - sin_th_inv) * (-uz * uz - uy * uy)) * vx + (-cos_th_inv * uz + (1.0 - sin_th_inv) * uy * ux) * vy + (cos_th_inv * uy + (1.0 - sin_th_inv) * uz * ux) * vz
    #matse3[1][3] = (cos_th_inv * uz + (1.0 - sin_th_inv) * uy * ux) * vx + (1.0 + (1.0 - sin_th_inv) * (-uz * uz - ux * ux)) * vy + (-cos_th_inv * ux + (1.0 - sin_th_inv) * uz * uy) * vz
    #matse3[2][3] = (-cos_th_inv * uy + (1.0 - sin_th_inv) * uz * ux) * vx + (cos_th_inv * ux + (1.0 - sin_th_inv) * uz * uy) * vy + (1.0 + (1.0 - sin_th_inv) * (-uy * uy - ux * ux)) * vz

    # Translation vector
    update_matse3[0][3] = (1.0 + (1.0 - sin_th_inv) * (ux * ux - 1.0)) * vx + (-cos_th_inv * uz + (1.0 - sin_th_inv) * uy * ux) * vy + (cos_th_inv * uy + (1.0 - sin_th_inv) * uz * ux) * vz
    update_matse3[1][3] = (cos_th_inv * uz + (1.0 - sin_th_inv) * uy * ux) * vx + (1.0 + (1.0 - sin_th_inv) * (uy * uy - 1.0)) * vy + (-cos_th_inv * ux + (1.0 - sin_th_inv) * uz * uy) * vz
    update_matse3[2][3] = (-cos_th_inv * uy + (1.0 - sin_th_inv) * uz * ux) * vx + (cos_th_inv * ux + (1.0 - sin_th_inv) * uz * uy) * vy + (1.0 + (1.0 - sin_th_inv) * (uz * uz - 1.0)) * vz

    # Fourth row
    update_matse3[3][0] = 0.0
    update_matse3[3][1] = 0.0
    update_matse3[3][2] = 0.0
    update_matse3[3][3] = 1.0

    return update_matse3

def rox_algse3_set_velocity( vec_data ):
    """
    '''
    Description: 
    Args:: vec_data: 6x1 vector
    Returns:: 
    '''        
    """
    alg_data = torch.zeros((4,4), dtype=torch.float64, device=vec_data.device)
    alg_data[0][0] = 0
    alg_data[0][1] = -vec_data[5][0]
    alg_data[0][2] =  vec_data[4][0]
    alg_data[0][3] =  vec_data[0][0]

    alg_data[1][0] =  vec_data[5][0]
    alg_data[1][1] = 0
    alg_data[1][2] = -vec_data[3][0]
    alg_data[1][3] =  vec_data[1][0]

    alg_data[2][0] = -vec_data[4][0]
    alg_data[2][1] =  vec_data[3][0]
    alg_data[2][2] = 0
    alg_data[2][3] =  vec_data[2][0]

    alg_data[3][0] = 0
    alg_data[3][1] = 0
    alg_data[3][2] = 0
    alg_data[3][3] = 0
    return alg_data
