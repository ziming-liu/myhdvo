import numpy as np
import math
import torch 



_EPS1 = torch.finfo(float).eps * 4.0


def euler_to_quaternion(roll, pitch, yaw):
    """Converts Euler angles representing spatial rotations to equivalent representations by unit quaterians. This
    function is vectorized and operates on arrays of angles.

    Args:
        roll (torch.ndarray): Nx1 array of roll angle in degrees for N observations pitch (torch.ndarray): Nx1 array of pitch
        angle in degrees for N observations yaw (torch.ndarray): Nx1 array of yaw angle in degrees for N observations

    Returns:
        torch.ndarray: Nx4 array, with N rows of observations and the four columns being x, y, z, w quaterion components.
    """
    roll = torch.deg2rad(roll)
    pitch = torch.deg2rad(pitch)
    yaw = torch.deg2rad(yaw)
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    w = cy * cp * cr + sy * sp * sr
    x = cy * cp * sr - sy * sp * cr
    y = sy * cp * sr + cy * sp * cr
    z = sy * cp * cr - cy * sp * sr
    return torch.cat([x, y, z, w],dim=1)


def transformation_matrix(t, q):
    """Generate 4x4 homogeneous transformation matrices from arrays of 3D point translations and unit quaternion
    representations of rotations. This function is vectorized and operates on arrays of transformation parameters.

    Args:
        t (torch.ndarray): Nx3 array of translations (tx, ty, tz)
        q (torch.ndarray): Nx4 array of unit quaternions (qx, qy, qz, qw)

    Returns:
        torch.ndarray: Nx4x4 array of 4x4 transformation matrices
    """
    # Allocate transformation matrices with only translation
    arr_transforms = torch.eye(4).unsqueeze(0).repeat(t.shape[0],1,1).cuda()
    arr_transforms[:, :3, 3] = t
    nq = torch.square(torch.linalg.norm(q, dim=1, keepdim=False))
    mask = nq >= _EPS1  # mask for rotations of magnitude greater than epsilon

    # For transformations with non-zero rotation, calculate rotation matrix
    q = torch.sqrt(2.0 / nq)[:, None] * q
    q = q[:, :, None] * q[:, None, :]  # outer product
    arr_transforms[mask, 0, 0] = 1.0 - q[mask, 1, 1] - q[mask, 2, 2]
    arr_transforms[mask, 0, 1] = q[mask, 0, 1] - q[mask, 2, 3]
    arr_transforms[mask, 0, 2] = q[mask, 0, 2] + q[mask, 1, 3]
    arr_transforms[mask, 1, 0] = q[mask, 0, 1] + q[mask, 2, 3]
    arr_transforms[mask, 1, 1] = 1.0 - q[mask, 0, 0] - q[mask, 2, 2]
    arr_transforms[mask, 1, 2] = q[mask, 1, 2] - q[mask, 0, 3]
    arr_transforms[mask, 2, 0] = q[mask, 0, 2] - q[mask, 1, 3]
    arr_transforms[mask, 2, 1] = q[mask, 1, 2] + q[mask, 0, 3]
    arr_transforms[mask, 2, 2] = 1.0 - q[mask, 0, 0] - q[mask, 1, 1]
    return arr_transforms

def compute_distance(arr_transforms):
    """
    Compute the distance of the translational components of an array of N 4x4 homogeneous matrices.

    Args:
        arr_transforms (torch.ndarray): Nx4x4 array of N 4x4 transformation matrices

    Returns:
        torch.ndarray: 1D array of length N with distance values
    """
    return torch.linalg.norm(arr_transforms[:, :3, 3], dim=1)


def compute_angle(arr_transforms):
    """
    Compute the rotation angle from an array of N 4x4 homogeneous matrices.

    Args:
        arr_transforms (torch.ndarray): Nx4x4 array of N 4x4 transformation matrices

    Returns:
        torch.ndarray: 1D array of length N with angle values
    """
    # an invitation to 3-d vision, p 27
    return torch.arccos(
        torch.minimum(
            torch.ones(1).to(arr_transforms.device),
            torch.maximum(
                -torch.ones(1).to(arr_transforms.device),
                (torch.vmap(torch.trace)(arr_transforms[:, :3, :3]) - 1) / 2,
            ),
        )
    )



# epsilon for testing whether a number is close to zero
_EPS = np.finfo(float).eps * 4.0

def isRotationMatrix(R) :
	Rt = np.transpose(R)
	shouldBeIdentity = np.dot(Rt, R)
	I = np.identity(3, dtype = R.dtype)
	n = np.linalg.norm(I - shouldBeIdentity)
	return n < 1e-6

def R_to_angle(Rt):
# Ground truth pose is present as [R | t] 
# R: Rotation Matrix, t: translation vector
# transform matrix to angles
	Rt = np.reshape(np.array(Rt), (3,4))
	t = Rt[:,-1]
	R = Rt[:,:3]

	assert(isRotationMatrix(R))
	
	x, y, z = euler_from_matrix(R)
	
	theta = [x, y, z]
	pose_15 = np.concatenate((theta, t, R.flatten()))
	assert(pose_15.shape == (15,))
	return pose_15

def eulerAnglesToRotationMatrix(theta) :
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         np.cos(theta[0]), -np.sin(theta[0]) ],
                    [0,         np.sin(theta[0]), np.cos(theta[0])  ]
                    ])
    R_y = np.array([[np.cos(theta[1]),    0,      np.sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-np.sin(theta[1]),   0,      np.cos(theta[1])  ]
                    ])
    R_z = np.array([[np.cos(theta[2]),    -np.sin(theta[2]),    0],
                    [np.sin(theta[2]),    np.cos(theta[2]),     0],
                    [0,                     0,                      1]
                    ])
    R = np.dot(R_z, np.dot( R_y, R_x ))
    return R

def torcheulerAnglesToRotationMatrix(theta) :
    bs = theta.shape[0]
    R_x, R_y, R_z = torch.eye(3).unsqueeze(0).repeat(bs,1,1).cuda(), torch.eye(3).unsqueeze(0).repeat(bs,1,1).cuda(), torch.eye(3).unsqueeze(0).repeat(bs,1,1).cuda()
    R_x[:,1,1], R_x[:,1,2], R_x[:,2,1], R_x[:,2,2] = torch.cos(theta[:,0]), -torch.sin(theta[:,0]),  torch.sin(theta[:,0]), torch.cos(theta[:,0]) 
    R_y[:,0,0], R_y[:,0,2], R_y[:,2,0], R_y[:,2,2] = torch.cos(theta[:,1]),   torch.sin(theta[:,1]), -torch.sin(theta[:,1]),  torch.cos(theta[:,1])
    R_z[:,0,0], R_z[:,0,1], R_z[:,1,0], R_z[:,1,1] = torch.cos(theta[:,2]),    -torch.sin(theta[:,2]), torch.sin(theta[:,2]),    torch.cos(theta[:,2]), 
 
    R = torch.bmm(R_z, torch.bmm( R_y, R_x ))
    return R

def euler_from_matrix(matrix):
    
	# y-x-z Tait–Bryan angles intrincic
	# the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    
    i = 2
    j = 0
    k = 1
    repetition = 0
    frame = 1
    parity = 0
	

    M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j]*M[i, j] + M[i, k]*M[i, k])
        if sy > _EPS:
            ax = math.atan2( M[i, j],  M[i, k])
            ay = math.atan2( sy,       M[i, i])
            az = math.atan2( M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2( sy,       M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i]*M[i, i] + M[j, i]*M[j, i])
        if cy > _EPS:
            ax = math.atan2( M[k, j],  M[k, k])
            ay = math.atan2(-M[k, i],  cy)
            az = math.atan2( M[j, i],  M[i, i])
        else:
            ax = math.atan2(-M[j, k],  M[j, j])
            ay = math.atan2(-M[k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def torcheuler_from_matrix(matrix):
    
	# y-x-z Tait–Bryan angles intrincic
	# the method code is taken from https://github.com/awesomebytes/delta_robot/blob/master/src/transformations.py
    
    i = 2
    j = 0
    k = 1
    repetition = 0
    frame = 1
    parity = 0
	

   # M = np.array(matrix, dtype=np.float64, copy=False)[:3, :3]
    M = matrix[:,:3,:3]
    if repetition:
        sy = torch.sqrt(M[:,i, j]*M[:,i, j] + M[:,i, k]*M[:,i, k])
        if sy > _EPS:
            ax = torch.atan2( M[:,i, j],  M[:,i, k])
            ay = torch.atan2( sy,       M[:,i, i])
            az = torch.atan2( M[:,j, i], -M[:,k, i])
        else:
            ax = torch.atan2(-M[:,j, k],  M[:,j, j])
            ay = torch.atan2( sy,       M[:,i, i])
            az = 0.0
    else:
        cy = torch.sqrt(M[:,i, i]*M[:,i, i] + M[:,j, i]*M[:,j, i])
        if cy > _EPS:
            ax = torch.atan2( M[:,k, j],  M[:,k, k])
            ay = torch.atan2(-M[:,k, i],  cy)
            az = torch.atan2( M[:,j, i],  M[:,i, i])
        else:
            ax = torch.atan2(-M[:,j, k],  M[:,j, j])
            ay = torch.atan2(-M[:,k, i],  cy)
            az = 0.0

    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az

def normalize_angle_delta(angle):
    if(angle > np.pi):
        angle = angle - 2 * np.pi
    elif(angle < -np.pi):
        angle = 2 * np.pi + angle
    return angle




def torch_mat2euler(M, cy_thresh=None, seq='xyz'):
    '''
    Taken From: http://afni.nimh.nih.gov/pub/dist/src/pkundu/meica.libs/nibabel/eulerangles.py
    Discover Euler angle vector from 3x3 matrix
    Uses the conventions above.
    Parameters
    ----------
    M : array-like, shape (3,3)
    cy_thresh : None or scalar, optional
     threshold below which to give up on straightforward arctan for
     estimating x rotation.  If None (default), estimate from
     precision of input.
    Returns
    -------
    z : scalar
    y : scalar
    x : scalar
     Rotations in radians around z, y, x axes, respectively
    Notes
    -----
    If there was no numerical error, the routine could be derived using
    Sympy expression for z then y then x rotation matrix, which is::
    [                       cos(y)*cos(z),                       -cos(y)*sin(z),         sin(y)],
    [cos(x)*sin(z) + cos(z)*sin(x)*sin(y), cos(x)*cos(z) - sin(x)*sin(y)*sin(z), -cos(y)*sin(x)],
    [sin(x)*sin(z) - cos(x)*cos(z)*sin(y), cos(z)*sin(x) + cos(x)*sin(y)*sin(z),  cos(x)*cos(y)]
    with the obvious derivations for z, y, and x
     z = atan2(-r12, r11)
     y = asin(r13)
     x = atan2(-r23, r33)
    for x,y,z order
    y = asin(-r31)
    x = atan2(r32, r33)
    z = atan2(r21, r11)
    Problems arise when cos(y) is close to zero, because both of::
     z = atan2(cos(y)*sin(z), cos(y)*cos(z))
     x = atan2(cos(y)*sin(x), cos(x)*cos(y))
    will be close to atan2(0, 0), and highly unstable.
    The ``cy`` fix for numerical instability below is from: *Graphics
    Gems IV*, Paul Heckbert (editor), Academic Press, 1994, ISBN:
    0123361559.  Specifically it comes from EulerAngles.c by Ken
    Shoemake, and deals with the case where cos(y) is close to zero:
    See: http://www.graphicsgems.org/
    The code appears to be licensed (from the website) as "can be used
    without restrictions".
    '''
    #M = np.asarray(M)
    if cy_thresh is None:
        cy_thresh = torch.FloatTensor([np.finfo(float).eps * 4]).cuda()
        #try:
        #    cy_thresh = torch.FloatTensor(np.finfo(M.dtype).eps * 4).cuda()
        ##except ValueError:
        #    cy_thresh = _FLOAT_EPS_4
    r11, r12, r13, r21, r22, r23, r31, r32, r33 = M.reshape(-1)
    # cy: sqrt((cos(y)*cos(z))**2 + (cos(x)*cos(y))**2)
    cy = torch.sqrt(r33*r33 + r23*r23)
    if seq=='zyx':
        if cy > cy_thresh: # cos(y) not close to zero, standard form
            z = torch.atan2(-r12,  r11) # atan2(cos(y)*sin(z), cos(y)*cos(z))
            y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
            x = torch.atan2(-r23, r33) # atan2(cos(y)*sin(x), cos(x)*cos(y))
        else: # cos(y) (close to) zero, so x -> 0.0 (see above)
            # so r21 -> sin(z), r22 -> cos(z) and
            z = torch.atan2(r21,  r22)
            y = torch.atan2(r13,  cy) # atan2(sin(y), cy)
            x = 0.0
    elif seq=='xyz':
        if cy > cy_thresh:
            y = torch.atan2(-r31, cy)
            x = torch.atan2(r32, r33)
            z = torch.atan2(r21, r11)
        else:
            z = 0.0
            if r31 < 0:
                y = (torch.FloatTensor([np.pi])).cuda()/2
                x = torch.atan2(r12, r13)
            else:
                y = -(torch.FloatTensor([np.pi])).cuda()/2
                x = 0.0 # fix the bug: torch.FloatTensor([x, y, z]).cuda()UnboundLocalError: local variable 'x' referenced before assignment
    else:
        raise Exception('Sequence not recognized')
    if seq=="zyx":
        return torch.FloatTensor([z, y, x]).cuda()
    elif seq=="xyz":
        return torch.FloatTensor([x, y, z]).cuda()
    else:
        raise ValueError

import functools
def torch_euler2mat(theta, isRadian=True, seq="xyz", cuda=True):
    ''' Return matrix for rotations around z, y and x axes
    Uses the z, then y, then x convention above
    Parameters
    ----------
    z : scalar
         Rotation angle in radians around z-axis (performed first)
    y : scalar
         Rotation angle in radians around y-axis
    x : scalar
         Rotation angle in radians around x-axis (performed last)
    Returns
    -------
    M : array shape (3,3)
         Rotation matrix giving same rotation as for given angles
    Examples
    --------
    >>> zrot = 1.3 # radians
    >>> yrot = -0.1
    >>> xrot = 0.2
    >>> M = euler2mat(zrot, yrot, xrot)
    >>> M.shape == (3, 3)
    True
    The output rotation matrix is equal to the composition of the
    individual rotations
    >>> M1 = euler2mat(zrot)
    >>> M2 = euler2mat(0, yrot)
    >>> M3 = euler2mat(0, 0, xrot)
    >>> composed_M = np.dot(M3, np.dot(M2, M1))
    >>> np.allclose(M, composed_M)
    True
    You can specify rotations by named arguments
    >>> np.all(M3 == euler2mat(x=xrot))
    True
    When applying M to a vector, the vector should column vector to the
    right of M.  If the right hand side is a 2D array rather than a
    vector, then each column of the 2D array represents a vector.
    >>> vec = np.array([1, 0, 0]).reshape((3,1))
    >>> v2 = np.dot(M, vec)
    >>> vecs = np.array([[1, 0, 0],[0, 1, 0]]).T # giving 3x2 array
    >>> vecs2 = np.dot(M, vecs)
    Rotations are counter-clockwise.
    >>> zred = np.dot(euler2mat(z=np.pi/2), np.eye(3))
    >>> np.allclose(zred, [[0, -1, 0],[1, 0, 0], [0, 0, 1]])
    True
    >>> yred = np.dot(euler2mat(y=np.pi/2), np.eye(3))
    >>> np.allclose(yred, [[0, 0, 1],[0, 1, 0], [-1, 0, 0]])
    True
    >>> xred = np.dot(euler2mat(x=np.pi/2), np.eye(3))
    >>> np.allclose(xred, [[1, 0, 0],[0, 0, -1], [0, 1, 0]])
    True
    Notes
    -----
    The direction of rotation is given by the right-hand rule (orient
    the thumb of the right hand along the axis around which the rotation
    occurs, with the end of the thumb at the positive end of the axis;
    curl your fingers; the direction your fingers curl is the direction
    of rotation).  Therefore, the rotations are counterclockwise if
    looking along the axis of rotation from positive to negative.
    '''
    x, y, z = theta[0], theta[1], theta[2]
    if not isRadian:
        z = ((torch.FloatTensor([np.pi])).cuda()/180.) * z
        y = ((torch.FloatTensor([np.pi])).cuda()/180.) * y
        x = ((torch.FloatTensor([np.pi])).cuda()/180.) * x
    #assert z>=(-(torch.FloatTensor([np.pi])).cuda()) and z < (torch.FloatTensor([np.pi])).cuda(), 'Inapprorpriate z: %f' % z
    #assert y>=(-(torch.FloatTensor([np.pi])).cuda()) and y < (torch.FloatTensor([np.pi])).cuda(), 'Inapprorpriate y: %f' % y
    #assert x>=(-(torch.FloatTensor([np.pi])).cuda()) and x < (torch.FloatTensor([np.pi])).cuda(), 'Inapprorpriate x: %f' % x    
    z = torch.clamp(z, -(torch.FloatTensor([np.pi])).cuda(), (torch.FloatTensor([np.pi])).cuda())
    y = torch.clamp(y, -(torch.FloatTensor([np.pi])).cuda(), (torch.FloatTensor([np.pi])).cuda())
    x = torch.clamp(x, -(torch.FloatTensor([np.pi])).cuda(), (torch.FloatTensor([np.pi])).cuda())

    Ms = []
    if z:
        cosz = torch.cos(z)
        sinz = torch.sin(z)
        Ms.append(torch.FloatTensor(
                        [[cosz, -sinz, 0],
                            [sinz, cosz, 0],
                            [0, 0, 1]]))
    if y:
        cosy = torch.cos(y)
        siny = torch.sin(y)
        Ms.append(torch.FloatTensor(
                        [[cosy, 0, siny],
                            [0, 1, 0],
                            [-siny, 0, cosy]]))
    if x:
        cosx = torch.cos(x)
        sinx = torch.sin(x)
        Ms.append(torch.FloatTensor(
                        [[1, 0, 0],
                            [0, cosx, -sinx],
                            [0, sinx, cosx]]))
        
    if Ms:  
        if seq=="zyx":
            return functools.reduce(torch.mm, Ms[::-1]).cuda()
        if seq=="xyz":
            return functools.reduce(torch.mm, Ms).cuda()
    return torch.eye(3).cuda()


if __name__ == '__main__':
    import torch 
    import numpy as np

    #angle = torch.FloatTensor([0.01, 0.002, 0.003]).cuda()
    angle = torch.FloatTensor([0.0, 45., 15.]).cuda()
    print(angle)
    R = torch_euler2mat(angle, isRadian=False, seq="xyz", cuda=True)
    print(R)
    
    
    back_angle = torch_mat2euler(R, )

    print(back_angle)


    print("func2 ")
    q = euler_to_quaternion(angle[0].reshape(1,1), angle[1].reshape(1,1), angle[2].reshape(1,1))    
    t = torch.FloatTensor([0, 0, 0]).cuda().unsqueeze(0)
    
    Rt = transformation_matrix(t,q)
    print(Rt)