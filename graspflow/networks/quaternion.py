
#!/usr/bin/env python3
# Author: Tasbolat Taunyazov

# Notes:
# 1. All quaternions are given as x,y,z,w form (unless specified in specific function)
# 2. Conversion to and from euler angles is done using intrinsic representation. It
# corresponds to capital XYZ in scipy rotation package (BT angles in wikipedia, not Proper Euler angles)
# 3. All operations are backpropogatable (albeight not the optimal one)


import torch
import numpy as np


def quaternion_add(q, q_grad, eta, noise):
    '''
    move to eta scaled quaternion impelemented only for GraspFlow.
    eta - learning rate
    noise - noise_factor
    '''
    
    # find the norm the q_grad
    q_grad_norm = q_grad.norm(p=2, dim=1, keepdim=True) # [B,1]

    # Check if norm is not zero, if then modify it a bit
    zero_mask = (q_grad_norm == 0).nonzero()
    q_grad[zero_mask, 3] = 1.0 # no rotations with zero gradients

    # normalize the q_grad
    q_grad = quat_normalize(q_grad)

    # check sign of the movement to deal with 2-to-1 map of quaternions
    q_dir = torch.sign((q_grad*q.data).sum(-1, keepdims=True))
    q_dir[q_dir == 0] = 1 # [B,1]

    # find where it's heading
    q_hat = quaternion_mult(q_grad, q.data)     

    # scale the eta according to norm
    eta *= q_grad_norm

    # correct direction
    q_hat_dir = torch.sign((q_hat*q.data).sum(-1, keepdims=True))
    q_hat_dir[q_hat_dir == 0] = 1 # [B,1]
    q_dir = q_dir*q_hat_dir

    # make move
    _q = slerp([q.data, q_dir*q_hat], [eta])[0]

    ###### ADD NOISE
    q_noise = sample_norm(q.shape[0], return_euler=False)
    q_hat_noise = quaternion_mult(q_noise, _q.data)
    noise = torch.mean(torch.sqrt(2*eta)) * noise # ad-hoc: implementing batch wise slerp is expensive
    #noise = np.sqrt(eta) * noise
    _q = slerp([_q.data, q_hat_noise], [noise])[0]        

    return _q, q_grad


def quat_dot(q1, q2):
    '''
    dot product between q1 and q2
    '''
    q_dot = (q1*q2).sum(-1)
    return q_dot

def slerp(quats, times):
    '''
    Slerp for list of two quaternions quats=[q1, q2]. Time range is [0,1]. Can be backpropogated

    Input:
        - quats list: list of quaternions containing [B,4] quats
        - times: list of times to calculate interpolated quaternions. Condition: 0 <= times <= 1
    Return:
        - res list: list of interpolated quaternions at times.

    Note:
    https://github.com/scipy/scipy/blob/606e84b0e4370c885eec4b7acca5ec00797b7d98/scipy/spatial/transform/_rotation.pyx#L2536
    '''

    assert len(quats) == 2

    times = np.atleast_1d(times)
    if (np.any(times) < 0) or (np.any(times) > 1):
        raise ValueError(f'Wrong times argument, must be in range of [0,1], got {times}')
    
    q1 = quats[0].clone()
    q2 = quats[1].clone()

    delta_q = quaternion_mult(quat_inv(q1), q2)
    delta_rotvec = quat2rotvec(delta_q)

    times = torch.FloatTensor(times).to(q1.device)

    res = []
    for t in times:
        res_q = rotvec2quat(delta_rotvec*t).transpose(1,0)
        res.append(quaternion_mult(q1, res_q))

    return res


def sample_norm(num_samples=1, return_euler=True, order='XYZ', dtype = torch.float, device='cpu'):
    """
    Samples quaternion in normal distribution.

    The code is taken from:
    https://github.com/ompl/ompl/blob/9c3a20faaddfcd7f58fce235495f043ebee3e735/src/ompl/base/spaces/src/SO3StateSpace.cpp#L119
    """
    
    std = 2.0/torch.sqrt(torch.Tensor([3])).type(dtype=dtype).to(device)
    zero_mean = torch.zeros([num_samples, 3], dtype=dtype).to(device)
    xyz = torch.normal(mean=zero_mean, std=std)
    theta = torch.sqrt(torch.sum(xyz**2, dim=1))

    q_w = torch.cos(theta/2)
    q_x = xyz[:,0]*torch.sin(theta/2)/theta
    q_y = xyz[:,1]*torch.sin(theta/2)/theta
    q_z = xyz[:,2]*torch.sin(theta/2)/theta

    q = torch.stack([q_x, q_y, q_z, q_w], dim=1)
    
    if return_euler:
        e = quat2euler(q, order=order)
        return e
    
    return q

def rotvec2quat(rotvec, verbose=False, epsilon=1e-3):
    '''
    Converts quaternions to rotvec
    Input:
        - rotvec [B, 3]: rotation vectors
        - verbose boolean: verbose for debug, defaul False
        - epslion float: assures numerical stability, defauly 1e-3
    Return:
        - quaternions [B, 3]: normalized quaternions

    Note:
    https://github.com/scipy/scipy/blob/606e84b0e4370c885eec4b7acca5ec00797b7d98/scipy/spatial/transform/_rotation.pyx#L880
    
    Backward not tested!

    '''

    assert rotvec.shape[-1] == 3
    assert rotvec.dim() == 2

    rv = rotvec.clone() # may breaks backprop

    angles = torch.linalg.norm(rv, ord=2, dim=1)
    small_angles_mask = torch.argwhere(angles <= epsilon)
    scale = torch.sin(angles / 2.0) / angles

    if len(small_angles_mask) != 0: # check numerical stability
        small_angles = angles[small_angles_mask]
        small_angles2 = small_angles**2
        new_scale = 0.5 - small_angles/48 + small_angles2**2/3840
        scale[small_angles_mask] = new_scale

    q = torch.stack([scale * rv[:,0], scale * rv[:,1], scale * rv[:,2], torch.cos(angles/2)])

    return q


def quat2rotvec(quat, verbose=False, epsilon=1e-3):
    '''
    Converts quaternions to rotvec
    Input:
        - quat [B, 4]: quaternions (can be unnormalized)
        - verbose boolean: verbose for debug, defaul False
        - epslion float: assures numerical stability, defauly 1e-3
    Return:
        - rotvec [B, 3]: rotation vectors in radians

    Note:
    https://github.com/scipy/scipy/blob/606e84b0e4370c885eec4b7acca5ec00797b7d98/scipy/spatial/transform/_rotation.pyx#L1384

    Backward not tested!
    '''
    
    assert quat.shape[-1] == 4
    assert quat.dim() == 2

    q = quat.clone() # may breaks backprop

    q = quat_normalize(q, verbose=verbose)

    
    # w > 0 to ensure 0 <= angle <= pi
    neg_w_masks = torch.argwhere(q[:, 3] < 0)
    if len(neg_w_masks) != 0:
        q[neg_w_masks] *= -1.0

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]


    q_norm3 = torch.linalg.norm(q[:,:-1], ord=2, dim=1)
    angles = 2 * torch.atan2(q_norm3, q3)
    small_angles_mask = torch.argwhere(angles <= epsilon)
    scale = angles / torch.sin(angles / 2.0)

    if len(small_angles_mask) != 0: # check numerical stability
        small_angles = angles[small_angles_mask]
        small_angles2 = small_angles**2
        new_scale = 2 + small_angles/12 + 7*small_angles2**2/2880
        scale[small_angles_mask] = new_scale



    rotvec = torch.stack([scale * q0, scale * q1, scale * q2]).transpose(1,0)

    return rotvec



def quat2euler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    # q0 = q[:, 0]
    # q1 = q[:, 1]
    # q2 = q[:, 2]
    # q3 = q[:, 3]

    q0 = q[:, 3]
    q1 = q[:, 0]
    q2 = q[:, 1]
    q3 = q[:, 2]

    if order == 'XYZ':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'YZX':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'ZXY':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'XZY':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'YXZ':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'ZYX':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise ValueError("Invalid order " + order)

    return torch.stack((x, y, z), dim=1).view(original_shape)



def quaternion_distance(quat1, quat2):
    '''
    Input Bx4
    Output B
    '''
    return 2*torch.arccos( torch.abs( torch.sum(quat1 * quat2, dim=1) ) )


def quat_normalize(quat, verbose=False):
    '''
    Normalizes quaternions
    Input:
        - quat [N,4]: unnormalized quaternions
    Return:
        - quat [N,4]: normalized quaternions 
    '''
    if quat.dim() > 2:
        raise ValueError(
            "Quaternion shape expected to be [N,4], but got '{}'.".format(quat.shape))
        
    if quat.dim() < 2:
        quat = quat.unsqueeze(dim=0)

    if not allclose(quat.norm(p=2, dim=1), 1.):
        if verbose:
            print('Quaternions are not normalized, normalizing ...')        
        quat = quat/quat.norm(p=2, dim=1, keepdim=True)

    return quat


def quat_to_rotmat(quat, ordering='xyzw'):
    """Form a rotation matrix from a unit length quaternion.
        Valid orderings are 'xyzw' and 'wxyz'.
    """
    if quat.dim() < 2:
        quat = quat.unsqueeze(dim=0)

    if not allclose(quat.norm(p=2, dim=1), 1.):
        print("Warning: Some quaternions not unit length ... normalizing.")
        quat = quat/quat.norm(p=2, dim=1, keepdim=True)

    if ordering == 'xyzw':
        qx = quat[:, 0]
        qy = quat[:, 1]
        qz = quat[:, 2]
        qw = quat[:, 3]
    elif ordering == 'wxyz':
        qw = quat[:, 0]
        qx = quat[:, 1]
        qy = quat[:, 2]
        qz = quat[:, 3]
    else:
        raise ValueError(
            "Valid orderings are 'xyzw' and 'wxyz'. Got '{}'.".format(ordering))

    # Form the matrix
    mat = quat.new_empty(quat.shape[0], 3, 3)

    qx2 = qx * qx
    qy2 = qy * qy
    qz2 = qz * qz

    mat[:, 0, 0] = 1. - 2. * (qy2 + qz2)
    mat[:, 0, 1] = 2. * (qx * qy - qw * qz)
    mat[:, 0, 2] = 2. * (qw * qy + qx * qz)

    mat[:, 1, 0] = 2. * (qw * qz + qx * qy)
    mat[:, 1, 1] = 1. - 2. * (qx2 + qz2)
    mat[:, 1, 2] = 2. * (qy * qz - qw * qx)

    mat[:, 2, 0] = 2. * (qx * qz - qw * qy)
    mat[:, 2, 1] = 2. * (qw * qx + qy * qz)
    mat[:, 2, 2] = 1. - 2. * (qx2 + qy2)

    return mat.squeeze_()


def quaternion_conj(q):
    """
      Conjugate of quaternion q (x,y,z,w) -> (-x,-y,-z,w).
    """
    q_conj = q.clone()
    q_conj[:, :, :3] *= -1
    return q_conj

def quaternion_mult(q, r):
    """
    Multiply quaternion(s) q (x,y,z,w) with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    # Compute outer product
    terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    
    x = + terms[:, 3, 0] + terms[:, 2, 1] - terms[:, 1, 2] + terms[:, 0, 3]
    y = - terms[:, 2, 0] + terms[:, 3, 1] + terms[:, 0, 2] + terms[:, 1, 3]
    z = + terms[:, 1, 0] - terms[:, 0, 1] + terms[:, 3, 2] + terms[:, 2, 3]
    w = - terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] + terms[:, 3, 3] 
   

    return torch.stack((x, y, z, w), dim=1).view(original_shape)

def rot_p_by_quaterion(p, q):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = p.shape
    q_shape = q.shape

    assert (len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (shape[-1] == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[-1] == 4), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = quaternion_conj(q)
    r = torch.cat([ p,
        torch.zeros(
            (shape[0], shape[1], 1), dtype=p.dtype).to(p.device)],
                  dim=-1)
    result = quaternion_mult(quaternion_mult(q, r), q_conj)
    return result[:,:,:3] 

def quat_norm_diff(q_a, q_b):
    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a-q_b).norm(dim=1), (q_a+q_b).norm(dim=1)).squeeze()

def quat_inv(q):
    #Note, 'empty_like' is necessary to prevent in-place modification (which is not auto-diff'able)
    if q.dim() < 2:
        q = q.unsqueeze()
    q_inv = torch.empty_like(q)
    q_inv[:, :3] = -1*q[:, :3]
    q_inv[:, 3] = q[:, 3]
    return q_inv # .squeeze() # check TODO


def allclose(mat1, mat2, tol=1e-6):
    """Check if all elements of two tensors are close within some tolerance.
    Either tensor can be replaced by a scalar.
    """
    return isclose(mat1, mat2, tol).all()


def isclose(mat1, mat2, tol=1e-6):
    """Check element-wise if two tensors are close within some tolerance.
    Either tensor can be replaced by a scalar.
    """
    return (mat1 - mat2).abs_().lt(tol)


def axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    """
    Return the rotation matrices for one of the rotations about an axis
    of which Euler angles describe, for each value of the angle given.
    Args:
        axis: Axis label "X" or "Y or "Z".
        angle: any shape tensor of Euler angles in radians
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """

    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)

    if axis == "X":
        R_flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        R_flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        R_flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError("letter must be either X, Y or Z.")

    return torch.stack(R_flat, -1).reshape(angle.shape + (3, 3))


def euler2matrix(euler_angles: torch.Tensor, order: str) -> torch.Tensor:
    """
    Convert rotations given as Euler angles in radians to rotation matrices.
    Args:
        euler_angles: Euler angles in radians as tensor of shape (..., 3).
        convention: Convention string of three uppercase letters from
            {"X", "Y", and "Z"}.
    Returns:
        Rotation matrices as tensor of shape (..., 3, 3).
    """
    if euler_angles.dim() == 0 or euler_angles.shape[-1] != 3:
        raise ValueError("Invalid input euler angles.")
    if len(order) != 3:
        raise ValueError("Convention must have 3 letters.")
    if order[1] in (order[0], order[2]):
        raise ValueError(f"Invalid convention {order}.")
    for letter in order:
        if letter not in ("X", "Y", "Z"):
            raise ValueError(f"Invalid letter {letter} in convention string.")
    matrices = [
        axis_angle_rotation(c, e)
        for c, e in zip(order, torch.unbind(euler_angles, -1))
    ]
    # return functools.reduce(torch.matmul, matrices)
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])   



def euler2quat(e, order='XYZ'):
    '''
    Converts Euler to Quaternion. Not for now only works for XYZ. Idk why, but it's enough for us
    Input:
    - e [B,3]: euler angles
    - order: ---
    Output:
    - q [B,4]: quaternion
    '''

    x = e[:, 0]
    y = e[:, 1]
    z = e[:, 2]

    rx = torch.stack((torch.sin(x/2), torch.zeros_like(x), torch.zeros_like(x), torch.cos(x/2)), dim=1)
    ry = torch.stack((torch.zeros_like(y), torch.sin(y/2), torch.zeros_like(y), torch.cos(y/2)), dim=1)
    rz = torch.stack((torch.zeros_like(z), torch.zeros_like(z), torch.sin(z/2), torch.cos(z/2)), dim=1)

    result = None
    for coord in order:
        if coord == 'X':
            r = rx
        elif coord == 'Y':
            r = ry
        elif coord == 'Z':
            r = rz
        else:
            raise
        if result is None:
            result = r
        else:
            result = quaternion_mult(result, r)

    if order in ['XYZ', 'YZX', 'ZXY']:
        result *= -1

    return result
