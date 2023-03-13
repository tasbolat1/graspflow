
#!/usr/bin/env python3
# Tasbolat Taunyazov

import torch


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
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
    return q_inv.squeeze()


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