#!/usr/bin/env python3
# Tasbolat Taunyazov

import torch
import numpy as np
from pathlib import Path
import networks.quaternion as quaternion


def save_model(model, path, epoch=None, reason=None):
    
    if reason is None:
        print(f'saving model at {epoch} epoch ...')
        save_path = Path(path)/f'{epoch}.pt'
    else:
        print(f'saving model at {epoch} epoch at reason {reason}...')
        save_path = Path(path)/f'{reason}.pt'

    if isinstance(model, torch.nn.DataParallel):
        torch.save(model.module.state_dict(), save_path)
    else:
        torch.save(model.state_dict(), save_path)
        
def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    decay = []
    no_decay = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if len(param.shape) == 1 or name in skip_list:
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {'params': no_decay, 'weight_decay': 0.},
        {'params': decay, 'weight_decay': weight_decay}]


def eye(batch_size, size):
    I = torch.eye(size)
    I = I.reshape((1, size, size))
    I = I.repeat(batch_size, 1, 1)
    return I

def convert_Avec_to_A(A_vec):
    """ Convert BxM tensor to BxNxN symmetric matrices """
    """ M = N*(N+1)/2"""
    if A_vec.dim() < 2:
        A_vec = A_vec.unsqueeze(dim=0)
    
    if A_vec.shape[1] == 10:
        A_dim = 4
    elif A_vec.shape[1] == 55:
        A_dim = 10
    else:
        raise ValueError("Arbitrary A_vec not yet implemented")

    idx = torch.triu_indices(A_dim,A_dim)
    A = A_vec.new_zeros((A_vec.shape[0],A_dim,A_dim))   
    A[:, idx[0], idx[1]] = A_vec
    A[:, idx[1], idx[0]] = A_vec
    return A.squeeze()

def A_vec_to_quat(A_vec):
    '''
    Input:
        - A_vec [B,10]
    Output:
        - quat [B,4]
    '''
    A = convert_Avec_to_A(A_vec)
    _, eigvector = torch.linalg.eigh(A, "U")
    if eigvector.dim() < 3:
        eigvector = eigvector.unsqueeze(0)
    return eigvector[:,:,0]

def convert_A_to_Avec(A):
    """ Convert BxNXN symmetric matrices to BxM vectors encoding unique values"""
    if A.dim() < 3:
        A = A.unsqueeze(dim=0)
    idx = torch.triu_indices(A.shape[1], A.shape[1])
    A_vec = A[:, idx[0], idx[1]]
    return A_vec.squeeze()

def quat_to_A_vec(quat):
    '''
    Smoot A_vec representation of quaternions using Eq[12]
    Input:
        - quat [B,4]
    Output:
        - A_vec [B,10]
    '''
    assert quat.dim() == 2, print(f'quat shape must be [B,2], but given [{quat.dim()}]')
    A = eye(quat.shape[0], 4).to(quat.device) - torch.bmm(quat.unsqueeze(2), quat.unsqueeze(1))
    return convert_A_to_Avec(A)
    
def transform_gripper_pc_old(quat, trans, config_path = 'configs/panda.npy'):
    # q: (x,y,z, w)
    # t: (x,y,z)
    
    # # upload gripper_pc
    # control_points = np.load(config_path)[:, :3]
    # # control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
    # #                   control_points[1, :], control_points[-2, :],
    # #                   control_points[-1, :]]
    # mid_point = control_points[0, :]*0.5 + control_points[1, :]*0.5
    # control_points = [[0, 0, 0], mid_point, control_points[0, :],
    #                   control_points[1, :], control_points[-2, :],
    #                   control_points[-1, :]]
    # control_points = np.asarray(control_points, dtype=np.float32)
    # control_points = np.tile(np.expand_dims(control_points, 0),
    #                          [quat.shape[0], 1, 1])

    # gripper_pc = torch.tensor(control_points).to(quat.device)

    gripper_pc = get_control_point_tensor(quat.shape[0], config_path=config_path).to(quat.device)

    # prepare q and t 
    quat = quat.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])
    trans = trans.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])

    # rotate and add
    gripper_pc = quaternion.rot_p_by_quaterion(gripper_pc, quat)
    gripper_pc +=trans

    return gripper_pc

def transform_gripper_pc_custom(quat, trans, pc_points, repeat=False):
    # q: [B, 4]
    # t: [B, 3]
    # pc_points: [B2, 4]

    if repeat:
        gripper_pc = pc_points[:, :3].repeat(quat.shape[0], 1, 1) # [B, B2, 3]
    else:
        gripper_pc = pc_points

    # prepare q and t 
    quat = quat.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])
    trans = trans.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])

    # rotate and add
    gripper_pc = quaternion.rot_p_by_quaterion(gripper_pc, quat)
    gripper_pc +=trans

    return gripper_pc


def get_gripper_pc(use_torch=True, path='configs/gripper_pc.npy', full=False):
    """
      Outputs a tensor of shape (batch_size x 28 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    if full:
        path = 'configs/full_gripper_pc.npy'
        
    control_points = np.load(path)
    control_points = np.asarray(control_points, dtype=np.float32)

    if use_torch:
        return torch.FloatTensor(control_points)

    return control_points


def transform_gripper_pc(quat, trans, full=False):
    # q: (x,y,z, w)
    # t: (x,y,z)
    
    # upload gripper_pc
    gripper_pc = get_gripper_pc(full=full).to(quat.device)
    gripper_pc = gripper_pc.unsqueeze(0).to(quat.device)
    gripper_pc = gripper_pc.repeat([quat.shape[0], 1, 1])

    # prepare q and t 
    quat = quat.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])
    trans = trans.unsqueeze(1).repeat([1, gripper_pc.shape[1], 1])

    # rotate and add
    gripper_pc = quaternion.rot_p_by_quaterion(gripper_pc, quat)
    gripper_pc +=trans

    return gripper_pc


def merge_pc_and_gripper_pc(pc, gripper_pc):
    """
    Merges the object point cloud and gripper point cloud and
    adds a binary auxiliary feature that indicates whether each point
    belongs to the object or to the gripper.
    """
    pc_shape = pc.shape
    gripper_shape = gripper_pc.shape
    assert (len(pc_shape) == 3)
    assert (len(gripper_shape) == 3)
    assert (pc_shape[0] == gripper_shape[0])

    batch_size = pc_shape[0]

    l0_xyz = torch.cat((pc, gripper_pc), 1)
    labels = [
        torch.ones(pc.shape[1], 1, dtype=pc.dtype),
        torch.zeros(gripper_pc.shape[1], 1, dtype=pc.dtype)
    ]
    labels = torch.cat(labels, 0)
    labels.unsqueeze_(0)
    labels = labels.repeat(batch_size, 1, 1)

    l0_points = torch.cat([l0_xyz, labels.to(pc.device)],
                            -1).transpose(-1, 1)

    return l0_xyz, l0_points

def merge_pc_and_gripper_pc2(pc, gripper_pc):
    """
    Merges the object point cloud and gripper point cloud and
    adds a binary auxiliary feature that indicates whether each point
    belongs to the object or to the gripper.
    """
    pc_shape = pc.shape
    gripper_shape = gripper_pc.shape
    assert (len(pc_shape) == 3)
    assert (len(gripper_shape) == 3)
    assert (pc_shape[0] == gripper_shape[0])

    batch_size = pc_shape[0]

    l0_xyz = torch.cat((pc, gripper_pc), 1)
    labels = [
        torch.ones(pc.shape[1], 1, dtype=pc.dtype),
        torch.zeros(gripper_pc.shape[1], 1, dtype=pc.dtype)
    ]
    labels = torch.cat(labels, 0)
    labels.unsqueeze_(0)
    labels = labels.repeat(batch_size, 1, 1).permute(0,2,1)

    return l0_xyz, labels.to(pc.device)


def normalize_pc_and_translation(pcs, trans):
    '''
    Shifts pointcloud and grasp translation
    Input:
    - pcs: [B,N,3]
    - trans: [B,3]
    Return:
    - pcs: [B,N,3]
    - trans: [B,3]
    - pc_mean: [B,3]
    '''
    pc_mean = pcs.mean(dim=1)
    pcs = pcs - pc_mean.unsqueeze(1)
    trans = trans-pc_mean

    return pcs, trans, pc_mean

def denormalize_translation(trans, mean=0, std=1):
    '''
    Denormalize the grasp using mean value
    Input:
    - trans: [B,3]
    - pc_mean: [B,3]
    Return:
    - trans: [B,3]
    - pc_mean: [B,3]
    '''
    # if trans.dim() == 3:
    #     mean = mean.repeat(trans.shape[0])
    trans = std * trans + mean

    return trans

def get_surrogate_grasp_points(grasp_rots, grasp_trans, is_euler=False):

    batch_size = grasp_trans.shape[0]

    surrogate_point = np.array( [[0.0000000e+00,  0.0000000e+00 , 7.5273141e-02+0.03]] ) # furthest one
    #surrogate_point = np.array( [[0.0000000e+00,  0.0000000e+00 , 7.5273141e-02+0.015]] ) # center
    #surrogate_point = np.array( [[0.0000000e+00,  0.0000000e+00 , 7.5273141e-02]] ) # closest one

    surrogate_point = np.tile(np.expand_dims(surrogate_point, 0),
                             [batch_size, 1, 1])
    surrogate_point = torch.FloatTensor(surrogate_point).to(grasp_trans.device)

    if is_euler:
        rot = quaternion.euler2matrix(grasp_rots,
                                  order='XYZ')
        surrogate_point = torch.matmul(surrogate_point, rot.permute(0, 2, 1))
        surrogate_point += grasp_trans.unsqueeze(1).expand(-1, surrogate_point.shape[1],
                                                       -1)
                                                    
    else:

        # prepare q and t 
        quat = grasp_rots.unsqueeze(1).repeat([1, surrogate_point.shape[1], 1])
        trans = grasp_trans.unsqueeze(1).repeat([1, surrogate_point.shape[1], 1])

        # rotate and add
        surrogate_point = quaternion.rot_p_by_quaterion(surrogate_point, quat)
        surrogate_point +=trans

    return surrogate_point

def get_control_point_tensor(batch_size, use_torch=True, config_path = 'configs/panda.npy'):
    """
      Outputs a tensor of shape (batch_size x 6 x 3).
      use_torch: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load(config_path)[:, :3]
    mid_point = control_points[0, :]*0.5 + control_points[1, :]*0.5
    control_points = [[0, 0, 0], mid_point, control_points[0, :],
                      control_points[1, :], control_points[-2, :],
                      control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [batch_size, 1, 1])

    if use_torch:
        return torch.tensor(control_points)

    return control_points

def control_points_from_rot_and_trans(grasp_eulers,
                                      grasp_translations, order='XYZ', config_path='configs/panda.npy'):

    rot = quaternion.euler2matrix(grasp_eulers,
                                  order=order)

    grasp_pc = get_control_point_tensor(grasp_eulers.shape[0], config_path=config_path).to(grasp_eulers.device)
    grasp_pc = torch.matmul(grasp_pc, rot.permute(0, 2, 1))
    grasp_pc += grasp_translations.unsqueeze(1).expand(-1, grasp_pc.shape[1],
                                                       -1)
    return grasp_pc