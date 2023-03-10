from cmath import isnan
from hashlib import new
from operator import is_
import numpy as np
from sympy import limit
import torch
import franka_ik_pybind
from utils.auxilary import construct_grasp_matrix
import trimesh
from utils.visualization import gripper_bd
from scipy.spatial.transform import Rotation as R
import pickle
from robot_model import calc_man_score

def get_extended_grasp_translation(quaternions, translations, standoff=0.1):
    transforms = construct_grasp_matrix(quaternions, translations)
    transform_extended = np.repeat(np.expand_dims(np.eye(4), axis=0), translations.shape[0], axis=0)
    transform_extended[:,2,:] = -standoff
    new_transform = np.einsum("bij, bjk -> bik", transforms, transform_extended)
    return new_transform[:,:3,3]

def check_limits(joint_angles, q_min_epsilon=0.2, q_max_epsilon=0.2):
    '''
    joint_angles: [N,7]
    '''
    q_min = np.array([-2.89, -1.76, -2.89, -3.07, -2.89, 0, -2.89]) + q_min_epsilon
    q_max = np.array([2.89, 1.76, 2.89, -0.0698, 2.89, 3.75, 2.89]) - q_max_epsilon
    res = np.zeros(joint_angles.shape[0], dtype=bool)
    for i in range(joint_angles.shape[0]):
        q = joint_angles[i,:]
        if np.isnan(np.sum(q)):
            res[i] = False
            continue
        lower_mask = (q > q_min).all()
        upper_mask = (q < q_max).all()
        if lower_mask and upper_mask:
            res[i] = True
    return res

def solve_ik(q, t, max_iterations=100):
    '''
    Input:
        - q: (4,) quaternions
        - t: (3,) translation
    Return:
        - joint_angles (7,) joint angles in radians
    '''

    angles = np.linspace(-2.69, 2.69, max_iterations)
    for angle in angles:
        q7 = angle

        initial_joint_angles = np.array([-0.00034485234427340453, -0.7847331501140928,
                                        -0.00048777872079729497, -2.3551600892113274,
                                        -0.0009046530319716893, 1.5725385250250496,
                                        q7])

        joint_angles = franka_ik_pybind.franka_IK(t, q, q7, initial_joint_angles)
        possible_solution_rows_mask = np.isnan(joint_angles).any(axis=1)
        
        if np.sum(possible_solution_rows_mask) < joint_angles.shape[0]:

            joint_angles = joint_angles[~possible_solution_rows_mask, :]
            limit_mask = check_limits(joint_angles)
            if np.sum(limit_mask) < joint_angles.shape[0]:
                return  joint_angles[np.argmax(limit_mask==True)]

    return np.array([np.nan]*7)
        

# solve IK
def grasp2thetas(quaternions, translations):
    B = translations.shape[0]

    is_tensor = False
    if isinstance(translations, torch.Tensor):
        dev = translations.device
        translations = translations.detach().cpu().numpy()
        quaternions = quaternions.detach().cpu().numpy()
        is_tensor = True

    
    # fake grasp
    ext_translations = get_extended_grasp_translation(quaternions, translations)

    # recast
    ext_translations = ext_translations.astype(np.float64)
    translations = translations.astype(np.float64)
    quaternions = quaternions.astype(np.float64)

    # solve ik
    thetas = np.zeros([translations.shape[0], 7])
    thetas_extended = np.zeros([translations.shape[0], 7])
    is_nan = np.zeros(translations.shape[0], dtype=bool)
    for i in range(B):
        thetas[i,:] = solve_ik(quaternions[i], translations[i])
        thetas_extended[i,:] = solve_ik(quaternions[i], ext_translations[i])
        if (np.sum(np.isnan(thetas[i,:])) == 7) or (np.sum(np.isnan(thetas_extended[i,:])) == 7):
            is_nan[i] = True

    # get scores
    omega = calc_man_score(thetas, link_name='panda_gripper_center')
    omega_extended = calc_man_score(thetas, link_name='fake_target')
    
    # case 1: find nans in thetas and theta_extended
    nonnan_idx = np.argwhere(~np.isnan(omega))
    nonnan_idx_extended = np.argwhere(~np.isnan(omega_extended))

    
    # case 2: find some thresholded values
    good_idx = np.argwhere(omega<0.01)
    good_idx_extended = np.argwhere(omega_extended<0.01)
    good_idx = np.vstack([nonnan_idx, nonnan_idx_extended, good_idx, good_idx_extended])
    good_idx = np.unique(good_idx).astype(int)

    if len(good_idx) == 0:
        raise ValueError('NO IK SOLUTION FOUND!.')
    
    # resample positives
    all_idx = np.arange(0, translations.shape[0], 1, dtype=int)
    bad_idx = np.setdiff1d(all_idx, good_idx, assume_unique=True)
    new_good_idx = np.random.choice(good_idx, size=len(bad_idx))
    thetas[bad_idx,:] = thetas[new_good_idx, :] + np.random.rand(len(bad_idx),7)*0.001

    if is_tensor:
        return torch.FloatTensor(thetas).to(dev)

    return thetas


# solve IK
def grasp2thetas_light(quaternions, translations, link_name='fake_target'):
    B = translations.shape[0]

    is_tensor = False
    if isinstance(translations, torch.Tensor):
        dev = translations.device
        translations = translations.detach().cpu().numpy()
        quaternions = quaternions.detach().cpu().numpy()
        is_tensor = True


    # recast
    translations = translations.astype(np.float64)
    quaternions = quaternions.astype(np.float64)

    # solve ik
    thetas = np.zeros([translations.shape[0], 7])
    is_nan = np.zeros(translations.shape[0], dtype=bool)
    for i in range(B):
        thetas[i,:] = solve_ik(quaternions[i], translations[i])
        if np.sum(np.isnan(thetas[i,:])) == 7:
            is_nan[i] = True

    # get scores
    omega = calc_man_score(thetas, link_name=link_name)
    
    # case 1: find nans in thetas and theta_extended
    nonnan_idx = np.argwhere(~np.isnan(omega))

    # case 2: find some thresholded values
    good_idx = np.argwhere(omega<0.01)
    good_idx = np.vstack([nonnan_idx, good_idx])
    good_idx = np.unique(good_idx).astype(int)

    if len(good_idx) == 0:
        raise ValueError('NO IK SOLUTION FOUND!.')
    
    # resample positives
    all_idx = np.arange(0, translations.shape[0], 1, dtype=int)
    bad_idx = np.setdiff1d(all_idx, good_idx, assume_unique=True)
    #new_good_idx = np.random.choice(good_idx, size=len(bad_idx))
    #thetas[bad_idx,:] = thetas[new_good_idx, :] + np.random.rand(len(bad_idx),7)*0.001

    if is_tensor:
        return torch.FloatTensor(thetas).to(dev), torch.FloatTensor(bad_idx).to(dev)

    return thetas, bad_idx

    
if __name__ == "__main__":
    # # load grasps
    # grasp_data = np.load('/home/tasbolat/some_python_examples/GRASP/grasp_network/experiments/experiment_real_test/grasps/sugarbox/sugar_box_graspnet200_no_refine.npz')
    # translations = grasp_data['translations']#[:100,:]
    # # print(translations)
    # quaternions = grasp_data['quaternions']#[:2,:]
    # # print(translations.shape, quaternions.shape)
    # grasp2thetas(quaternions, translations)

    # q_min = torch.FloatTensor([-2.89, -1.76, -2.89, -3.07, -2.89, 0, -2.89]) + q_min_epsilon
    # q_max = torch.FloatTensor([2.89, 1.76, 2.89, -0.0698, 2.89, 3.75, 2.89]) - q_max_epsilon
    q = np.array([[ 0.1009,  0.8533,  0.3999,  0.3191], [-0.0420,  0.9885,  0.0656, -0.1296]])
    t = np.array([[-0.0956, -0.0729,  0.0486], [ 0.0281, -0.0078,  0.1136]])
    a = grasp2thetas(q,t)
    # a = np.array([[ 0.8854,  1.0191,  0.6736, -2.0673, -1.9636,  0.9517, -1.7657]])
    print(a)