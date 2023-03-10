import torch
import numpy as np
from scipy.spatial.transform import Rotation as R
from networks.quaternion import rot_p_by_quaterion


def compensate_camera_frame(transforms, standoff = 0.2):
    # this is ad-hoc
    # TODO: check

    new_transforms = transforms.copy()
    for i, transform in enumerate(transforms):
        standoff_mat = np.eye(4)
        standoff_mat[2] = -standoff
        
        new_transforms[i, :3, 3] = np.matmul(transform,standoff_mat)[:3,3]
    return new_transforms
    

r_g = R.random().as_matrix()
t_g = np.array([0.1,0.2,0.3])

G = np.eye(4)
G[:3,:3] = r_g
G[:3,3] = t_g
G = np.expand_dims(G, axis=0)

new_G = compensate_camera_frame(G, standoff=0.5)

print('Old version:')
print(R.from_matrix(r_g).as_quat())
print(new_G)
print(R.from_matrix(new_G[0][:3,:3]).as_quat())
print(new_G[0][:3,3])
print(np.linalg.norm(new_G[0][:3,3]-t_g))


r = R.from_matrix(r_g).as_quat()
r_torch  = torch.FloatTensor(r).unsqueeze(0).unsqueeze(0)
t_torch = torch.FloatTensor(t_g).unsqueeze(0).unsqueeze(0)

t_torch_new = rot_p_by_quaterion(torch.FloatTensor([0,0,-0.5]).unsqueeze(0).unsqueeze(0), r_torch)

t_torch_new = t_torch_new + t_torch

print(t_torch_new)