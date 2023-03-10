from re import T
import torch
import torch.nn as nn
from networks.quaternion import quat2euler, quaternion_mult, rot_p_by_quaterion, quaternion_conj, euler2matrix, quat_normalize
from scipy.spatial.transform import Rotation as R
from networks.utils import control_points_from_rot_and_trans, transform_gripper_pc_old
import numpy as np


# Given, p0, q, e, f
def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)

def conj(q):
    q_conj = q.clone()
    q_conj[:3] *= -1
    return q_conj

p0 = torch.FloatTensor([[[1.0, 2.0, 3.0]]])


f = nn.Linear(3, 1)
f.weight.data.fill_(1.0)
f.bias.data.fill_(1.0)


print('Quat')
q = torch.FloatTensor([[123, 214, 513, 123]])
q = quat_normalize(q)
print(q)
print(q.shape)
# q = torch.FloatTensor([[[0.1826, 0.7303, 0.3651, 0.5477]]]) # wxyz
# # q.requires_grad_(True)

# # if q.grad is not None:
# #     q.grad.zero_()

# # step 1
# p1 = rot_p_by_quaterion(p0, q)#qrot(q,p0)#
# print(p1)
# s = f(p1)
# print(s)
# # s.backward(torch.ones_like(s))
# # print(q.grad)


# # new_e = quat2euler(q.grad, order='XYZ')
# # print(new_e)

# # # step 2
# print('Euler')
# e = quat2euler(q, order='XYZ')
# print(e)
# e = torch.FloatTensor([[[-1.9515,  1.2035,  2.7613]]])

# # e = e.clone()
# e.requires_grad_(True)

# if e.grad is not None:
#     e.grad.zero_()
# # if q.grad is not None:
# #     q.grad.zero_()

# e_temp = e.squeeze(0)
# rot = euler2matrix(e_temp, order='XYZ')
# p1_hat = torch.matmul(p0, rot.permute(0,2,1))

# print(p1_hat)

# s_hat = f(p1_hat)
# print(s_hat)

# s_hat.backward(torch.ones_like(s_hat))
# print(e.grad)


# print( R.from_euler('XYZ', e.grad.squeeze().numpy()).as_quat())


