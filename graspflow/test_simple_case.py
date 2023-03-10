from math import degrees
import torch
import torch.nn as nn
from networks.quaternion import quat2euler, quaternion_mult, rot_p_by_quaterion, quaternion_conj, euler2matrix, quat_normalize
from scipy.spatial.transform import Rotation as R
from networks.utils import control_points_from_rot_and_trans, transform_gripper_pc_old
import numpy as np
import trimesh
from utils.auxilary import InfoHolder, construct_grasp_matrix, eulers2quaternions
from utils.visualization import gripper_bd
from graspflow_toy import GraspFlow
import argparse

'''
1. Build distance model
2. Write down GraspFlow
3. Create a grasp only angle dependent
4. Test them

quaternion version
python test_simple_case.py --max_iteration 100 --eta_trans 0.0 --eta_eulers 0.1 --noise_factor 0.0

euler version
python test_simple_case.py --max_iteration 100 --eta_trans 0.0 --eta_eulers 0.1 --noise_factor 0.0


'''

parser = argparse.ArgumentParser("Experiment 1. Test new grasped samples")
parser.add_argument("--max_iterations", type=int, help="Maximum iterations to refine samples.", default=10)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=64)
parser.add_argument("--noise_factor", type=float, help="Noise factor for DFflow.", default=0.000001)
parser.add_argument("--eta_trans", type=float, help="Refiement rate for DFflow.", default=0.000025)
parser.add_argument("--eta_eulers", type=float, help="Refiement rate for DFflow.", default=0.001)
parser.add_argument("--device", type=int, help="device index. Pass -1 for cpu.", default=0)
# parser.add_argument("--n", type=int, help="Number of grasps to generate.", default=11)
args = parser.parse_args()

z_plane = trimesh.primitives.Box(extents=[1,1,0.001])
z_plane.visual.face_colors=[155,155,155,50]

# _1 = R.from_rotvec([30, np.pi/2,-20], degrees=True).as_matrix()
# # _1 = R.from_euler(angles=[30, np.pi/2,-20], degrees=True, seq='XYZ').as_matrix()
# _2 = R.from_matrix(_1).as_euler(seq='XYZ', degrees=True)
# print(_2)
# _3 = R.from_matrix(_1).as_rotvec(degrees=True)
# print(_3)

N = 5000
graspflow = GraspFlow(args)
graspflow.load_evaluator()
e = torch.FloatTensor([[1.5, -1.6, -2.5]])
# e = e.repeat([N, 1])
q = torch.FloatTensor(eulers2quaternions(e, seq='XYZ')).to(torch.device('cuda:0'))

# q = torch.FloatTensor([[0,0.707,0,0.707]])
# q = torch.FloatTensor([[0,0,0,1]])
# q = torch.FloatTensor([[1,0,0,0]])

t = torch.FloatTensor([[0,0,0.0]]).to(torch.device('cuda:0'))
# t = t.repeat([N, 1])
# print(t.shape)
# print(q.shape)
g = construct_grasp_matrix(q,t)

scene = trimesh.Scene()
scene.add_geometry(z_plane)
scene.add_geometry(gripper_bd(), transform=g[0])
scene.show()

q_final, t_final, success = graspflow.refine_grasps_SO3(q,t)
# q_final, t_final, success = graspflow.refine_grasps_euler(q,t)

print(success.shape)

graspflow.info.save('temporary')

# q_final, t_final, success = graspflow.refine_grasps_metropolis(q,t)
# q_final, t_final, success = graspflow.refine_grasps_graspnet(q,t)
# q_final, t_final, success = graspflow.refine_grasps(q,t)

# # draw all trajectories
all_q = graspflow.info.quaternions
all_t = graspflow.info.translations
scores=  graspflow.info.success

scene = trimesh.Scene()

scene.add_geometry(z_plane)

for i, (q,t) in enumerate(zip(all_q, all_t)):
    g = np.eye(4)
    g[:3,3] = t
    g[:3,:3] = R.from_quat(q).as_matrix()
    # print(scores[i])
    if i == 0:
        scene.add_geometry(gripper_bd(), transform=g)
    else:
        scene.add_geometry(gripper_bd(scores[i][0]), transform=g)

scene.show()
print(all_q.shape, all_t.shape)