

import torch
import numpy as np 
import argparse

from graspflow5 import GraspFlow
# from scipy.spatial.transform import Rotation as R
import pickle
from networks.utils import normalize_pc_and_translation, denormalize_translation
from pathlib import Path
import matplotlib.pyplot as plt
import complex_environment_utils
from utils.points import regularize_pc_point_count

from utils.auxilary import quaternions2eulers, eulers2quaternions, add_base_options, PandaGripper


'''

'''

obj_names = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser("Experiment: Refine Isaac samples")

parser = add_base_options(parser)
parser.add_argument("--cat", type=str, help="Rot.", choices=obj_names, default='box' )
parser.add_argument("--idx", type=int, help="Rot.", default=14)
parser.add_argument("--grasp_folder", type=str, help="Npz grasps folder.", default="../experiments/generated_grasps_experiment28")
parser.add_argument('--experiment_type', type=str, help='define experiment type for isaac. Ex: complex, single', default='single', choices=['single', 'complex'])

args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# define input options
cat = args.cat
idx = args.idx

print(f'Running {args.method} for {cat}{idx:003} object ... .')
save_info_dir = f'{args.grasp_folder}/refinement_infoholders'
prefix = f'{args.sampler}_{args.classifier}_{args.method}_{args.grasp_space}'

# Load pointclouds and pcs
grasps_file = f'{args.grasp_folder}/{cat}{idx:003}_{args.sampler}.npz'
data = np.load(grasps_file, allow_pickle=True)
if args.experiment_type == 'single':    
    pc = data['pc']
    pc_env = data['pc']
    
elif args.experiment_type == 'complex':
    pc, pc_env, obj_trans, obj_quat, pc1, pc1_view, isaac_seed = complex_environment_utils.parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/shelf001.npz',
                                                               cat=cat, idx=idx, env_num=0,
                                                               filter_epsion=0.3)
    
    pc = np.expand_dims( regularize_pc_point_count(pc, npoints=4096), axis=0)
    pcs_env = np.expand_dims( regularize_pc_point_count(pc_env, npoints=4096), axis=0)


ts = data[f'{args.sampler}_N_N_N_grasps_translations'] # [NUM_ENV, NUM_GRASP, 3]
qs = data[f'{args.sampler}_N_N_N_grasps_quaternions']
thetas = data[f'{args.sampler}_N_N_N_theta']
thetas_pre = data[f'{args.sampler}_N_N_N_theta_pre']
scores = data[f'{args.sampler}_N_N_N_original_scores']
n = ts.shape[1] # number of grasps
num_unique_pcs = ts.shape[0] # number of unique pc, corresponds to num_env
pcs_env = np.repeat(np.expand_dims(pcs_env, axis=1), repeats=n, axis=1)
pc = np.repeat(np.expand_dims(pc, axis=1), repeats=n, axis=1)


# initialize model
graspflow = GraspFlow(args)
graspflow.load_cfg('configs/graspflow_isaac_params.yaml')

# Preprocess inputs
pc = torch.FloatTensor(pc).to(graspflow.device)
pcs_env = torch.FloatTensor(pcs_env).to(graspflow.device)
qs, ts = torch.FloatTensor(qs), torch.FloatTensor(ts)
qs, ts = qs.to(graspflow.device), ts.to(graspflow.device)

# panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
# bb = panda_gripper.get_obbs()

# print('left')
# print(bb[0].to_dict())
# print('right')
# print(bb[1].to_dict())
# print('base')
# print(bb[2].to_dict())

# exit()

kkk = 8 # 1 is ok, 5 shall be red, 8

pc = pc[0,kkk].unsqueeze(0).unsqueeze(0)
pcs_env = pcs_env[0,kkk].unsqueeze(0).unsqueeze(0)
qs = qs[0,kkk].unsqueeze(0).unsqueeze(0)
ts = ts[0,kkk].unsqueeze(0).unsqueeze(0)

# # ts = torch.FloatTensor([[[0,0,0]]]).to(graspflow.device)
# # ts =torch.FloatTensor([[-0.1023,  0.0150,  0.0902],
# #         [-0.0256, -0.1149,  0.0827],
# #         [-0.1266, -0.0393,  0.0295]]).to(graspflow.device)
# # qs = torch.FloatTensor([[[ 0.6787,  0.6464,  0.2130,  0.2761],
# #         [ 0.6295, -0.0352,  0.1427, -0.7630],
# #         [ 0.5746,  0.5956,  0.4869,  0.2794]]]).to(graspflow.device)

# # check
# # ts = ts + 0.1 # NOTE!

# normalize data
pcs_normalized, ts_normalized, pc_means = [], [], []
for i in range(pc.shape[0]):
    _pc, _ts, _pc_means = normalize_pc_and_translation(pc[i], ts[i])
    pcs_normalized.append(_pc)
    ts_normalized.append(_ts)
    pc_means.append(_pc_means)

from networks.models import CollisionDistanceEvaluator
from utils.visualization import gripper_bd
from utils.auxilary import construct_grasp_matrix

ts_normalized[0].requires_grad_(True)
qs[0].requires_grad_(True)

# IDEALLY ONLY THIS PART MUST BE MODIFIED FOR TESTING OTHER CLASSIFIERS
model = CollisionDistanceEvaluator(dist_coeff=10000, dist_threshold=0.000)
_t = ts_normalized[0] + pc_means[0]
logit = model.forward(trans=_t, quat=qs[0], pc_env=pcs_env[0])
scores = torch.sigmoid(logit).detach().cpu().numpy()

print(f'scores: {scores}')
import trimesh
from networks import quaternion

loss = torch.nn.functional.logsigmoid(logit)
loss.backward(torch.ones_like(loss))

print(ts_normalized[0].grad)

scene = trimesh.Scene()
pc_env_mesh = trimesh.points.PointCloud( pcs_env[0,0].detach().cpu().numpy(), colors =[0,255,0,50])
pc_mesh = trimesh.points.PointCloud( pc[0,0].detach().cpu().numpy(), colors =[0,0,255,50])
scene.add_geometry(pc_env_mesh)
scene.add_geometry(pc_mesh)

for i in range(_t.shape[0]):
    g = construct_grasp_matrix(qs[0][i].detach().cpu().numpy(), _t[i].detach().cpu().numpy())
    scene.add_geometry(gripper_bd(scores[i][0]), transform=g[0])

    panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
    # panda_gripper.apply_transformation(transform=g[0])
    bb = panda_gripper.get_bb(all=True)
    bb.visual.face_colors = [125,125,125,80]
    scene.add_geometry(bb, transform=g[0])
    # for _mesh in panda_gripper.get_meshes():
    #     _mesh.visual.face_colors = [255,0,0,200]
    #     scene.add_geometry(_mesh)

    panda_gripper2 = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
    panda_gripper2.apply_transformation(transform=g[0])
    for _mesh in panda_gripper2.get_meshes():
        _mesh.visual.face_colors = [255,0,0,200]
        scene.add_geometry(_mesh)

scene.show()

_pc = pcs_env[0].clone()

_pc = _pc - _t.unsqueeze(1).repeat([1,_pc.shape[1],1])

quat_inv = quaternion.quat_inv(qs[0])
quat_inv = quat_inv.unsqueeze(1).repeat([1, _pc.shape[1], 1])
_pc = quaternion.rot_p_by_quaterion(_pc, quat_inv)

print(_pc.shape)
scene = trimesh.Scene()
pc_env_mesh2 = trimesh.points.PointCloud( _pc[0].detach().cpu().numpy(), colors =[255,0,0,50])
# pc_mesh = trimesh.points.PointCloud( pc[0,0].detach().cpu().numpy(), colors =[0,0,255,50])
scene.add_geometry(pc_env_mesh2)
scene.add_geometry(pc_env_mesh)
# scene.add_geometry(pc_mesh)

scene.add_geometry(gripper_bd(), transform=g[0])


panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
bb = panda_gripper.get_bb(all=True)
bb.visual.face_colors = [125,125,125,80]
scene.add_geometry(bb)

scene.add_geometry(gripper_bd(scores[0][0]))
scene.show()

exit()
