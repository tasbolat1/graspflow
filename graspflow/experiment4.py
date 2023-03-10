### EXPERIMENT 1:
'''
Experiment 4

Objective: Main experiment file


'''

import torch
import numpy as np 
import argparse

from utils.auxilary import get_object_mesh, get_transform, construct_grasp_matrix
import trimesh
from utils.visualization import gripper_bd
from graspflow import GraspFlow
from scipy.spatial.transform import Rotation as R
import pickle
from networks.utils import normalize_pc_and_translation, denormalize_translation
from pathlib import Path
import matplotlib.pyplot as plt
# from utils.auxilary import PandaGripper
from utils.ad_hoc import load_processed_pc, load_grasps
from utils.auxilary import quaternions2eulers, eulers2quaternions, add_base_options

obj_names = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser("Experiment 1. Test new grasped samples")

parser = add_base_options(parser)
parser.add_argument("--cat", type=str, help="Rot.", choices=obj_names, default='box' )
parser.add_argument("--idx", type=int, help="Rot.", default=14)
parser.add_argument("--n", type=int, help="Number of grasps to generate.", default=1)
parser.add_argument('--save_dir', type=str, help="Directory to save experiment and it's results.", default='experiment')
parser.add_argument('--view_size', type=int, help="Number of view sizes respect for PointCloud.", default=4)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# define input options
cat = args.cat
idx = args.idx

print(f'Running {args.method} for {cat}{idx:003} object ... .')

graspflow = GraspFlow(args)
graspflow.load_evaluator('saved_models/evaluator/165228576343/100.pt') # 164711189287/62.pt

# Load pointclouds
num_uniq_pcs = 10#max(1,int(args.n/1))
pc, obj_pose_relative = load_processed_pc(n=args.n, cat=cat, idx=idx, num_uniq_pcs=num_uniq_pcs, view_size=args.view_size)
pc_mesh = trimesh.points.PointCloud(pc[0])
print(f'Num of unique pcs: {num_uniq_pcs}')

# Load grasps
translations, quaternions = load_grasps(cat, idx, grasps_path=f"/home/tasbolat/some_python_examples/refinement_experiments/grasps_generated_graspnet")
quaternions = torch.FloatTensor(quaternions)
translations = torch.FloatTensor(translations)

# check mismatch
if translations.shape[0] != pc.shape[0]:
    print(f'Warning! pointcloud and grasps size mismatch {translations.shape[0]} != {pc.shape[0]}')
    print('Resizing grasps size')
    translations = translations[:args.n]
    quaternions = quaternions[:args.n]

# Preprocess inputs
pc = torch.FloatTensor(pc).to(graspflow.device)
quaternions, translations = quaternions.to(graspflow.device), translations.to(graspflow.device)
pc, translations, pc_mean = normalize_pc_and_translation(pc, translations)

# # visualize grasps
# grasp_transforms = construct_grasp_matrix(quaternions, translations)
# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc[0]))
# for i, g in enumerate(grasp_transforms):
#     scene.add_geometry(gripper_bd(), transform=g)
# scene.show()

### STEP 4: evaluate and refine grasps
if args.method == 'GraspFlow':
    quaternions_final, translations_final, success = graspflow.refine_grasps(quaternions, translations, pc)
elif args.method == 'graspnet':
    quaternions_final, translations_final, success = graspflow.refine_grasps_graspnet(quaternions, translations, pc)
if args.method == 'metropolis':
    quaternions_final, translations_final, success = graspflow.refine_grasps_metropolis(quaternions, translations, pc)
    
# init_scores = graspflow.info.success[0]
# final_scores = graspflow.info.success[-1]

# # visualize the gradients
# fig, ax = plt.subplots(nrows=2)
# graspflow.info.plot_trans_gradients(ax=ax[0], idx=0)
# graspflow.info.plot_euler_gradients(ax=ax[1], idx=0)
# plt.show()

# # visualize init and final grasps
# vis_n = 50
# grasp_transforms_final = construct_grasp_matrix(quaternions_final, translations_final)
# if grasp_transforms_final.shape[0] > vis_n:
#     print(f'Too many grasps, visualizing only {vis_n}')
# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc[0]))

# for i, g in enumerate(grasp_transforms[:vis_n]):
#     # scene.add_geometry(gripper_bd(init_scores[i]), transform=g)
#     scene.add_geometry(gripper_bd(), transform=g)

# for i, g in enumerate(grasp_transforms_final[:vis_n]):
#     scene.add_geometry(gripper_bd(final_scores[i]), transform=g)
# scene.show()



# # draw all trajectories
# all_q = graspflow.info.quaternions
# all_t = graspflow.info.translations
# scores=  graspflow.info.success
# print(scores)
# scene = trimesh.Scene()
# scene.add_geometry()

# for i, (q,t) in enumerate(zip(all_q, all_t)):
#     g = np.eye(4)
#     g[:3,3] = t
#     g[:3,:3] = R.from_quat(q).as_matrix()
#     # print(scores[i])
#     if i == 0:
#         scene.add_geometry(gripper_bd(), transform=g)
#     else:
#         scene.add_geometry(gripper_bd(scores[i][0]), transform=g)

# scene.show()
# print(all_q.shape, all_t.shape)



init_translations = denormalize_translation(graspflow.info.translations[0,...], pc_mean.cpu().numpy())
init_quaternions = graspflow.info.quaternions[0, ...]

final_translations = denormalize_translation(graspflow.info.translations[-1,...], pc_mean.cpu().numpy())
final_quaternions = graspflow.info.quaternions[-1, ...]

data_dir = f'experiments/{args.save_dir}/{cat}/{cat}{idx:03}'
Path(data_dir).mkdir(parents=True, exist_ok=True)
np.savez(f'{data_dir}/grasps_initial',
            obj_pose_relative = obj_pose_relative,
            translations = init_translations,
            quaternions = init_quaternions)

np.savez(f'{data_dir}/grasps_final',
            obj_pose_relative = obj_pose_relative,
            translations = final_translations,
            quaternions = final_quaternions)
graspflow.info.save(save_dir=f'{data_dir}/info')
with open(f'{data_dir}/args.pkl', 'wb') as fp:
    pickle.dump(args, fp)