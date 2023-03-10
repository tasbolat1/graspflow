### EXPERIMENT 1:
'''
Debugger of the saved grasps. To work properly, need to load
1. Experiment folder
2. Args
3. infoholders

'''

import numpy as np 
np.random.seed(40)
import argparse

from utils.auxilary import get_object_mesh, get_transform, construct_grasp_matrix
import trimesh
from utils.visualization import gripper_bd, quality2color
# from scipy.spatial.transform import Rotation as R
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
# from utils.auxilary import PandaGripper
# from utils.ad_hoc import load_processed_pc, load_grasps
from utils.auxilary import add_base_options
from utils.auxilary import InfoHolder
import pandas as pd
import seaborn as sns
import complex_environment_utils

from utils.auxilary import PandaGripper

def construct_df(x, name='x_vector', norm=False, classifier=None, max_rows=30):

    df_data = {name:[], 'iteration':[], 'grasp_number':[]}

    if classifier is not None:
        df_data['classifier'] = []

    no_skip_row = np.ceil( (x.shape[0]-1)/max_rows)

    for i in range(x.shape[0]):
        if i % no_skip_row != 0:
            continue
        for j in range(x.shape[1]):
            if norm:
                df_data[name].append(np.linalg.norm(x[i,j]))
            else:
                df_data[name].append(x[i,j])
            df_data['iteration'].append(i)
            df_data['grasp_number'].append(j)
            if classifier is not None:
                df_data['classifier'].append(classifier)

    df = pd.DataFrame(df_data)
    return df

obj_names = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser("Experiment: Refine Isaac samples")

parser = add_base_options(parser)
parser.add_argument("--cat", type=str, help="Category of the object.", choices=obj_names, default='box' )
parser.add_argument("--idx", type=int, help="Idx of object.", default=14)
parser.add_argument("--grasp_folder", type=str, help="Experiment dir.", default='../experiments/generated_grasps' )
parser.add_argument("--which_env", type=int, help="Environment idx for isaac.", default=0)
parser.add_argument("--max_grasps", type=int, help="Maximum grasp to visualize the trajectory", default=5)
parser.add_argument('--experiment_type', type=str, help='define experiment type for isaac. Ex: complex, single', default='single')
parser.add_argument('--show_body', type=int, help='1 if the gripper body is to be shown', default=1)

args = parser.parse_args()

# load args
args_and_info_file_dirs = f'{args.grasp_folder}/refinement_infoholders/{args.cat}'
prefix = f'{args.sampler}_{args.classifier}_{args.method}_{args.grasp_space}'

info = InfoHolder()
info.clear()
info.load(f'{args_and_info_file_dirs}/{args.cat}{args.idx:003}_{prefix}_{args.which_env}_info.npz')

# load args
with open(f'{args_and_info_file_dirs}/{args.cat}{args.idx:003}_{prefix}_args.pkl', 'rb') as fp:
    r_args = pickle.load(fp)

print('------------- Debug Gradients ------------------ ')
# TODO: visualize trajectory for specific environment


# ####### GRASPFLOW ###########
# # plot gradients
# fig, ax = plt.subplots(nrows=4, ncols=1)
# all_score = 1
# palletes = {'S': 'red', 'C': 'blue', 'E': 'orange', 'T': 'green', 'N': 'brown'}
# for classifier in args.classifier:
#     t_grad = info.data[f'translations_{classifier}_grad'] # ITERATIONS, B, SIZE
#     r_grad = info.data[f'rots_{classifier}_grad']
#     score = info.data[f'{classifier}_scores'].squeeze(-1)
#     df = construct_df(t_grad, 't_grad_norm', norm=True, classifier=classifier)
#     sns.lineplot(ax=ax[0], data=df, x="iteration", y="t_grad_norm", hue="classifier", palette=[palletes[classifier]])
#     ax[0].set_xlim([0, df.iteration.max()])
#     df = construct_df(r_grad, 'r_grad_norm', norm=True, classifier=classifier)
#     sns.lineplot(ax=ax[1], data=df, x="iteration", y="r_grad_norm", hue="classifier", palette=[palletes[classifier]])
#     ax[1].set_xlim([0, df.iteration.max()])
#     df = construct_df(score, f'Score', classifier=classifier)
#     sns.lineplot(ax=ax[2], data=df, x="iteration", y=f'Score', hue="classifier", palette=[palletes[classifier]])
#     ax[2].set_xlim([0, df.iteration.max()])
#     all_score = all_score*score


# df = construct_df(all_score, f'combined_score')
# sns.lineplot(ax=ax[3], data=df, x="iteration", y=f'combined_score')
# ax[3].set_xlim([0, df.iteration.max()])
# plt.draw()

###### GraspOpt ###########

# plot gradients
fig, ax = plt.subplots(nrows=5, ncols=1)
all_score = 1
t_grad = info.data[f'translations_grad'] # ITERATIONS, B, SIZE
r_grad = info.data[f'rots_grad']
df = construct_df(t_grad, 't_grad_norm', norm=True)
sns.lineplot(ax=ax[0], data=df, x="iteration", y="t_grad_norm")
ax[0].set_xlim([0, df.iteration.max()])
df = construct_df(r_grad, 'r_grad_norm', norm=True)
sns.lineplot(ax=ax[1], data=df, x="iteration", y="r_grad_norm")
ax[1].set_xlim([0, df.iteration.max()])

palletes = {'S': 'red', 'C': 'blue', 'E': 'orange', 'T': 'green', 'N': 'brown'}
for classifier in args.classifier:

    score = info.data[f'{classifier}_scores'].squeeze(-1)
    df = construct_df(score, f'Score', classifier=classifier)
    sns.lineplot(ax=ax[2], data=df, x="iteration", y=f'Score', hue="classifier", palette=[palletes[classifier]])
    ax[2].set_xlim([0, df.iteration.max()])
    all_score = all_score*score

df = construct_df(all_score, f'combined_score')
sns.lineplot(ax=ax[3], data=df, x="iteration", y=f'combined_score')
ax[3].set_xlim([0, df.iteration.max()])

df = construct_df(all_score, f'utility')
sns.lineplot(ax=ax[4], data=df, x="iteration", y=f'utility')
ax[4].set_xlim([0, df.iteration.max()])

plt.show()


# plot histograms
fig, ax = plt.subplots(nrows=len(args.classifier))
bins = np.arange(0,11)/10
ax=ax if len(args.classifier)>1 else [ax]

for idx, classifier in enumerate(args.classifier):
    score = info.data[f'{classifier}_scores'].squeeze(-1)
    
    ax[idx].hist(score[0], bins=bins, facecolor='r', alpha=0.5, label='Initial')
    ax[idx].hist(score[-1], bins=bins, facecolor='b', alpha=0.5, label='Final')
    ax[idx].set_xlim([0,1])
    ax[idx].grid(True)
    ax[idx].set_title(f"{classifier} scores")
    ax[idx].legend()
plt.show()

print('------------- TRAJECTORY VISUALIZATION ------------------ ')
# TODO: visualize trajectory for specific environment

# load npz
data_npz = np.load(f'{args.grasp_folder}/{args.cat}{args.idx:003}_{args.sampler}.npz')

pc = data_npz['pc'][args.which_env]
pc_mesh = trimesh.points.PointCloud(pc)
obj_stable_trans = data_npz['obj_stable_translations'][args.which_env]
obj_stable_quat = data_npz['obj_stable_quaternions'][args.which_env]
obj_transform = get_transform(obj_stable_quat, obj_stable_trans)

# we use all classifier outputs to get quality color
scores = 1
for classifier in args.classifier:
    scores = scores * info.data[f'{classifier}_scores'].squeeze(-1)

ts = info.data[f'translations']
qs = info.data[f'quaternions']

num_grasps = ts.shape[1]
num_iterations = ts.shape[0]

grasps_idx = np.arange(num_grasps) if num_grasps < args.max_grasps else np.random.choice(np.arange(num_grasps), size=args.max_grasps, replace=False)

# load mesh file
if args.experiment_type == 'single':
    obj_mesh = get_object_mesh(args.cat, args.idx)
else:
    obj_mesh = complex_environment_utils.load_shape_for_complex(args.cat, args.idx, path=f'../experiments/composites/{args.experiment_type}')
    _, pc_env, _, _, _, _, _ = complex_environment_utils.parse_isaac_complex_data(path_to_npz=f'../experiments/pointclouds/{args.experiment_type}.npz',
                                                               cat=args.cat, idx=args.idx, env_num=0,
                                                               filter_epsion=0.3)

scene = trimesh.Scene()
scene.add_geometry(pc_mesh)
scene.add_geometry(obj_mesh, transform=obj_transform)

if args.experiment_type != 'single':
    pc_env_mesh = trimesh.points.PointCloud(pc_env, colors = [0,255,0,50])
    scene.add_geometry(pc_env_mesh)

eps = 1e-6

for grasp_idx in grasps_idx:

    # construct grasp transform
    grasps = construct_grasp_matrix(translation=ts[:,grasp_idx,:], quaternion=qs[:,grasp_idx,:])
    # add first grasp as gripper
    scene.add_geometry(gripper_bd(scores[0, grasp_idx]), transform=grasps[0])
    # scene.add_geometry(gripper_bd(), transform=grasps[0])

    # add trajectory as PATH3D
    colors = quality2color(scores[1:,grasp_idx])
    entities = []
    for idx in range(num_iterations-1):
        entities.append( trimesh.path.entities.Line([idx, idx+1]) )

    std = ts[:,grasp_idx,:].std(axis=0)
    if std.sum() > 3*eps:

        traj_handle = trimesh.path.Path3D(entities=entities,
                                                    vertices = ts[:,grasp_idx,:], colors=colors)
        scene.add_geometry(traj_handle)

    # add last grasp
    scene.add_geometry(gripper_bd(scores[-1, grasp_idx]), transform=grasps[-1])

    if args.show_body == 1:
        panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
        panda_gripper.apply_transformation(transform=grasps[-1])
        for _mesh in panda_gripper.get_meshes():
            _mesh.visual.face_colors = [125,125,125,80]
            scene.add_geometry(_mesh)

scene.show()