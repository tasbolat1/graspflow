### EXPERIMENT 1:
'''
Experiment 4

Objective: Main experiment file


'''

from functools import reduce
import torch
import numpy as np 
import argparse

import trimesh
# from grasper.graspsampler.utils import gripper_bd

# from utils.auxilary import get_object_mesh, get_transform, construct_grasp_matrix
# import trimesh
from utils.visualization import gripper_bd
from graspflow5 import GraspFlow
# from scipy.spatial.transform import Rotation as R
import pickle
from networks.pointnet2_utils import pc_normalize
from networks.utils import normalize_pc_and_translation, denormalize_translation
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
from utils.points import regularize_pc_point_count


# from utils.auxilary import PandaGripper
# from utils.ad_hoc import load_processed_pc, load_grasps
from utils.auxilary import construct_grasp_matrix, quaternions2eulers, eulers2quaternions, add_base_options
from robot_ik_model import RobotModel
IK_Q7_ITERATIONS = 30

'''
metropolis:
python refine_isaac_grasps.py  --cat box --idx 14 --max_iterations 3 --method metropolis --grasp_space Euler

graspnet:
python refine_isaac_grasps.py  --cat box --idx 14 --max_iterations 3 --method graspnet --grasp_space Euler

graspflow (Euler):
python refine_isaac_grasps.py  --cat box --idx 14 --max_iterations 3 --method graspflow --grasp_space Euler

graspflow (SO3):
python refine_isaac_grasps.py  --cat box --idx 14 --max_iterations 3 --method graspflow --grasp_space SO3

graspflow (Theta):
python refine_isaac_grasps.py  --cat box --idx 14 --max_iterations 3 --method graspflow --grasp_space Theta

graspflow S+E (Theta):
python refine_isaac_grasps.py  --cat box --idx 14 --max_iterations 3 --method graspflow --grasp_space Theta --include_robot

'''


parser = argparse.ArgumentParser("Experiment: Refine real world samples")

parser = add_base_options(parser)
parser.add_argument("--grasp_folder", type=str, help="Npz grasps folder.", default="../experiments/generated_grasps")
parser.add_argument("--idx", type=int, help="Experiment no.", default=0)
# parser.add_argument("--n", type=int, help="Number of grasps to generate.", default=1)
# parser.add_argument('--save_dir', type=str, help="Directory to save experiment and it's results.", default='experiment')
args = parser.parse_args()

include_robot = False
# if args.include_robot == 1:
#     include_robot = True

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# define input options
#cat = args.cat
idx = args.idx

print('######################################################')
print('######################################################')


print(f'Running {args.method} for {idx:003} object ... .')

#info directory
save_info_dir = f'{args.grasp_folder}/refinement_infoholders'
classifier = args.classifier
prefix = f'{args.sampler}_{classifier}_{args.method}_{args.grasp_space}'

print(f'Prefix {prefix}')

# finetune params for graspflow_euler
# print(f'eta_t = {args.eta_t}')
# print(f'eta_e = {args.eta_e}')
# print(f'noise_t = {args.noise_t}')
# print(f'noise_e = {args.noise_e}')

# print(f'eta_theta_s = {args.eta_theta_s}')
# print(f'eta_theta_e = {args.eta_theta_e}')
# print(f'noise_theta = {args.noise_theta}')
# print(f'robot_threshold = {args.robot_threshold}')
# print(f'robot_coeff = {args.robot_coeff}')

# print(f'S = {classifier}')
# print(f'grasp_space = {args.grasp_space}')
# print(f'max_iterations = {args.max_iterations}')
# print('')

# Load pointclouds and pcs
num_unique_pcs = 1
grasps_file = f'{args.grasp_folder}/{idx:003}_{args.sampler}.npz'
data = np.load(grasps_file)

ts = data['graspnet_N_N_N_grasps_translations'][0]
qs = data['graspnet_N_N_N_grasps_quaternions'][0]
scores = data['graspnet_N_N_N_original_scores'][0]
# print(dict(data).keys())

# print(ts.shape)
# exit()
# exit()
# ts = data['generated_grasps'][:,:3, 3] # [B, 3] = [60, 3]
# qs = R.from_matrix(data['generated_grasps'][:, :3, :3]).as_quat() # [B, 4] = [60, 4]
# scores = data[f'generated_scores'] # [B,] = [60] 
n = ts.shape[0] # number of grasps

data_dir = Path('/home/crslab/catkin_graspflow/src/processed_data/current_data')
pc = np.load(data_dir / 'pc_combined_in_world_frame.npy')
pc = pc[:, :3] # [B, 1024, 3]
pc = regularize_pc_point_count(pc, npoints=1024)

# pc_test = pc.copy()

# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc_test))
# tr = construct_grasp_matrix(qs, ts)
# for g in tr:
#     scene.add_geometry(gripper_bd(), transform=g)
# scene.show()

pc = np.repeat(np.expand_dims(pc, axis=0), repeats=n, axis=0) # [B, N, 3] aka [num samples, num points per pc, 3] = [60, 1024, 3]



# initialize model
graspflow = GraspFlow(args, include_robot=include_robot)
graspflow.load_cfg('configs/graspflow_real_params.yaml')

# Preprocess inputs
pc = torch.FloatTensor(pc).to(graspflow.device)
qs, ts = torch.FloatTensor(qs), torch.FloatTensor(ts)
qs, ts = qs.to(graspflow.device), ts.to(graspflow.device)
#thetas = torch.FloatTensor(thetas).to(graspflow.device)

# normalize data
pc_normalized, ts_normalized, pc_mean = normalize_pc_and_translation(pc, ts)

# define robot model
panda_robot = RobotModel(angle_iterations=IK_Q7_ITERATIONS)

# print information befor refinement
print(f'Initial score mean (std) {np.mean(scores)} ({np.std(scores)}) from original {args.sampler} evaluator. ')

# Evaluate and refine grasps
total_refined_grasps_translations = np.zeros([num_unique_pcs, n, 3])
total_refined_grasps_quaternions =  np.zeros([num_unique_pcs, n, 4])
total_refined_grasps = np.zeros([num_unique_pcs, n, 4,4])
total_refined_scores = np.zeros([num_unique_pcs, n])
total_refined_time = np.zeros([num_unique_pcs, n])


all_infos = []
all_init_scores = np.zeros([num_unique_pcs, n])

all_robot_init_scores = np.zeros([num_unique_pcs, n])
all_robot_refined_scores = np.zeros([num_unique_pcs, n])


for i in range(num_unique_pcs):
    # ts_refined, qs_refined, init_scores, final_scores, info = graspflow.refine_grasps_tactile(qs, ts_normalized, pc_normalized)
    ts_refined, qs_refined, info = graspflow.refine_grasps_GraspFlow(qs, ts_normalized, pc_normalized, pc_mean, classifiers=args.classifier)
    
    print(info.var_names)
    
    ts_refined = denormalize_translation(ts_refined, pc_mean.cpu().numpy())

    total_refined_grasps_translations[i,:,:] = ts_refined
    total_refined_grasps_quaternions[i,:,:] = qs_refined
    
    init_scores = [info.data[f'{cl}_scores'][0,...].squeeze(-1) for cl in args.classifier]
    init_score = init_scores[0]
    for sc in init_scores[1:]:
        init_score = init_score.dot(sc)
    all_init_scores[i] = init_score

    final_scores = [info.data[f'{cl}_scores'][-1,...].squeeze(-1) for cl in args.classifier]
    final_score = final_scores[0]
    for sc in final_scores[1:]:
        final_score = final_score.dot(sc)
    total_refined_scores[i] = final_score
    #total_refined_theta[i] = refined_theta
    # total_refined_theta_pre[i] = refined_theta_pre
    total_refined_time[i] = info.exec_time

    all_infos.append(info)
    
    

    # save info holder
    Path(f'{save_info_dir}/').mkdir(parents=True, exist_ok=True)
    info.save(save_dir=f'{save_info_dir}/{idx:003}_{prefix}_{i}_info')

print('Refinement completed: ')
print(f'Initial mean score {np.mean(all_init_scores)}')
print(f'Final mean score {np.mean(total_refined_scores)}')
print(f'Initial robot mean score {np.mean(all_robot_init_scores)}')
print(f'Final robot mean score {np.mean(all_robot_refined_scores)}')
#print(f'Mean refined time: {np.mean(total_refined_time)}')

# append on npz
data_dict = dict(data)
print(prefix)
data_dict[f"{prefix}_grasps_translations"] = total_refined_grasps_translations
data_dict[f"{prefix}_grasps_quaternions"] = total_refined_grasps_quaternions
data_dict[f"{prefix}_scores"] = total_refined_scores
data_dict[f"{prefix}_init_scores"] = all_init_scores
data_dict[f"{prefix}_robot_scores"] = all_robot_refined_scores
data_dict[f"{prefix}_init_robot_scores"] = all_robot_init_scores
data_dict[f"{prefix}_time"] = total_refined_time

np.savez(grasps_file, **data_dict)

# # save args
with open(f'{save_info_dir}/{idx:003}_{prefix}_args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Success.')

print('------------------------------------------------------')
print('------------------------------------------------------')


# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc_test))

# tr2 = construct_grasp_matrix(total_refined_grasps_quaternions[0], total_refined_grasps_translations[0])
# for g in tr:
#     scene.add_geometry(gripper_bd(), transform=g)
# for g in tr2:
#     scene.add_geometry(gripper_bd(1), transform=g)
# scene.show()