### EXPERIMENT 1:
'''
Experiment 4

Objective: Main experiment file


'''

import torch
import numpy as np 
import argparse

# from utils.auxilary import get_object_mesh, get_transform, construct_grasp_matrix
# import trimesh
# from utils.visualization import gripper_bd
from graspflow2 import GraspFlow
# from scipy.spatial.transform import Rotation as R
import pickle
from networks.utils import normalize_pc_and_translation, denormalize_translation
from pathlib import Path
import matplotlib.pyplot as plt

# from utils.auxilary import PandaGripper
# from utils.ad_hoc import load_processed_pc, load_grasps
from utils.auxilary import quaternions2eulers, eulers2quaternions, add_base_options
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

obj_names = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser("Experiment: Refine Isaac samples")

parser = add_base_options(parser)
parser.add_argument("--cat", type=str, help="Rot.", choices=obj_names, default='box' )
parser.add_argument("--idx", type=int, help="Rot.", default=14)
parser.add_argument("--grasp_folder", type=str, help="Npz grasps folder.", default="../experiments/generated_grasps_experiment8")
# parser.add_argument("--n", type=int, help="Number of grasps to generate.", default=1)
# parser.add_argument('--save_dir', type=str, help="Directory to save experiment and it's results.", default='experiment')
args = parser.parse_args()

include_robot = False
if args.include_robot == 1:
    include_robot = True

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# define input options
cat = args.cat
idx = args.idx

print('######################################################')
print('######################################################')


print(f'Running {args.method} for {cat}{idx:003} object ... .')

#info directory
save_info_dir = f'{args.grasp_folder}/refinement_infoholders'
# classifier = 'S'
# if include_robot:
#     classifier = 'SE' 
prefix = f'{args.sampler}_{args.classifier}_{args.method}_{args.grasp_space}'

print(f'Prefix {prefix}')

print(f'eta_t = {args.eta_t}')
print(f'eta_e = {args.eta_e}')
print(f'noise_t = {args.noise_t}')
print(f'noise_e = {args.noise_e}')

print(f'eta_theta_s = {args.eta_theta_s}')
print(f'eta_theta_e = {args.eta_theta_e}')
print(f'noise_theta = {args.noise_theta}')
print(f'robot_threshold = {args.robot_threshold}')
print(f'robot_coeff = {args.robot_coeff}')

print(f'Classifier = {args.classifier}')
print(f'grasp_space = {args.grasp_space}')
print(f'max_iterations = {args.max_iterations}')
print('')

# Load pointclouds and pcs
grasps_file = f'{args.grasp_folder}/{cat}{idx:003}_{args.sampler}.npz'
data = np.load(grasps_file)
pc = data['pc']
ts = data[f'{args.sampler}_N_N_N_grasps_translations'] # [NUM_ENV, NUM_GRASP, 3]
qs = data[f'{args.sampler}_N_N_N_grasps_quaternions']
thetas = data[f'{args.sampler}_N_N_N_theta']
thetas_pre = data[f'{args.sampler}_N_N_N_theta_pre']
scores = data[f'{args.sampler}_N_N_N_original_scores']
n = ts.shape[1] # number of grasps
num_unique_pcs = ts.shape[0] # number of unique pc, corresponds to num_env
pc = np.repeat(np.expand_dims(pc, axis=1), repeats=n, axis=1)

# initialize model
graspflow = GraspFlow(args, include_robot=include_robot)
graspflow.load_evaluator('saved_models/evaluator/165228576343/100.pt') # 164711189287/62.pt

# Preprocess inputs
pc = torch.FloatTensor(pc).to(graspflow.device)
qs, ts = torch.FloatTensor(qs), torch.FloatTensor(ts)
qs, ts = qs.to(graspflow.device), ts.to(graspflow.device)
thetas = torch.FloatTensor(thetas).to(graspflow.device)

# normalize data
pcs_normalized, ts_normalized, pc_means = [], [], []
for i in range(pc.shape[0]):
    _pc, _ts, _pc_means = normalize_pc_and_translation(pc[i], ts[i])
    pcs_normalized.append(_pc)
    ts_normalized.append(_ts)
    pc_means.append(_pc_means)

# define robot model
panda_robot = RobotModel(angle_iterations=IK_Q7_ITERATIONS)

# print information befor refinement
print(f'Initial score mean (std) {np.mean(scores)} ({np.std(scores)}) from original {args.sampler} evaluator. ')

# Evaluate and refine grasps
total_refined_grasps_translations = np.zeros([num_unique_pcs, n, 3])
total_refined_grasps_quaternions =  np.zeros([num_unique_pcs, n, 4])
total_refined_grasps = np.zeros([num_unique_pcs, n, 4,4])
total_refined_time = np.zeros([num_unique_pcs])

total_init_scores = np.zeros([num_unique_pcs, n])
total_refined_scores = np.zeros([num_unique_pcs, n])

total_robot_init_scores = np.zeros([num_unique_pcs, n])
total_robot_refined_scores = np.zeros([num_unique_pcs, n])

total_table_init_scores = np.zeros([num_unique_pcs, n])
total_table_refined_scores = np.zeros([num_unique_pcs, n])

for i in range(num_unique_pcs):

    #### GRASPNET #####
    if args.method == "graspnet":
        ts_refined, qs_refined, init_scores, final_scores, info = graspflow.refine_grasps_graspnet(qs[i], ts_normalized[i], pcs_normalized[i])
    elif args.method == "metropolis":
        ts_refined, qs_refined, init_scores, final_scores, info = graspflow.refine_grasps_metropolis(qs[i], ts_normalized[i], pcs_normalized[i])
    elif args.method == "GraspFlow":
        ts_refined, qs_refined, info = graspflow.refine_grasps_GrapsFlow(qs[i], ts_normalized[i], pcs_normalized[i], pc_means[i], classifiers=args.classifier)
        
        total_robot_refined_scores[i] = info.data['robot_scores'][-1,...].squeeze(-1)
        total_robot_init_scores[i] = info.data['robot_scores'][0,...].squeeze(-1)
        total_table_refined_scores[i] = info.data['table_scores'][-1,...].squeeze(-1)
        total_table_init_scores[i] = info.data['table_scores'][0,...].squeeze(-1)

    # denormalize
    ts_refined = denormalize_translation(ts_refined, pc_means[i].cpu().numpy())
    info.data['translations'] = denormalize_translation(info.data['translations'], np.tile(pc_means[i].cpu().numpy(), [info.data['translations'].shape[0], 1, 1]))

    total_refined_grasps_translations[i,:,:] = ts_refined
    total_refined_grasps_quaternions[i,:,:] = qs_refined
    total_refined_scores[i] = info.data['grasp_scores'][-1,...].squeeze(-1)
    total_refined_time[i] = info.exec_time
    total_init_scores[i] = info.data['grasp_scores'][0,...].squeeze(-1)

    # save info holder
    Path(f'{save_info_dir}/{cat}/').mkdir(parents=True, exist_ok=True)
    info.save(save_dir=f'{save_info_dir}/{cat}/{cat}{idx:003}_{prefix}_{i}_info')

print('Refinement completed: ')
print(f'Initial mean score {np.mean(total_init_scores)}')
print(f'Final mean score {np.mean(total_refined_scores)}')
print(f'Initial robot mean score {np.mean(total_robot_init_scores)}')
print(f'Final robot mean score {np.mean(total_robot_refined_scores)}')
print(f'Initial table mean score {np.mean(total_table_init_scores)}')
print(f'Final table mean score {np.mean(total_table_refined_scores)}')
print(f'Mean refined time: {np.mean(total_refined_time)}')

# append on npz
data_dict = dict(data)
print(prefix)
data_dict[f"{prefix}_grasps_translations"] = total_refined_grasps_translations
data_dict[f"{prefix}_grasps_quaternions"] = total_refined_grasps_quaternions
data_dict[f"{prefix}_scores"] = total_refined_scores
data_dict[f"{prefix}_init_scores"] = total_init_scores
data_dict[f"{prefix}_robot_scores"] = total_robot_refined_scores
data_dict[f"{prefix}_init_robot_scores"] = total_robot_init_scores
data_dict[f"{prefix}_table_scores"] = total_table_refined_scores
data_dict[f"{prefix}_init_table_scores"] = total_table_init_scores
data_dict[f"{prefix}_time"] = total_refined_time

np.savez(grasps_file, **data_dict)

# # save args
with open(f'{save_info_dir}/{cat}/{cat}{idx:003}_{prefix}_args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Success.')

print('------------------------------------------------------')
print('------------------------------------------------------')