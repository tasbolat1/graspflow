

import torch
import numpy as np 
import argparse

from graspflow6 import GraspFlow
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
parser.add_argument('--experiment_type', type=str, help='define experiment type for isaac. Ex: complex, single', default='single')
parser.add_argument('--cfg', type=str, help='Path to config file', default='configs/graspopt_isaac_params.yaml')

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
    
else:

    pc, pc_env, obj_trans, obj_quat, pc1, pc1_view, isaac_seed = complex_environment_utils.parse_isaac_complex_data(path_to_npz=f'../experiments/pointclouds/{args.experiment_type}.npz',
                                                               cat=cat, idx=idx, env_num=0,
                                                               filter_epsion=0.3)
    
    pc = np.expand_dims( regularize_pc_point_count(pc, npoints=4096, use_farthest_point=True), axis=0)
    pcs_env = np.expand_dims( regularize_pc_point_count(pc_env, npoints=4096), axis=0)


ts = data[f'{args.sampler}_N_N_N_grasps_translations'] # [NUM_ENV, NUM_GRASP, 3]
qs = data[f'{args.sampler}_N_N_N_grasps_quaternions']
# thetas = data[f'{args.sampler}_N_N_N_theta']
# thetas_pre = data[f'{args.sampler}_N_N_N_theta_pre']
scores = data[f'{args.sampler}_N_N_N_original_scores']
n = ts.shape[1] # number of grasps
num_unique_pcs = ts.shape[0] # number of unique pc, corresponds to num_env
pcs_env = np.repeat(np.expand_dims(pcs_env, axis=1), repeats=n, axis=1)
pc = np.repeat(np.expand_dims(pc, axis=1), repeats=n, axis=1)


# initialize model
graspflow = GraspFlow(args)
graspflow.load_cfg(args.cfg)

# Preprocess inputs
pc = torch.FloatTensor(pc).to(graspflow.device)
pcs_env = torch.FloatTensor(pcs_env).to(graspflow.device)
qs, ts = torch.FloatTensor(qs), torch.FloatTensor(ts)
qs, ts = qs.to(graspflow.device), ts.to(graspflow.device)

# normalize data
pcs_normalized, ts_normalized, pc_means = [], [], []
for i in range(pc.shape[0]):
    _pc, _ts, _pc_means = normalize_pc_and_translation(pc[i], ts[i])
    pcs_normalized.append(_pc)
    ts_normalized.append(_ts)
    pc_means.append(_pc_means)

def read_query(fname='query.txt', n_envs=1, n_grasps=1):
    file = open(fname,mode='r')
    query = file.read()
    file.close()
    query = np.array(query, dtype='object').reshape((1,))
    query = np.tile(query, [n_envs, n_grasps, 1])
    return query

query = read_query("query.txt", n_envs=ts.shape[0], n_grasps=ts.shape[1])

# print information befor refinement
print(f'Initial score mean (std) {np.mean(scores)} ({np.std(scores)}) from original {args.sampler} evaluator. ')

# Evaluate and refine grasps
total_refined_grasps_translations = np.zeros([num_unique_pcs, n, 3])
total_refined_grasps_quaternions =  np.zeros([num_unique_pcs, n, 4])
total_init_grasps_translations = np.zeros([num_unique_pcs, n, 3])
total_init_grasps_quaternions =  np.zeros([num_unique_pcs, n, 4])
total_refined_time = np.zeros([num_unique_pcs])

total_init_utility = np.zeros([num_unique_pcs, n])
total_refined_utility = np.zeros([num_unique_pcs, n])


total_init_scores={}
total_refined_scores={}
print(args.classifier)
for classifier in args.classifier:
    total_init_scores[f'{classifier}_scores'] = np.zeros([num_unique_pcs, n])
    total_refined_scores[f'{classifier}_scores'] = np.zeros([num_unique_pcs, n])


for i in range(num_unique_pcs):
    ts_refined, qs_refined, info = graspflow.refine_grasps_GraspOptES2(qs[i], ts_normalized[i], pcs_normalized[i], pc_means[i], pc_envs=pcs_env[i], queries=query[i])

    # denormalize
    ts_refined = denormalize_translation(ts_refined, pc_means[i].cpu().numpy())
    info.data['translations'] = denormalize_translation(info.data['translations'], np.tile(pc_means[i].cpu().numpy(), [info.data['translations'].shape[0], 1, 1]))

    total_refined_grasps_translations[i,:,:] = ts_refined
    total_refined_grasps_quaternions[i,:,:] = qs_refined

    total_init_grasps_translations[i,:,:] = info.data['translations'][0,...]
    total_init_grasps_quaternions[i,:,:] = info.data['quaternions'][0,...]

    total_init_utility[i,:] = info.data['utility'][0,...].squeeze(-1)
    total_refined_utility[i,:] = info.data['utility'][-1,...].squeeze(-1)

    total_refined_time[i] = info.exec_time

    for classifier in args.classifier:
        total_refined_scores[f'{classifier}_scores'][i] = info.data[f'{classifier}_scores'][-1,...].squeeze(-1)
        total_init_scores[f'{classifier}_scores'][i] = info.data[f'{classifier}_scores'][0,...].squeeze(-1)

    # save info holder
    Path(f'{save_info_dir}/{cat}/').mkdir(parents=True, exist_ok=True)
    info.save(save_dir=f'{save_info_dir}/{cat}/{cat}{idx:003}_{prefix}_{i}_info')

print('Refinement completed. ')
print(f'Mean refined time: {np.mean(total_refined_time)}')

for classifier in args.classifier:
    print(f'Classifer {classifier} scores:')
    print(f'Initial mean score {np.mean(total_init_scores[f"{classifier}_scores"])}')
    print(f'Final mean score {np.mean(total_refined_scores[f"{classifier}_scores"])}')


# append on npz
data_dict = dict(data)

data_dict[f"{prefix}_grasps_translations"] = total_refined_grasps_translations
data_dict[f"{prefix}_grasps_quaternions"] = total_refined_grasps_quaternions
data_dict[f"{prefix}_utilty"] = total_refined_utility

data_dict[f"{prefix}_init_grasps_translations"] = total_init_grasps_translations
data_dict[f"{prefix}_init_grasps_quaternions"] = total_init_grasps_quaternions
data_dict[f"{prefix}_init_utilty"] = total_init_utility

for classifier in args.classifier:
    data_dict[f"{prefix}_{classifier}_init_scores"] = total_init_scores[f"{classifier}_scores"]
    data_dict[f"{prefix}_{classifier}_final_scores"] = total_refined_scores[f"{classifier}_scores"]
    

data_dict[f"{prefix}_time"] = total_refined_time

np.savez(grasps_file, **data_dict)

# # save args
with open(f'{save_info_dir}/{cat}/{cat}{idx:003}_{prefix}_args.pkl', 'wb') as fp:
    pickle.dump(args, fp)

print('Success.')

print('------------------------------------------------------')
print('------------------------------------------------------')