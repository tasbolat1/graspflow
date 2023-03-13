
import torch
import numpy as np 
import argparse

from torch.utils import data
from torch.utils.data import dataset
from datasets import GraspDatasetWithTight as GraspDataset
from utils.auxilary import get_object_mesh, get_transform, sample_grasps, construct_grasp_matrix
from utils.points import regularize_pc_point_count
import trimesh
from utils.visualization import gripper_bd
from graspnet import GraspNet
from scipy.spatial.transform import Rotation as R
import pickle
from networks.utils import normalize_pc_and_translation, denormalize_translation
from pathlib import Path
import matplotlib.pyplot as plt
from utils.auxilary import PandaGripper


obsss = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor']

parser = argparse.ArgumentParser("Experiment 4. Compare sampled grasps from various techniques with refined grasps")

parser.add_argument(
    "--sampler", type=str, choices=["heuristics", "uniform","graspnet"], help="Sampler model to generate grasps.", required=True,
)

parser.add_argument("--max_iterations", type=int, help="Maximum iterations to refine samples.", default=10)
parser.add_argument("--max_success", type=float, help="Maximum success score to refine samples.", default=0.9)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=64)
parser.add_argument("--rot_reps", type=str, help="Rotation representation for refinement.", choices=["quaternion", "euler", "A"], default='euler' )
parser.add_argument("--cat", type=str, help="Rot.", choices=obsss, default='box' )
parser.add_argument("--idx", type=int, help="Rot.", default=0)
parser.add_argument("--method", type=str, choices=['GraspFlow', 'graspnet', 'metropolis'], help="Method for refinement.", default='GraspFlow')

parser.add_argument("--success_threshold", type=float, help="Threshold for defining successfull grasp from logits.", default=0.8)
parser.add_argument("--noise_factor", type=float, help="Noise factor for GraspFlow.", default=0.000001)
parser.add_argument("--f", type=str, choices=['KL', 'JS', 'logD'], help="f-divergence for GraspFlow.", default='KL')
parser.add_argument("--eta_trans", type=float, help="Refiement rate for GraspFlow.", default=0.000001)
parser.add_argument("--eta_rots", type=float, help="Refiement rate for GraspFlow.", default=0.01)
parser.add_argument("--Np", type=int, help="Number of positive examples in the training set.", default=1)
parser.add_argument("--Nq", type=int, help="Number of negative examples in the training set.", default=1)
parser.add_argument("--device", type=int, help="device index. Pass -1 for cpu.", default=-1)
parser.add_argument("--n", type=int, help="Number of grasps to generate.", default=10)
parser.add_argument("--seed", type=int, help="Seed for randomness.", default=40)
# parser.add_argument('--save_dir', type=str, help="Directory to save experiment and it's results.", default='experiment4')
parser.add_argument('--grasp_folder', type=str, help="Initial grasps dir.", default='experiment4')
# parser.add_argument('--view_size', type=int, help="Number of view sizes respect for PointCloud.", default=1)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# define input options
cat = args.cat
idx = args.idx
graspnet = GraspNet(args)
graspnet.load_evaluator('saved_models/165228576343/100.pt') # 164711189287/62.pt

# Load pointclouds and pcs

grasps_file = f'{args.grasp_folder}/{cat}{idx:003}_{args.sampler}.npz'
data = np.load(grasps_file, allow_pickle=True)
pc = data['pc']
pc_env = data['pc']
ts = torch.FloatTensor(data[f'{args.sampler}_N_N_N_grasps_translations'])
qs = torch.FloatTensor(data[f'{args.sampler}_N_N_N_grasps_quaternions'])
num_unique_pcs = ts.shape[0] # number of unique pc, corresponds to num_env
n = ts.shape[1] # number of grasps

pc = data['pc']
pc_env = data['pc']

### STEP 3: prepare grasps
pc = torch.FloatTensor(pc).to(graspnet.device)
qs, ts = qs.to(graspnet.device), ts.to(graspnet.device)



print(f'Running {args.method} for {cat}{idx:003} object ... .')
classifier = 'S'
grasp_space = 'Euler'
save_info_dir = f'{args.grasp_folder}/refinement_infoholders'
prefix = f'{args.sampler}_{classifier}_{args.method}_{grasp_space}'


# Evaluate and refine grasps
total_refined_grasps_translations = np.zeros([num_unique_pcs, n, 3])
total_refined_grasps_quaternions =  np.zeros([num_unique_pcs, n, 4])
total_init_grasps_translations = np.zeros([num_unique_pcs, n, 3])
total_init_grasps_quaternions =  np.zeros([num_unique_pcs, n, 4])
total_refined_time = np.zeros([num_unique_pcs])


for i in range(ts.shape[0]):

    # print(pc[i].unsqueeze(0).shape)
    pc_temp = torch.repeat_interleave(pc[i].unsqueeze(0), repeats=n, dim=0)
    _pc, translations, pc_mean = normalize_pc_and_translation(pc_temp, ts[i])

    ### STEP 4: evaluate and refine grasps
    if args.method == 'GraspFlow':
        quaternions_final, translations_final, success = graspnet.refine_grasps(qs[i], translations, _pc)
    elif args.method == 'graspnet':
        quaternions_final, translations_final, success = graspnet.refine_grasps_graspnet(qs[i], translations, _pc)
    elif args.method == 'metropolis':
        quaternions_final, translations_final, success = graspnet.refine_grasps_metropolis(qs[i], translations, _pc)

    ## candidates are in 
    candidate_grasp_filter = graspnet.info.filter_mask[-1,...]

    print("candidate_grasp_filter :", candidate_grasp_filter )


    # ### STEP 6: save results
    init_translations = denormalize_translation(graspnet.info.translations[0,...], pc_mean.cpu().numpy())
    init_quaternions = graspnet.info.quaternions[0, ...]

    final_translations = denormalize_translation(graspnet.info.translations[-1,...], pc_mean.cpu().numpy())
    final_quaternions = graspnet.info.quaternions[-1, ...]

    total_refined_grasps_translations[i, :,:] = final_translations
    total_init_grasps_translations[i, :,:] = init_translations


# append on npz
data_dict = dict(data)

data_dict[f"{prefix}_grasps_translations"] = total_refined_grasps_translations
data_dict[f"{prefix}_grasps_quaternions"] = total_refined_grasps_quaternions

data_dict[f"{prefix}_init_grasps_translations"] = total_init_grasps_translations
data_dict[f"{prefix}_init_grasps_quaternions"] = total_init_grasps_quaternions

np.savez(grasps_file, **data_dict)

# # # save args
# with open(f'{save_info_dir}/{cat}/{cat}{idx:003}_{prefix}_args.pkl', 'wb') as fp:
#     pickle.dump(args, fp)

print('Success.')

print('------------------------------------------------------')
print('------------------------------------------------------')