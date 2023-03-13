### EXPERIMENT 4:
'''
Experiment 4:
Outline:
1. Sample grasps from [heuristics, graspnet]
2. Refine the grasps
    - check how accuracy increases
    - filter out: far grasps, low gradient & far grasps, and?
3. Plot the grasps
    - initial grasps, final grasps
    - plot gradients
4. Save results
'''

from nbformat import read
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

parser = argparse.ArgumentParser("Experiment 4. Compare sampled grasps from various techniques with refined grasps")

parser.add_argument(
    "--sampler", type=str, choices=["heuristics", "uniform","graspnet"], help="Sampler model to generate grasps.", required=True,
)

parser.add_argument("--max_iterations", type=int, help="Maximum iterations to refine samples.", default=10)
parser.add_argument("--max_success", type=float, help="Maximum success score to refine samples.", default=0.9)
parser.add_argument("--batch_size", type=int, help="Batch size.", default=64)
parser.add_argument("--rot_reps", type=str, help="Rotation representation for refinement.", choices=["quaternion", "euler", "A"], default='euler' )
parser.add_argument("--method", type=str, choices=['DGflow', 'graspnet', 'metropolis'], help="Method for refinement.", default='DGflow')

parser.add_argument("--success_threshold", type=float, help="Threshold for defining successfull grasp from logits.", default=0.8)
parser.add_argument("--noise_factor", type=float, help="Noise factor for DFflow.", default=0.000001)
parser.add_argument("--f", type=str, choices=['KL', 'JS', 'logD'], help="f-divergence for DFflow.", default='KL')
parser.add_argument("--eta_trans", type=float, help="Refiement rate for DFflow.", default=0.000001)
parser.add_argument("--eta_rots", type=float, help="Refiement rate for DFflow.", default=0.01)
parser.add_argument("--Np", type=int, help="Number of positive examples in the training set.", default=1)
parser.add_argument("--Nq", type=int, help="Number of negative examples in the training set.", default=1)
parser.add_argument("--device", type=int, help="device index. Pass -1 for cpu.", default=-1)
parser.add_argument("--n", type=int, help="Number of grasps to generate.", default=10)
parser.add_argument("--seed", type=int, help="Seed for randomness.", default=40)
parser.add_argument('--save_dir', type=str, help="Directory to save experiment and it's results.", default='experiment4')
parser.add_argument('--view_size', type=int, help="Number of view sizes respect for PointCloud.", default=1)
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)

# define input options
graspnet = GraspNet(args)
graspnet.load_evaluator('saved_models/evaluator/165228576343/47.pt')#164711189287/62.pt
num_uniq_pcs = max(1,int(args.n/100))

print(f'Num of unique pcs: {num_uniq_pcs}')
obj_name = 'sugarbox'

### STEP 1: prepare data
def read_pc():

    with open(f'experiments/experiment_real_test/pcl/{obj_name}.pkl', 'rb') as f:
        pc = np.array(pickle.load(f))[:,:3]

    pc = regularize_pc_point_count(pc, 1024, False)
    return pc

# load pointclouds
repeat_times = args.n
_pc = read_pc()
pc = np.expand_dims(_pc, axis=0)
pc = np.repeat(pc, repeat_times, axis=0)

assert pc.shape[0] == args.n, print(f'pc size does not match args.n {pc.shape[0]} neq {args.n}')

### STEP 2: sample grasps
def load_grasps(args):
    data_filename = f"experiments/grasps_for_real_test/{obj_name}.npz"
    data = np.load(data_filename)
    translations = data["translations"]
    quaternions = data["quaternions"]
    return translations, quaternions

def move_backward_grasp(transform, standoff = 0.2):
    standoff_mat = np.eye(4)
    standoff_mat[2] = -standoff
    new = np.matmul(transform,standoff_mat)
    return new[:3,3]

if args.sampler == "graspnet":
    translations, quaternions = load_grasps(args)
    quaternions = torch.from_numpy(quaternions)
    translations = torch.from_numpy(translations)
    print("translation:", translations.shape)
else:
    quaternions, translations = graspnet.sample_grasp(pc)
    

# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc[0]))
# for i, g in enumerate(grasp_transforms):
#     scene.add_geometry(gripper_bd(), transform=g)
#     if i == 2:
#         break

# scene.show()


### STEP 3: prepare grasps
pc = torch.FloatTensor(pc).to(graspnet.device)
quaternions, translations = quaternions.to(graspnet.device), translations.to(graspnet.device)
pc, translations, pc_mean = normalize_pc_and_translation(pc, translations)
grasp_transforms = construct_grasp_matrix(quaternions, translations)

### move the grasp backward 
for i in range(len(grasp_transforms)):
    new_translation = move_backward_grasp(grasp_transforms[i], standoff=0.01)
    grasp_transforms[i][:3,3] = new_translation
    translations[i] = torch.from_numpy(new_translation).to(graspnet.device)
# translations[:,1] += 0.01 # work for bowl



# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc[0].cpu().numpy(), colors=[255,0,0,255]))
# for i, g in enumerate(grasp_transforms):
#     scene.add_geometry(gripper_bd(), transform=g)

# scene.show()


### STEP 4: evaluate and refine grasps
if args.method == 'DGflow':
    quaternions_final, translations_final, success = graspnet.refine_grasps(quaternions, translations, pc)
elif args.method == 'graspnet':
    quaternions_final, translations_final, success = graspnet.refine_grasps_graspnet(quaternions, translations, pc)
elif args.method == 'metropolis':
    quaternions_final, translations_final, success = graspnet.refine_grasps_metropolis(quaternions, translations, pc)

## candidates are in 
candidate_grasp_filter = graspnet.info.filter_mask[-1,...]

# _idx = np.where(candidate_grasp_filter == 1)[0]
# fig, ax = plt.subplots(nrows=2)
# graspnet.info.plot_trans_gradients(ax=ax[0], idx=_idx)
# graspnet.info.plot_euler_gradients(ax=ax[1], idx=_idx)
# plt.show()


print("candidate_grasp_filter :", candidate_grasp_filter )
### STEP 5: visualize the grasps
# translations_final, quaternions_final, success = graspnet.info.get_refined_grasp()
# translations_final = denormalize_translation(graspnet.info.translations[-1,...], pc_mean.cpu().numpy())
#translations_final = denormalize_translation(translations_final, pc_mean.cpu().numpy())
# quaternions_final = graspnet.info.quaternions[-1, ...]
grasp_transforms_final = construct_grasp_matrix(quaternions_final, translations_final)
success = graspnet.info.success[-1, ...]

# # visualize the original grasps
# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc[0].cpu().numpy(), colors=[255,0,0,255]))
# for i, g in enumerate(grasp_transforms):
#     if candidate_grasp_filter[i] == 1:
#         scene.add_geometry(gripper_bd(success[i]), transform=g)
# scene.show()

# visualize the refined grasps
scene = trimesh.Scene()
scene.add_geometry(trimesh.points.PointCloud(pc[0].cpu().numpy(), colors=[255,0,0,255]))
for i, g in enumerate(grasp_transforms_final):
    if candidate_grasp_filter[i] == 1:
        scene.add_geometry(gripper_bd(success[i]), transform=g)
scene.show()




# ### STEP 6: save results

# init_translations = denormalize_translation(graspnet.info.translations[0,...], pc_mean.cpu().numpy())[candidate_grasp_filter==1]
# init_quaternions = graspnet.info.quaternions[0, ...][candidate_grasp_filter==1]

# final_translations = denormalize_translation(graspnet.info.translations[-1,...], pc_mean.cpu().numpy())[candidate_grasp_filter==1]
# final_quaternions = graspnet.info.quaternions[-1, ...][candidate_grasp_filter==1]

init_translations = denormalize_translation(graspnet.info.translations[0,...], pc_mean.cpu().numpy())
init_quaternions = graspnet.info.quaternions[0, ...]

final_translations = denormalize_translation(graspnet.info.translations[-1,...], pc_mean.cpu().numpy())
final_quaternions = graspnet.info.quaternions[-1, ...]

data_dir = f'experiments/experiment_real_test/grasps/{obj_name}'
Path(data_dir).mkdir(parents=True, exist_ok=True)
# np.savez(f'{data_dir}/grasps_{args.sampler}_initial',
#             translations = init_translations,
#             quaternions = init_quaternions)

            

# final grasps
np.savez(f'{data_dir}/grasps_{args.sampler}_final',
            translations = final_translations,
            quaternions = final_quaternions,
            success = success)

# with open(f'{data_dir}/info.pkl', 'wb') as fp:
#     pickle.dump(graspnet.info, fp)

# with open(f'{data_dir}/args.pkl', 'wb') as fp:
#     pickle.dump(args, fp)