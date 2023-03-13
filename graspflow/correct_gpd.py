from utils.visualization import gripper_bd
import numpy as np
import pickle
from utils.points import regularize_pc_point_count
from utils.auxilary import construct_grasp_matrix
from scipy.spatial.transform import Rotation as R
import glob
import re
from pathlib import Path

# THIS SCRIPT RUNS TWO THINGS:
# 1. CORRECT BADLY FORMATTED GRASP
# 2. SAVE THEM AS NPZ

# load gpd
# data_dir = '/home/tasbolat/some_python_examples/GRASP/gpd_sampled_grasps/box'

data_dir = '/home/tasbolat/some_python_examples/GRASP/gpd_sampled_grasps'
new_data_dir = '/home/tasbolat/some_python_examples/GRASP/gpd_sampled_grasps2'

# from numpy import genfromtxt, tri
# my_data = genfromtxt(f'{data_dir}/box014_gpd_grasps.csv', delimiter=',')

def proper_parse(txt):
    # is negative?
    coeff = 1.0
    if txt[0] == '-':
        coeff = -1.0

    txt = txt[1:]
    total_length = len(txt)
    if txt.find('-') == -1:
        a = total_length-txt[::-1].find('.')-2
    else:
        a = total_length-txt[::-1].find('.')-3
    return coeff*float(txt[:a]), float(txt[a:]) 

def convert_bad_csv_to_good(filename):
    all_csv = []
    with open(filename) as file:
        lines = file.readlines()
        for line in lines:
            splits = line.split(',')
            _csv = []
            for i, split in enumerate(splits):
                if i == 3:
                    a, b = proper_parse(split)
                    _csv.append(a)
                    _csv.append(b)
                else:
                    a = float(split)
                    _csv.append(a)

            all_csv.append(_csv)
    
    all_csv = np.array(all_csv)
    return all_csv


def move_backward_grasp(transform, standoff = 0.2):
    standoff_mat = np.eye(4)
    standoff_mat[2] = -standoff
    new = np.matmul(transform,standoff_mat)
    return new[:3,3]
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def normalize(x):
    low = np.min(x)-1
    high = np.max(x)+1
    return (x-low) / (high - low)
def gpd2grasps(rotations, translations, scores=None):
    '''
    rotations: [B,3,3]
    translations: [B,3]
    scores: [B]
    '''
    # create grasps
    grasps = np.eye(4)
    grasps = np.repeat(grasps, translations.shape[0], axis=1).reshape(4,4,-1)
    grasps = np.transpose(grasps, (2,0,1))
    # apply rotations
    r = R.from_matrix(rotations)
    r2 = R.from_quat([0,0.707,0,0.707])
    r3 = R.from_quat([0,0,0.707,0.707])
    grasps[:, :3,:3] = (r*r2*r3).as_matrix()
    # apply translations
    print(translations.shape)
    grasps[:,:3,3] = translations
    
    for i in range(grasps.shape[0]):
        grasps[i,:3,3] = move_backward_grasp(grasps[i], standoff=0.0725)
    if scores is not None:
        # scores = sigmoid(scores)
        scores = normalize(scores)
        return grasps, scores
    return grasps


def read_pc(cat, id, view_size=4):

    with open(f'data/pcs/{cat}/{cat}{id:03}.pkl', 'rb') as fp:
        data = pickle.load(fp)
        obj_pose_relative = data['obj_pose_relative']
        pcs = data['pcs']
    
    pc_indcs = np.random.randint(low=0, high=999, size=view_size)
    
    if len(pc_indcs) == 1:
        pc = pcs[pc_indcs[0]]
    else:
        __pcs = []
        for pc_index in pc_indcs:
            __pcs = __pcs + [pcs[pc_index]]
        pc = np.concatenate( __pcs )
   
    pc = regularize_pc_point_count(pc, 1024, False)
    return pc, obj_pose_relative



def process(filename, cat, idx):
    a = convert_bad_csv_to_good(filename)
    translations = a[:,1:4]
    rotations = a[:,4:].reshape([-1,3,3])
    score = a[:,0]

    grasps = gpd2grasps(rotations, translations)
    quaternions = R.from_matrix(grasps[:,:3,:3]).as_quat()
    translations = grasps[:,:3,3]

    _, obj_pose_relative = read_pc(cat, idx)

    data_dir2 = f'{new_data_dir}/{cat}/{cat}{idx:03}'
    Path(data_dir2).mkdir(parents=True, exist_ok=True)
    np.savez(f'{data_dir2}/grasps_initial',
                obj_pose_relative = obj_pose_relative,
                translations = translations,
                quaternions = quaternions)



ddirs = glob.glob(f"{data_dir}/*/*")

### DONE
for ddir in ddirs:
    print(ddir)
    cat = ddir.split('/')[-2]
    idx = int(re.findall(r'\d+', ddir.split('/')[-1])[0])
    process(ddir, cat, idx)

# all_trans = construct_grasp_matrix(quaternions, translations)

# import trimesh
# scene = trimesh.Scene()
# scene.add_geometry(trimesh.points.PointCloud(pc))
# for i in range(2):
#     scene.add_geometry(gripper_bd(), transform=grasps[i])
# scene.show()


