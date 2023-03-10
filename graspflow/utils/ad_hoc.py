import pickle
import numpy as np
from .points import regularize_pc_point_count

def read_pc(cat, id, pc_path=f'/home/tasbolat/some_python_examples/GRASP/grasp_network/data/pcs', view_size=4):

    with open(f'{pc_path}/{cat}/{cat}{id:03}.pkl', 'rb') as fp:
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

def load_processed_pc(n, cat, idx, num_uniq_pcs=100, view_size=4):
    repeat_times = int(n/num_uniq_pcs)
    pcs = []
    for i in range(num_uniq_pcs):
        _pc, obj_pose_relative = read_pc(cat, idx, view_size=view_size)
        pc = np.expand_dims(_pc, axis=0)
        pc = np.repeat(pc, repeat_times, axis=0)
        pcs.append(pc)

    pc =  np.concatenate(pcs)
    assert pc.shape[0] == n, print(f'pc size does not match args.n {pc.shape[0]} neq {n}')

    return pc, obj_pose_relative

def load_grasps(cat, idx, grasps_path=f"/home/tasbolat/some_python_examples/refinement_experiments/grasps_generated_graspnet"):
    data_filename = f"{grasps_path}/{cat}{idx:03}.npz"
    data = np.load(data_filename)
    translations = data["translations"]
    quaternions = data["quaternions"]
    return translations, quaternions