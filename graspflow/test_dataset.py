import numpy as np
import torch
from datasets import GraspDatasetWithTight as GraspDataset
from utils.visualization import gripper_bd
from utils.auxilary import get_transform
import json
from networks.utils import transform_gripper_pc_old, transform_gripper_pc
import pickle
import trimesh
from utils.points import regularize_pc_point_count


cat = 'mug'
obj_id = 2
# load specific item

def load_obj_mesh(cat, obj_id):
    obj = trimesh.load(f'../grasper/grasp_data/meshes/{cat}/{cat}{obj_id:03}.obj', force='mesh')
    info = json.load(open(f'../grasper/grasp_data/info/{cat}/{cat}{obj_id:03}.json', 'r'))
    obj.apply_scale(info['scale'])
    obj_mesh_mean = np.mean(obj.vertices, 0)
    obj.vertices -= np.expand_dims(obj_mesh_mean, 0)
    return obj


dataset = GraspDataset(path_to_grasps = 'data/grasps_lower/preprocessed',
                             path_to_grasps_tight='data/grasps_tight_lower/preprocessed',
                             path_to_pc='data/pcs', split='test', augment=True, mode=1, full_pc=False,
                             allowed_categories=['mug', 'box', 'bottle', 'cylinder', 'bowl'])

# cats = np.array(dataset.categories)
# # print(np.sum(cats == cat))
# mask_idcs = np.where((cats == cat) & (dataset.metadata == obj_id))[0]
# print(len(cats))
# print(len(dataset.quaternions))
# print(len(dataset.metadata))
# chosen_idcs = np.random.choice(mask_idcs, size=20)
# scene = trimesh.Scene()
# scene.add_geometry(load_obj_mesh(cat, obj_id))

# pc = dataset.pcs[cat][obj_id][:50]
# pc = np.concatenate(pc)
# print(pc.shape)
# pc = regularize_pc_point_count(pc, 1024)
# scene.add_geometry(trimesh.points.PointCloud(pc))
# for idx in chosen_idcs:
#     q = dataset.quaternions[idx]
#     t = dataset.translations[idx]
#     l = dataset.labels[idx]
#     scene.add_geometry(gripper_bd(l), transform=get_transform(q,t))

# scene.show()

q,t,pc1,l,cat, obj_id = dataset[101]
scene = trimesh.Scene()

scene.add_geometry(trimesh.points.PointCloud(pc1))
scene.add_geometry(load_obj_mesh(cat, obj_id))



scene.add_geometry(gripper_bd(l), transform=get_transform(q,t))
grasp_pc = transform_gripper_pc_old(q.unsqueeze(0),t.unsqueeze(0))

scene.add_geometry(trimesh.points.PointCloud(grasp_pc.squeeze(0).numpy()))


# with open(f'data/pcs/{cat}/{cat}{obj_id:03}.pkl', 'rb') as fp:
#     _pc = pickle.load(fp)[0]

# print(_pc.shape)
# scene.add_geometry(trimesh.points.PointCloud(_pc))

# scene.add_geometry(trimesh.points.PointCloud(pc2))
# scene.add_geometry(trimesh.points.PointCloud(pc3))
# scene.add_geometry(trimesh.points.PointCloud(pc4))
# scene.add_geometry(trimesh.points.PointCloud(pc5))

scene.show()