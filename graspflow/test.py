import torch
from networks.utils import get_control_point_tensor, get_surrogate_grasp_points, transform_gripper_pc_old
import trimesh
from utils.auxilary import PandaGripper
import numpy as np

import complex_environment_utils

cat = 'bla'
idx = 10
# obj_mesh = complex_environment_utils.load_shape_for_complex(cat=cat, idx=idx, path='../experiments/composites/shelf003')
obj_mesh = trimesh.load(f'../experiments/composites/shelf003/{cat}{idx:003}.obj', force='mesh')

print(obj_mesh.is_watertight)
scene = trimesh.Scene()
scene.add_geometry(obj_mesh)
scene.show()