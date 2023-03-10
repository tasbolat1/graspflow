

import torch
import numpy as np 
import argparse

from graspflow5 import GraspFlow
# from scipy.spatial.transform import Rotation as R
import pickle
from networks.utils import normalize_pc_and_translation, denormalize_translation
from pathlib import Path
import matplotlib.pyplot as plt
import complex_environment_utils
from utils.points import regularize_pc_point_count

from utils.auxilary import quaternions2eulers, eulers2quaternions, add_base_options, PandaGripper
import trimesh

dist_threshold = 0.000
base_displacement_extension = 0.1

p_left_center = torch.FloatTensor([-0.0531353785, 7.649999999998804e-06, 0.085390348])
half_extends_left = torch.FloatTensor([0.026536043000000002/2+dist_threshold, 0.0209743/2+dist_threshold, 0.053717304/2+dist_threshold])
p_left_transform = torch.eye(4)
p_left_transform[:3,3] = p_left_center

# right finger
p_right_center = torch.FloatTensor([0.0531353785, -7.649999999997069e-06, 0.085390348])
half_extends_right = torch.FloatTensor([0.026536043000000002/2+dist_threshold, 0.0209743/2+dist_threshold, 0.053717304/2+dist_threshold])
p_right_transform = torch.eye(4)
p_right_transform[:3,3] = p_right_center

# middle_between_fingers
p_middle_center = torch.FloatTensor([0.0, -7.649999999997069e-06, 0.085390348])
half_extends_middle = torch.FloatTensor([0.0796714215/2+dist_threshold, 0.0209743/2+dist_threshold, 0.053717304/2+dist_threshold])
p_middle_transform = torch.eye(4)
p_middle_transform[:3,3] = p_middle_center

box_middle = trimesh.primitives.Box(extents=half_extends_middle.numpy()*2, transform=p_middle_transform.numpy())
box_middle.visual.face_colors = [0,0,255,200]

# base
p_base_center = torch.FloatTensor([-0.0017819999999999989, 1.005000000000103e-05, 0.0200187-base_displacement_extension/2])
half_extends_base = torch.FloatTensor([0.204416/2+dist_threshold, 0.0632517/2+dist_threshold, (0.091887+base_displacement_extension)/2+dist_threshold])
p_base_transform = torch.eye(4)
p_base_transform[:3,3] = p_base_center

panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')

box_left = trimesh.primitives.Box(extents=half_extends_left.numpy()*2, transform=p_left_transform.numpy())
box_left.visual.face_colors = [0,255,0,100]
box_right = trimesh.primitives.Box(extents=half_extends_right.numpy()*2, transform=p_right_transform.numpy())
box_right.visual.face_colors = [0,255,0,100]
box_center = trimesh.primitives.Box(extents=half_extends_base.numpy()*2, transform=p_base_transform.numpy())
box_center.visual.face_colors = [0,255,0,100]


scene = trimesh.Scene()


for _mesh in panda_gripper.get_meshes():
    _mesh.visual.face_colors = [255,0,0,200]
    scene.add_geometry(_mesh)

scene.add_geometry(box_left)
scene.add_geometry(box_right)
scene.add_geometry(box_center)
scene.add_geometry(box_middle)

scene.show()