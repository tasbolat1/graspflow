import numpy as np
import torch
from graspflow_classifiers import N_classifier
import complex_environment_utils
from utils.points import regularize_pc_point_count
from utils.visualization import gripper_bd
from utils.auxilary import PandaGripper
from networks.utils import normalize_pc_and_translation

import trimesh
import trimesh.viewer
from trimesh.viewer.windowed import (SceneViewer,
                        render_scene)
from scipy.spatial.transform import Rotation as R
from theseus.geometry import SO3

# vars
cat = 'pan'
idx = 12

# load pcs
pc, pc_env, obj_trans, obj_quat, pc1, pc1_view, isaac_seed = complex_environment_utils.parse_isaac_complex_data(path_to_npz=f'../experiments/pointclouds/diner001.npz',
                                                               cat=cat, idx=idx, env_num=0,
                                                               filter_epsion=0.2)
pc = regularize_pc_point_count(pc, npoints=4096, use_farthest_point=True)
pc = np.expand_dims( regularize_pc_point_count(pc, npoints=1024), axis=0)
pc_mesh = trimesh.points.PointCloud(pc[0], colors=[0,0,0,100])
pc_env_mesh = trimesh.points.PointCloud(pc_env, colors=[0,255,0,100])

panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
panda_meshes = panda_gripper.get_meshes()
panda_mesh = trimesh.util.concatenate(panda_meshes)
panda_mesh.visual.face_colors = [125,125,125,80]

# setup evaluator
device = torch.device('cuda:0')
classifier = N_classifier(batch_size=5, device=device)

# setup scene
scene = trimesh.Scene()
scene.add_geometry(pc_mesh)
scene.add_geometry(pc_env_mesh)

holder_SO3 = SO3()
pc = torch.FloatTensor(pc).to(device)

def callback(callback_period):
    # try:
    f = open("grasp_N.txt", "r")
    txt = f.readlines()
    translation = [float(numeric_string) for numeric_string in txt[1][:-1].split(' ')] 
    euler = [float(numeric_string) for numeric_string in txt[3][:-1].split(' ')]
    grasp = np.eye(4)
    grasp[:3,:3] = R.from_euler(angles=euler, seq='xyz', degrees=True).as_matrix()
    grasp[:3, 3] = translation

    query = np.expand_dims(np.array(["handover"]), 0)

    with torch.no_grad():
        t = torch.FloatTensor(translation).unsqueeze(0).to(device)
        q = torch.FloatTensor(R.from_euler(angles=euler, seq='xyz', degrees=True).as_quat()).unsqueeze(0).to(device)
        _pc, _t, _pc_means = normalize_pc_and_translation(pc, t)
        r = SO3(quaternion=q).log_map()
        logit, scores = classifier.forward(_t, r, _pc, pc_mean=_pc, query=query, grasp_space='SO3')
        score = scores.cpu().numpy()[0]

    print(f'Score: {score}')
    scene.delete_geometry('gripper_bd')
    scene.delete_geometry('panda_gripper')
    scene.add_geometry(gripper_bd(score), transform=grasp, geom_name='gripper_bd')
    scene.add_geometry(panda_mesh, transform=grasp, geom_name='panda_gripper')
        
    # except:
    #     pass

    

SceneViewer(scene, callback=callback, callback_period=1)