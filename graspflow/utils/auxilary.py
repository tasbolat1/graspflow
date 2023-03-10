import json
from secrets import choice
from tkinter import E
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh.transformations as tra
from pathlib import Path
from tqdm import tqdm


class InfoHolder():
    def __init__(self, var_names=[]):
        '''
        Holds all necessary information to debug the refinement methods. must be used alongside with args.
        '''

        self.var_names = var_names
        self.exec_time = 0

        # register new variables in data holder
        self.data = {} 
        for name in self.var_names:
            self.data[name]=[]

    def batch_init(self):
        self.data_batch = {}
        for name in self.var_names:
            self.data_batch[name]=[]

    def batch_update(self, **kwargs):
        for var in kwargs:
            if torch.is_tensor(kwargs[var]):
                self.data_batch[var].append(kwargs[var].detach().cpu().numpy().copy())
            else:
                self.data_batch[var].append(kwargs[var])

    def update(self):
        for var in self.data.keys():
            self.data[var].append(np.stack(self.data_batch[var]))

    def conclude(self, exec_time):

        for var in self.data.keys():
            self.data[var] = np.concatenate(self.data[var], axis=1)

        self.exec_time = exec_time

        self.batch_init()

    def clear(self):
        self.__init__()
        self.var_names = []
        self.exec_time = 0
        self.batch_init()

    def save(self, save_dir=None):
        np.savez(save_dir, exec_time=self.exec_time, **self.data)

    def load(self, file_dir=None):
        '''
        Overrides all values
        '''
        
        self.clear()
        
        data=dict(np.load(file_dir))
        self.exec_time = data.pop('exec_time', None)
        self.data = data

        # initialize variables
        for name in self.data.keys():
            self.var_names.append(name)

        self.batch_init()

def get_object_mesh(category, idx):
    extension = 'obj'
    if category in ['box', 'cylinder']:
        extension = 'stl'
    obj_filename = f'../grasper/grasp_data/meshes/{category}/{category}{idx:03}.{extension}'
    metadata_filename = f'../grasper/grasp_data/info/{category}/{category}{idx:03}.json'
    metadata = json.load(open(metadata_filename,'r'))
    mesh = trimesh.load(obj_filename)
    mesh.apply_scale(metadata['scale'])

    return mesh

def get_transform(quat, trans):
    if torch.is_tensor(quat):
        quat = quat.detach().cpu().numpy()
    if torch.is_tensor(trans):
        trans = trans.detach().cpu().numpy()

    transform = np.eye(4)
    transform[:3,:3] = R.from_quat(quat).as_matrix()
    transform[:3,3] = trans
    return transform


# only for box!
def sample_grasps(N=10, label_type=0 , dataset=None):
    size = len(dataset)
    if label_type == 0:
        indcs = np.random.randint(low=int(size/9), high=int(size), size=N)
    elif label_type == 1:
        indcs = np.random.randint(low=0, high=int(size/9), size=N)
    else:
        indcs = np.random.randint(low=0, high=int(size), size=N)

    trans = []
    quats = []
    labels = []
    for i in range(N):
        q,t,pc,l,_ = dataset[indcs[i]]
        quats.append(q)
        trans.append(t)
        labels.append(l)
    
    quats = torch.stack(quats)
    trans = torch.stack(trans)
    labels = torch.stack(labels)

    return quats, trans, labels, pc


def quaternions2eulers(quaternions, seq='XYZ'):
    '''
    quaternions: [B,4] (torch tensor)
    NO BACKWARD!
    '''
    r = R.from_quat(quaternions.cpu().numpy())
    eulers = torch.FloatTensor(r.as_euler(seq=seq).copy()).to(quaternions.device)
    return eulers

def eulers2quaternions(eulers, seq='XYZ'):
    '''
    eulers: [N,B,3] or [B,3] (numpy)
    NO BACKWARD!
    '''

    if isinstance(eulers, torch.Tensor):
        eulers=eulers.detach().cpu().numpy()

    if len(eulers.shape) == 2:
        r = R.from_euler(angles = eulers, seq=seq)
        quaternions = r.as_quat()
    else:
        T, B, _ = eulers.shape
        eulers = eulers.reshape(-1, 3)
        r = R.from_euler(angles = eulers, seq=seq)
        quaternions = r.as_quat().reshape(T, B, 4)
    return quaternions


def construct_grasp_matrix(quaternion, translation):
    '''
    Input:
    quaternion: [B,4]
    translations: [B,3]
    Return:
    grasps: [B,4,4]
    '''
    
    if torch.is_tensor(quaternion):
        quaternion = quaternion.detach().cpu().numpy()
    if torch.is_tensor(translation):
        translation = translation.detach().cpu().numpy()

    squeezed = False
    if len(quaternion.shape) == 1:
        quaternion = np.expand_dims(quaternion, 0)
        translation = np.expand_dims(translation, 0)
        squeezed=True
    
    grasps = np.repeat(np.expand_dims(np.eye(4), axis=0), quaternion.shape[0], axis=0)
    grasps[:, :3, :3] = R.from_quat(quaternion).as_matrix()
    grasps[:, :3, 3] = translation

    if squeezed:
        np.squeeze(grasps, axis=0)

    return grasps
        

# define gripper
class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, num_contact_points_per_finger=10, root_folder='', num_get_distance_rays=20):
        """Create a Franka Panda parallel-yaw gripper object.
        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
            face_color {list of 4 int} (optional) -- RGBA, make A less than 255 to have transparent mehs visualisation
        """
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04
        self.num_contact_points_per_finger = num_contact_points_per_finger
        self.num_get_distance_rays = num_get_distance_rays

        if q is None:
            q = self.default_pregrasp_configuration

        self.q = q

        self.base = trimesh.load(Path(root_folder)/'assets/urdf_files/meshes/collision/hand.obj')
        self.base.metadata['name'] = 'base'
        self.finger_left = trimesh.load(Path(root_folder)/'assets/urdf_files/meshes/collision/finger.obj')
        self.finger_left.metadata['name'] = 'finger_left'
        self.finger_right = self.finger_left.copy()
        self.finger_right.metadata['name'] = 'finger_right'

        # transform fingers relative to the base
        self.finger_left.apply_transform(tra.euler_matrix(0, 0, np.pi))
        self.finger_left.apply_translation([0, -q, 0.0584])  # moves relative to y
        self.finger_right.apply_translation([0, +q, 0.0584])

        self.fingers = trimesh.util.concatenate([self.finger_left, self.finger_right])
        self.hand = trimesh.util.concatenate([self.fingers, self.base])

        # this makes to rotate the gripper to match with real world
        self.apply_transformation(tra.euler_matrix(0, 0, -np.pi/2))        

    def apply_transformation(self, transform):
        #transform = transform.dot(tra.euler_matrix(0, 0, -np.pi/2))
        # applies relative to the latest transform
        self.finger_left.apply_transform(transform)
        self.finger_right.apply_transform(transform)
        self.base.apply_transform(transform)
        self.fingers.apply_transform(transform)
        self.hand.apply_transform(transform)

    def get_obbs(self):
        """Get list of obstacle meshes.
        Returns:
            list of trimesh -- bounding boxes used for collision checking
        """
        return [self.finger_left.bounding_box, self.finger_right.bounding_box, self.base.bounding_box]

    def get_combined_obbs(self):
        return trimesh.util.concatenate(self.get_obbs())

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.
        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_left, self.finger_right, self.base]

    def get_bb(self, all=False):
        if all:
            return trimesh.util.concatenate(self.get_meshes()).bounding_box
        return trimesh.util.concatenate(self.get_meshes())

def tensor_nans_like(x):
    return torch.ones_like(x)*torch.tensor(float('nan'))
    

def add_base_options(parser):
    parser.add_argument("--max_iterations", type=int, help="Maximum iterations to refine samples.", default=10)
    parser.add_argument("--batch_size", type=int, help="Batch size.", default=64)
    parser.add_argument("--method", type=str, choices=['GraspFlow', 'graspnet', 'metropolis', 'GraspOpt', 'GraspOptES'], help="Method for refinement.", default='GraspFlow')
    parser.add_argument("--sampler", type=str, help="Sampling methods.", choices=['graspnet', 'gpd'], default='graspnet')
    
    parser.add_argument("--device", type=int, help="device index. Pass -1 for cpu.", default=-1)
    parser.add_argument("--seed", type=int, help="Seed for randomness.", default=40)
    parser.add_argument("--grasp_space", type=str, help="Space in which grasp update happens.", choices=["SO3", "Euler", "Theta"], default="SO3")
    # parser.add_argument('--include_robot', action='store_true', help='If set, includes E classifier. Only works with grasp_space Theta', default=False)
    # parser.add_argument('--no-include_robot', dest='include_robot', action='store_false')
    # parser.set_defaults(include_robot=False)
    # parser.add_argument('--include_robot', type=int, help='If set, includes E classifier. Only works with grasp_space Theta', default=0)

    parser.add_argument('--classifier', type=str, help='Set of classifiers to refife', default=0)

    # SO3 and Euler
    # parser.add_argument("--noise_e", type=float, help="Noise factor for GraspFlow.", default=0.000000)
    # parser.add_argument("--noise_t", type=float, help="Noise factor for DFflow.", default=0.000000)
    # parser.add_argument("--eta_t", type=float, help="Refiement rate for DFflow.", default=0.000025)
    # parser.add_argument("--eta_e", type=float, help="Refiement rate for DFflow.", default=0.000025)

    # parser.add_argument("--eta_table_t", type=float, help="Refiement rate for DFflow.", default=0.000025)
    # parser.add_argument("--eta_table_e", type=float, help="Refiement rate for DFflow.", default=0.000025)

    # Theta
    # parser.add_argument("--eta_theta_s", type=float, help="Refinement rate for S classifier for Theta grasp space.", default=0.000025)
    # parser.add_argument("--eta_theta_e", type=float, help="Refiement rate for E classifier for Theta grasp space.", default=0.000025)
    # parser.add_argument("--noise_theta", type=float, help="Noise factor for DFflow with Theta grasp space.", default=0.000001)

    # Robot
    # parser.add_argument("--robot_threshold", type=float, help="MI score threshold.", default=0.01)
    # parser.add_argument("--robot_coeff", type=float, help="MI score coef.", default=10000)

    return parser