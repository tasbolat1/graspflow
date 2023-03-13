import json
import trimesh
from scipy.spatial.transform import Rotation as R
import numpy as np
import torch
import matplotlib.pyplot as plt
import trimesh.transformations as tra
from pathlib import Path
from tqdm import tqdm

class InfoHolder():
    def __init__(self):
        # global_variables
        self.rots = []
        self.translations = []
        self.rots_grad = []
        self.translations_grad = []
        self.success = []
        self.time = 0
        self.quaternions = []
        self.filter_mask = []

    def update(self):
        self.rots.append(np.stack(self.batch_rots))
        self.translations.append(np.stack(self.batch_translations))
        if len(self.batch_rots_grad) != 0:
            self.rots_grad.append(np.stack(self.batch_rots_grad))
            self.translations_grad.append(np.stack(self.batch_translations_grad))
        self.success.append(np.stack(self.batch_success))
        self.quaternions.append(np.stack(self.batch_quats))
        self.filter_mask.append(np.stack(self.batch_filter_mask))

    def batch_init(self):
        # batch_variables
        self.batch_rots = []
        self.batch_translations = []
        self.batch_translations_grad = []
        self.batch_rots_grad = []
        self.batch_success = []
        self.batch_quats = []
        self.batch_filter_mask = []

    def batch_update(self, _rots, _rots_grad, _translations, _translations_grad, success, filter_mask, rot_reps='euler'):
        # record batch info

        if isinstance(_rots, torch.Tensor):
            if rot_reps == 'euler':
                self.batch_quats.append(eulers2quaternions(_rots.detach().cpu().numpy()))
            elif rot_reps == 'quaternion':
                _rots = torch.nn.functional.normalize(_rots, p=2) # normalize
                self.batch_quats.append(_rots.detach().cpu().numpy())
            else:
                #self.batch_quats.append(utils.A_vec_to_quat(_rots).detach().cpu().numpy())
                ValueError('Not Implemented!')

            if _rots is not None:
                self.batch_rots.append(_rots.detach().cpu().numpy())
            if _rots_grad is not None:
                self.batch_rots_grad.append(_rots_grad.detach().cpu().numpy())
            if _translations is not None:
                self.batch_translations.append(_translations.detach().cpu().numpy())
            if _translations_grad is not None:
                self.batch_translations_grad.append(_translations_grad.detach().cpu().numpy())
            if success is not None:
                self.batch_success.append(success.detach().cpu().numpy())
            if filter_mask is not None:
                self.batch_filter_mask.append(filter_mask.detach().cpu().numpy())
        else:
            raise ValueError('Not implemented for numpy input.')

    def conclude(self, exec_time):
        self.rots = np.concatenate(self.rots, axis=1)
        self.translations = np.concatenate(self.translations, axis=1)
        if len(self.rots_grad) != 0:
            self.rots_grad = np.concatenate(self.rots_grad, axis=1)
            self.translations_grad = np.concatenate(self.translations_grad, axis=1)
        self.success = np.concatenate(self.success, axis=1).squeeze(-1)
        self.filter_mask = np.concatenate(self.filter_mask, axis=1)
        self.time = exec_time
        self.quaternions = np.concatenate(self.quaternions, axis=1)

    def get_refined_grasp(self):
        # retrieve last grasp from [seq, B, :]
        return self.quaternions[-1, ...], self.translations[-1,...], self.success[-1, ...]

    def compute_init_success(self, filtered=True):
        if filtered:
            mask = self.filter_mask[0] == 1
            return np.mean(self.success[0][mask])
        return np.mean(self.success[0])

    def compute_final_success(self, filtered=True):
        if filtered:
            mask = self.filter_mask[-1] == 1
            return np.mean(self.success[-1][mask])
        return np.mean(self.success[0])

    def draw_trajectory(self, ax, idx=None):
        pass

    def plot_euler_gradients(self, ax, idx=None):
        # repr [seq, B, :]

        if idx == 'all':
            ax.plot(self.rots_grad[:, :, 0], '-', label='alpha')
            ax.plot(self.rots_grad[:, :, 1], '--', label='beta')
            ax.plot(self.rots_grad[:, :, 2], '-.-', label='gamma')
        else:
            ax.plot(self.rots_grad[:, idx, 0], '-', label='alpha')
            ax.plot(self.rots_grad[:, idx, 1], '--', label='beta')
            ax.plot(self.rots_grad[:, idx, 2], '-.', label='gamma')

        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())


    def plot_trans_gradients(self, ax, idx=None):
        # repr [seq, B, :]

        if idx == 'all':
            ax.plot(self.translations_grad[:, :, 0], '-', label='x')
            ax.plot(self.translations_grad[:, :, 1], '--', label='y')
            ax.plot(self.translations_grad[:, :, 2], '-.-', label='z')
        else:
            ax.plot(self.translations_grad[:, idx, 0], '-', label='x')
            ax.plot(self.translations_grad[:, idx, 1], '--', label='y')
            ax.plot(self.translations_grad[:, idx, 2], '-.', label='z')

        handles, labels = ax.get_legend_handles_labels() #plt.gca()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())
        

def get_object_mesh(category, i):
    extension = 'obj'
    if category in ['box', 'cylinder']:
        extension = 'stl'
    obj_filename = f'../grasper/grasp_data/meshes/{category}/{category}{i:03}.{extension}'
    metadata_filename = f'../grasper/grasp_data/info/{category}/{category}{i:03}.json'
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


def quaternions2eulers(quaternions, seq='xyz'):
    '''
    quaternions: [B,4] (torch tensor)
    NO BACWARD!
    '''
    r = R.from_quat(quaternions.cpu().numpy())
    eulers = torch.FloatTensor(r.as_euler(seq=seq).copy()).to(quaternions.device)
    return eulers

def eulers2quaternions(eulers, seq='xyz'):
    '''
    eulers: [N,B,3] or [B,3] (numpy)
    NO BACWARD!
    '''
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
