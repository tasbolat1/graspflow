import numpy as np
import trimesh
from utils.points import regularize_pc_point_count
from scipy.spatial.transform import Rotation as R
from utils.visualization import gripper_bd
from utils.auxilary import PandaGripper


complex_items = {
    'shelf006': {
    'bottle000': 1,
    'pan012': 2,
    'shelf001': 3,
    'bottle014': 4,
    'bowl008': 5,
    'bowl010': 6,
    'bowl016': 7,
    'fork006': 8,
    'mug002': 9,
    'mug008': 10,
    'pan006': 11,
    'scissor007': 12,
    'affix001': 13,
    'table000': 100
    },

    'shelf008': {
    'bottle000': 1,
    'pan012': 2,
    'shelf001': 3,
    'bottle014': 4,
    'bowl008': 5,
    'bowl010': 6,
    'bowl016': 7,
    'fork006': 8,
    'mug002': 9,
    'mug008': 10,
    'pan006': 11,
    'scissor007': 12,
    'affix001': 13,
    'table000': 100
    },

    'shelf009': {
    'bottle000': 1,
    'pan012': 2,
    'shelf001': 3,
    'bottle014': 4,
    'bowl008': 5,
    'bowl010': 6,
    'bowl016': 7,
    'fork006': 8,
    'mug002': 9,
    'mug008': 10,
    'pan006': 11,
    'scissor007': 12,
    'affix001': 13,
    'table000': 100
    },

    'diner001': {
        "bottle000": 1,
        "bowl008": 2,
        "bowl009": 3,
        "fork006": 4,
        "mug002": 5,
        "pan012": 6,
        "spatula014": 7,
        "affix001": 8,
        "shelf003": 9
    }
}

def get_transform(quat, trans):
    '''
    Converts translation and quaternion into transform
    '''
    transform = np.eye(4)
    transform[:3,:3] = R.from_quat(quat).as_matrix()
    transform[:3,3] = trans
    return transform

def depth_to_pointcloud(depth):
        '''
        Converts depth to pointcloud.
        Arguments:
            * ``depth`` ((W,H) ``float``): Depth data.
        Returns:
            * pc ((3,N) ``float``): Pointcloud.
        W        '''
        fov = 75.0 * np.pi/180
        width = height = 512
        fy = fx = 0.5 / np.tan(fov * 0.5) # aspectRatio is one.
 
        depth = np.nan_to_num(depth, posinf=0, neginf=0, nan=0)
        mask = np.where(depth > 0)
        x = mask[1]
        y = mask[0]

        normalized_x = (x.astype(np.float32) - width * 0.5) / width
        normalized_y = (y.astype(np.float32) - height * 0.5) / height

        world_x = normalized_x * depth[y, x] / fx
        world_y = normalized_y * depth[y, x] / fy
        world_z = depth[y, x]
        pc = np.vstack((world_x, world_y, world_z)).T

        return pc

def isaac_cam2world(pc, view):
    '''
    Converts pointcloud from camera view to world frame
    '''
    view_rotmat_pre = get_transform(R.from_euler('zyx', [0, np.pi/2, -np.pi/2]).as_quat(), np.array([0,0,0]))
    pc = np.hstack([pc, np.expand_dims(np.ones(pc.shape[0]), axis=1)]) # Nx4 w.r.t CAMERA
    pc_world = np.matmul(view_rotmat_pre, pc.T).T 
    pc_world = np.matmul(view, pc_world.T).T 
    return pc_world[:,:3]


def world2isaac_cam(pc, view):
    '''
    Converts pointcloud from camera view to world frame
    '''
    view_rotmat_pre = get_transform(R.from_euler('zyx', [0, np.pi/2, -np.pi/2]).as_quat(), np.array([0,0,0]))
    pc = np.hstack([pc, np.expand_dims(np.ones(pc.shape[0]), axis=1)]) # Nx4 w.r.t CAMERA
    pc_world = np.matmul(np.linalg.inv(view), pc.T).T 
    pc_world = np.matmul(np.linalg.inv(view_rotmat_pre), pc_world.T).T 
    return pc_world[:,:3]


def filter_pc_by_distance(pc, axis=0, lim=[-np.inf, np.inf]):
    '''
    Filters pointclouds by distance
    '''
    mask = (pc[:,int(axis)] >= lim[0]) & (pc[:,int(axis)] <= lim[1])
    return pc[mask]

def get_obj_pose(path_to_npz='../experiments/pointclouds/complex.npz', obj=None, env_num = 0):
    # read data
    data = np.load(path_to_npz)

    seg_labels = complex_items[path_to_npz.split('/')[-1]]

    if obj not in seg_labels:
        raise f"Object {obj} is not in environment!"

    return data['obj_stable_translations'][seg_labels[obj][0]-1], data['obj_stable_quaternions'][seg_labels[obj][0]-1]

def parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/complex.npz',
                             cat='scissor', idx=7,
                             env_num = 0, filter_epsion=0.075):

    # read data
    data = np.load(path_to_npz)

    isaac_seed = data['seed']
    

    total_view = 18 # change to 17 for 006, 18 for diner001

    pcs = []
    pcs_env = []
    obj = f'{cat}{idx:003}'

    seg_labels = complex_items[path_to_npz.split('/')[-1][:-4]]

    if obj not in seg_labels:
        raise f"Object {obj} is not in environment!"


    pc1_combined = {}
    for custom_obj in seg_labels.keys():
        pc1_combined[custom_obj] = []

    for view_id in range(total_view):

        # extract imgs
        depth = data[f'depth_{view_id}_{env_num}'] # [512,512]
        segment = data[f'segment_{view_id}_{env_num}'] # [512, 512]
        view_trans = data[f'view_pos_{view_id}'] # (4,)
        view_rots = data[f'view_rot_{view_id}'] # (3,)



        # get segments
        depth_segmented_obj = np.where(segment == seg_labels[obj], depth, np.inf)
        depth_segmented_env = np.where(( (segment > 0)  ) & (segment != seg_labels[obj]), depth, np.inf)
        # depth_segmented_env = np.where(( ( (segment > 0)  ) & (segment != seg_labels[obj]) & ((segment < 100) ) ), depth, np.inf)


        # convert to pc
        pc = depth_to_pointcloud(depth_segmented_obj)
        pc_env = depth_to_pointcloud(depth_segmented_env)

        # get custom views combined to the camera_view
        _pc = np.copy(pc)
        _pc1_view1 = get_transform(quat=view_rots, trans=view_trans)
        _pc = isaac_cam2world(_pc, _pc1_view1)
        _pc1_view2 = get_transform( data[f'view_rot_11'], trans=data[f'view_pos_11'])
        _pc = world2isaac_cam(_pc, _pc1_view2)
        pc1_combined[obj].append(_pc)
        

        # get pc r.t camera frame
        if view_id == 11:
            pc1_view = get_transform(quat=view_rots, trans=view_trans)



        # map to world frame
        pc = isaac_cam2world(pc, get_transform(quat=view_rots, trans=view_trans))
        pc_env = isaac_cam2world(pc_env, view=get_transform(quat=view_rots, trans=view_trans))

        # NOTE: no pc_env is considered if PC is empty
        if pc.shape[0] > 0:
            pc_mean = pc.mean(axis=0)
            for i in range(3):
                pc_env = filter_pc_by_distance(pc_env, axis=i, lim=[pc_mean[i]-filter_epsion,pc_mean[i]+filter_epsion])
        else:
            pc_env = np.array([])
        
        # exclude empty ones
        if pc.shape[0] > 0:
            pcs.append(pc)
        if pc_env.shape[0] > 0:
            pcs_env.append(pc_env)

    pc = np.concatenate(pcs, axis=0)
    pc_env = np.concatenate(pcs_env, axis=0)

    pc1 = np.concatenate( pc1_combined[obj], axis=0)


    return pc, pc_env, data['obj_stable_translations'][seg_labels[obj]-1], data['obj_stable_quaternions'][seg_labels[obj]-1], pc1, pc1_view, isaac_seed

def load_shape_for_complex(cat, idx, path='../experiments/composites/shelf006'):
    '''
    Loads mesh from complex environment
    '''
    f=f'{path}/{cat}{idx:003}.stl'
    # mesh = trimesh.load(f, force='mesh')
    mesh = trimesh.load(f)#, force='mesh')
    return mesh


if __name__ == "__main__":

    cat = 'scissor'
    idx = 7
    pc, pc_env, obj_trans, obj_quat, pc1, pc1_view, isaac_seed = parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/shelf008.npz',
                                                               cat=cat, idx=idx, env_num=0,
                                                               filter_epsion=1.0)
    obj_mesh = load_shape_for_complex(cat=cat, idx=idx, path='../experiments/composites/shelf008')
    obj_pose = get_transform(obj_quat, obj_trans)

    # pc_env = regularize_pc_point_count(pc_env, npoints=10000)
    pc = regularize_pc_point_count(pc, npoints=1024, use_farthest_point=True)

    pc_mesh = trimesh.points.PointCloud(pc, colors=[255, 0, 0, 40])
    pc_env = trimesh.points.PointCloud(pc_env, colors=[0, 255, 0, 10])
    scene = trimesh.Scene()

    scene.add_geometry(obj_mesh, transform=obj_pose)
    scene.add_geometry(pc_mesh)
    scene.add_geometry(pc_env)
    

    # scene.show()

    # for cat in seg_labels.keys():
    #     pc, pc_env, obj_trans, obj_quat,  pc1, pc1_view, isaac_seed = parse_isaac_complex_data(path_to_npz='../experiments/pointclouds/shelf001.npz', cat=cat[:-3], idx=int(cat[-3:]), env_num=0)
        

    #     pc_mesh = trimesh.points.PointCloud(pc, colors=[255, 0, 0, 50])
        
    #     # pc = regularize_pc_point_count(pc, npoints=100)

    #     obj_mesh = load_shape_for_complex(cat=cat[:-3], idx=int(cat[-3:]))
    #     obj_pose = get_transform(obj_quat, obj_trans)

    #     scene.add_geometry(obj_mesh, transform=obj_pose)
    #     # scene.add_geometry(obj_mesh)

    #     scene.add_geometry(pc_mesh)
    # # scene.add_geometry(pc_env)

    # scene.show()


    trans = np.array([0.45,0.25,0.3])
    rotmat = R.from_euler(angles=[-90,0,50], seq='xyz', degrees=True).as_matrix()
    transform = np.eye(4)
    transform[:3,:3] = rotmat
    transform[:3,3] = trans

    panda_gripper = PandaGripper(root_folder='/home/tasbolat/some_python_examples/graspflow_models/grasper')
    panda_meshes = panda_gripper.get_meshes()
    panda_mesh = trimesh.util.concatenate(panda_meshes)
    panda_mesh.visual.face_colors = [125,125,125,80]

    scene.add_geometry(gripper_bd(), transform=transform, geom_name='gripper_bd')
    scene.add_geometry(panda_mesh, transform=transform, geom_name='panda_gripper')
    
    import trimesh.viewer
    from trimesh.viewer.windowed import (SceneViewer,
                           render_scene)

    def callback(callback_period):
        try:
            f = open("grasp.txt", "r")
            txt = f.readlines()
            translation = [float(numeric_string) for numeric_string in txt[1][:-1].split(' ')] 
            euler = [float(numeric_string) for numeric_string in txt[3][:-1].split(' ')]
            grasp = np.eye(4)
            grasp[:3,:3] = R.from_euler(angles=euler, seq='xyz', degrees=True).as_matrix()
            grasp[:3, 3] = translation

            scene.delete_geometry('gripper_bd')
            scene.delete_geometry('panda_gripper')
            scene.add_geometry(gripper_bd(), transform=grasp, geom_name='gripper_bd')
            scene.add_geometry(panda_mesh, transform=grasp, geom_name='panda_gripper')

            
        except:
            pass
        # scene.show()

        

    SceneViewer(scene, callback=callback, callback_period=1/5)

    # scene.show()

    # from dynamic_grasp_scene import GraspVisualizer
    # visualizer_app = GraspVisualizer()
    # # visualizer_app.add_scene(scene)
    # visualizer_app.run()

    

    
