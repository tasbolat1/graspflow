import numpy as np
import trimesh
from pathlib import Path
import trimesh.transformations as tra
import time
import torch
import torch.nn as nn

from scipy.spatial.transform import Rotation as R

# define gripper
class PandaGripper(object):
    """An object representing a Franka Panda gripper."""

    def __init__(self, q=None, root_folder=''):
        """Create a Franka Panda parallel-yaw gripper object.
        Keyword Arguments:
            q {list of int} -- configuration (default: {None})
            num_contact_points_per_finger {int} -- contact points per finger (default: {10})
            root_folder {str} -- base folder for model files (default: {''})
            face_color {list of 4 int} (optional) -- RGBA, make A less than 255 to have transparent mehs visualisation
        """
        self.joint_limits = [0.0, 0.04]
        self.default_pregrasp_configuration = 0.04

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

    def get_meshes(self):
        """Get list of meshes that this gripper consists of.
        Returns:
            list of trimesh -- visual meshes
        """
        return [self.finger_left, self.finger_right, self.base]

    def get_bb(self, all=False):
        if all:
            return trimesh.util.concatenate(self.get_meshes()).bounding_box
        return trimesh.util.concatenate(self.get_obbs())

from networks import utils
from networks import quaternion

class CollisionDistanceEvaluator(nn.Module):
    def __init__(self, npoints=10, dist_threshold=0.001, dist_coeff=10000):
        super(CollisionDistanceEvaluator, self).__init__()
        '''
        Evaluator on collision with point cloud.
        Very naive implementation. Relies on ideal data. Please clean pc very good before using this module.
        '''
        self.npoints = npoints
        self.dist_threshold = dist_threshold
        self.dist_coeff = dist_coeff

    def forward(self, trans, quat, pc):
        '''
        trans [B,3]
        quat [B,4]
        pc [B,n,3]
        '''

        # rotate and translate point clouds to inverse transformation
        quat_inv = quaternion.quat_inv(quat)
        quat_inv = quat_inv.unsqueeze(1).repeat([1, pc.shape[1], 1])
        pc = quaternion.rot_p_by_quaterion(pc, quat_inv)
        pc = pc - trans.unsqueeze(1).repeat([1,pc.shape[1],1])

        return self.evaluate_grasp(pc)

    def forward_with_eulers(self, trans, eulers, pc):
        # rotate and translate point clouds to inverse transformation
        rotmat = quaternion.euler2matrix(eulers, order='XYZ') # [B,3,3]
        pc = torch.bmm(pc,rotmat) # TODO: NEED TO BE CHECKED
        pc = pc - trans.unsqueeze(1).repeat([1,pc.shape[1],1])

        return self.evaluate_grasp(pc)

    def evaluate_grasp(self, pc):

        B = pc.shape[0]

        # big bounding box of the grasp
        p_all_center = torch.FloatTensor([-1.78200e-03,1.00500e-05,4.31621e-02]).to(pc.device)
        half_extends_all = torch.FloatTensor([0.204416/2+self.dist_threshold, 0.0632517/2+self.dist_threshold, 0.1381738/2+self.dist_threshold]).to(pc.device)

        mask = self.bb_aligned(pc=pc, p=p_all_center, half_extents=half_extends_all)

        dist = torch.ones([B,1]).to(pc.device)
        for i in range(B):
            inside_pc = pc[i,mask[i],:]
            if inside_pc.shape[0] > 0:
                dist[i,0] = -1*torch.mean(torch.linalg.norm(pc[i,mask[i],:], dim=1))

        logit = dist*self.dist_coeff

        return logit

    def bb_aligned(self, p=torch.FloatTensor([-1.78200e-03,1.00500e-05,4.31621e-02]), half_extents=torch.FloatTensor([0.204416, 0.0632517, 0.1381738]), pc=None):
        '''
        Checks if point is inside bb
        pc [B,1024,3]
        '''

        bb_x_lims = [p[0] - half_extents[0], p[0] + half_extents[0]]
        bb_y_lims = [p[1] - half_extents[1], p[1] + half_extents[1]]
        bb_z_lims = [p[2] - half_extents[2], p[2] + half_extents[2]]

        # check half spaces
        mask1 = pc[...,0] >= bb_x_lims[0]
        mask2 = pc[...,0] <= bb_x_lims[1]
        mask3 = pc[...,1] >= bb_y_lims[0]
        mask4 = pc[...,1] <= bb_y_lims[1]
        mask5 = pc[...,2] >= bb_z_lims[0]
        mask6 = pc[...,2] <= bb_z_lims[1]

        mask = mask1 & mask2 & mask3 & mask4 & mask5 & mask6

        return mask

if __name__ == "__main__":
    
    torch.manual_seed(1)
    pc = torch.rand([1024,3])*0.4-0.02
    pc_orig = pc.numpy()
    pc = pc.unsqueeze(0).repeat([2, 1, 1])
    trans = torch.rand([2,3])*0.05
    # print(trans)
    # trans = torch.zeros([2,3])
    # quats = torch.FloatTensor(R.random(4).as_quat())

    quats = torch.FloatTensor([[ 0.7242,  0.5577, -0.3350,  0.2286],
                              [ 0.8214,  0.1087,  0.5012,  0.2495]])

    eulers = torch.FloatTensor(R.from_quat(quats).as_euler(seq='XYZ'))

    print(pc.shape, quats.shape, trans.shape)

    model = CollisionDistanceEvaluator()

    out = model.forward(trans=trans, quat=quats, pc=pc)
    # out, (logit, pc_inside) = model.forward_with_eulers(trans=trans, eulers=eulers, pc=pc)

    print(out)
    # import copy
    
    # gripper = PandaGripper(root_folder='../grasper')
    # gripper_mesh = trimesh.util.concatenate(gripper.get_meshes())
    # gripper_mesh.visual.face_colors = [125,125,152,50]

    # gripper_bb = gripper.get_bb(all=True)
    # gripper_bb.visual.face_colors = [125,255,152,50]

    
    # gripper_obbs = gripper.get_obbs()
    # left_finger = gripper_obbs[0]
    # right_finger = gripper_obbs[1]
    # base = gripper_obbs[2]

    # # print(gripper_obbs[1].to_dict())
    # # pc = np.random.rand(3,1024,3)*0.1
    # # pc_mesh = trimesh.points.PointCloud(pc, colors=[255,0,0,25])
    # # check if inside the box


    # # pc = np.random.rand(1024,3)*0.1 + 0.00  
    # # r = R.random()
    # # pc_rotated = r.inv().apply(pc)
    # # transform_rotated = np.eye(4)
    # # transform_rotated[:3,:3] = r.as_matrix()

    # # scene = trimesh.Scene()
    # # scene.add_geometry(trimesh.points.PointCloud(pc, colors=[255,0,0,100]))

    # # gripper = PandaGripper(root_folder='../grasper')
    # # gripper_bb = gripper.get_bb(all=True)
    # # gripper_bb.visual.face_colors = [255,0,0,100]
    # # scene.add_geometry(gripper_bb, transform=transform_rotated)



    # # scene.add_geometry(trimesh.points.PointCloud(pc_rotated, colors=[0,255,0,100]))
    # # gripper = PandaGripper(root_folder='../grasper')
    # # gripper_bb = gripper.get_bb(all=True)
    # # gripper_bb.visual.face_colors = [0,255,0,100]
    # # scene.add_geometry(gripper_bb)
    

    # # # do using torch
    # # pc_t = torch.FloatTensor(pc).unsqueeze(0)
    # # print(pc_t.shape)
    # # quat = torch.FloatTensor(r.as_quat()).unsqueeze(0)
    # # print(quat.shape)
    # # quat_inv = quaternion.quat_inv(quat)
    # # print(quat_inv.shape)
    # # quat_inv = quat_inv.unsqueeze(1).repeat([1, pc_t.shape[1], 1])
    # # print(quat_inv.shape)
    # # pc_t = quaternion.rot_p_by_quaterion(pc_t, quat_inv)
    # # print(pc_t.shape)
    # # scene.add_geometry(trimesh.points.PointCloud(pc_t.squeeze(0).numpy(), colors=[0,0,255,100]))

    # # scene.show()
    # # exit()
    
    # s = time.time()

    # torch.manual_seed(1)
    # pc = torch.rand([1024,3])*0.4-0.02
    # pc_orig = pc.numpy()
    # pc = pc.unsqueeze(0).repeat([2, 1, 1])
    # trans = torch.rand([2,3])*0.05
    # # print(trans)
    # # trans = torch.zeros([2,3])
    # # quats = torch.FloatTensor(R.random(4).as_quat())

    # quats = torch.FloatTensor([[ 0.7242,  0.5577, -0.3350,  0.2286],
    #                           [ 0.8214,  0.1087,  0.5012,  0.2495]])

    # eulers = torch.FloatTensor(R.from_quat(quats).as_euler(seq='XYZ'))

    # print(pc.shape, quats.shape, trans.shape)

    # model = CollisionDistanceEvaluator()

    # out, (logit, pc_inside) = model.forward(trans=trans, quat=quats, pc=pc)
    # # out, (logit, pc_inside) = model.forward_with_eulers(trans=trans, eulers=eulers, pc=pc)

    # print(logit)
    # print(f'time:{time.time()-s}')



    # # print(quats)

    # # print(quaternion.quat_inv(quats))



    # pc = pc.numpy()[1]
    
    # trans = trans.numpy()
    # quats = quats.numpy()
    
    # from utils.auxilary import construct_grasp_matrix

    # g = construct_grasp_matrix(translation=trans, quaternion=quats)

    # scene = trimesh.Scene()
    # for i in range(1):
    #     scene.add_geometry(gripper_bb)#, transform=g[i])
    

    # # my_pc = np.array([[ 0.13424024, -0.05131505,  0.14224389],
    # #                     [ 0.15344042, -0.04729388,  0.11810458],
    # #                     [ 0.1764972,  -0.02794172,  0.1427191 ],
    # #                     [ 0.08555949, -0.0573077,   0.11334571],
    # #                     [ 0.10933961, -0.02101273,  0.13953242],
    # #                     [ 0.15553665, -0.02524695,  0.10222766],
    # #                     [ 0.19290596, -0.04914754,  0.07529056],
    # #                     [ 0.15774454, -0.05648039,  0.08938295],
    # #                     [ 0.16793387, -0.04113238,  0.13949662],
    # #                     [ 0.13576244, -0.02339374,  0.16391948],
    # #                     [ 0.09903967, -0.056318,    0.11620012]])
    
    # # my_pc = trimesh.points.PointCloud(my_pc, colors=[255,0,0,255])
    # # scene.add_geometry(my_pc)
    # pc2 =out.numpy()[0]
    # scene.add_geometry(trimesh.points.PointCloud(pc2))

    # pc2 =out.numpy()[1]
    # scene.add_geometry(trimesh.points.PointCloud(pc2))

    # _pc = trimesh.points.PointCloud(pc_inside[0].numpy(), colors=[0,255,0,255])
    # scene.add_geometry(_pc)
    # _pc = trimesh.points.PointCloud(pc_inside[1].numpy(), colors=[0,255,255,255])
    # scene.add_geometry(_pc)
    
    # # scene.add_geometry(trimesh.points.PointCloud(pc))
    
    
    # scene.show()

    # import copy

    # scene = trimesh.Scene()
    # for i in range(0,2):
    #     gripper2 = PandaGripper(root_folder='../grasper')
    #     gripper_bb2 = gripper2.get_bb(all=True)
    #     gripper_bb2.visual.face_colors = [125,0,0,100]
    #     scene.add_geometry(gripper_bb2, transform=g[i])

    #     _pc = pc_inside[i].numpy()
    #     # _pc = R.from_matrix(g[i,:3,:3]).apply(_pc)
    #     if i == 0:

    #         scene.add_geometry(trimesh.points.PointCloud(_pc, colors=[0,255,0,255]), transform=g[i])
    #     else:
    #         scene.add_geometry(trimesh.points.PointCloud(_pc, colors=[255,255,255,255]), transform=g[i])


    # # scene.add_geometry(trimesh.points.PointCloud(pc2))
    # scene.add_geometry(trimesh.points.PointCloud(pc_orig))

    

    # scene.show()





    # # s = time.time()

    # # p_all_center = torch.FloatTensor([-1.78200e-03,1.00500e-05,4.31621e-02])
    # # extends_all_center = [0.204416, 0.0632517, 0.1381738]

    # # mask = bb_aligned(x=pc)
    
    # # lengths = torch.ones([B,1])
    # # for i in range(B):
    # #     inside_pc = pc[i,mask[i],:]
    # #     if inside_pc.shape[0] > 0:
    # #         lengths[i,0] = -1*torch.mean(torch.linalg.norm(pc[i,mask[i],:], dim=1))

    # # print(lengths)



    # # print(f'time:{time.time()-s}')


    # # s = time.time()
    # # sd = trimesh.proximity.signed_distance(base, pc)
    # # print(f'time: {time.time()-s}')

    # # s = time.time()
    # # sd = trimesh.proximity.signed_distance(gripper_mesh, pc)
    # # print(f'time: {time.time()-s}')

    


    

    # # pos_idx = sd<0

    # # pc_pos = pc[pos_idx,:]
    # # print(pc_pos.shape)
    # # pc_pos_mesh = trimesh.points.PointCloud(pc_pos, colors=[255,0,0,125])
    
    # # scene = trimesh.Scene()
    # # scene.add_geometry(gripper_mesh)
    # # scene.add_geometry(gripper_bb)
    # # scene.add_geometry(pc_mesh)
    # # scene.show()
