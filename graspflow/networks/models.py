#!/usr/bin/env python3
# Tasbolat Taunyazov

import numpy as np
import torch
import torch.nn as nn

from .pointnet2_utils import PointNetSetAbstraction
import yaml
import networks.utils as utils
import torch.nn.functional as F
from networks import quaternion
from sklearn.decomposition import PCA


class GraspTopFeatures(nn.Module):
    def __init__(self, config_path='configs/pointnet2_GRASPNET6DOF_evaluator4.yaml', control_point_path='configs/panda.npy'):
        super(GraspTopFeatures, self).__init__()
        
        # PointNet++ features
        self.pointnet_feature_extractor = PointnetHeader(config_path)

        # Latent space
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        # Evaluator
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2_2 = nn.Linear(128, 64)
        self.bn2_2 = nn.BatchNorm1d(64)
        
        self.classification = nn.Linear(64, 1)
        
        # Grasp output
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.quat_A_representation = nn.Linear(128, 10)
        self.trans_representation = nn.Linear(128, 3)

        self.control_point_path = control_point_path

        # semantics part

    def forward(self, quat, trans, pc):
        '''
        Input:
        either: quat: [B,4]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        # rotate and translate gripper point clouds
        gripper_pc = utils.transform_gripper_pc_old(quat, trans, config_path=self.control_point_path)            
            
        return self.evaluate_grasp(pc, gripper_pc)

    def forward_with_eulers(self, eulers, trans, pc):
        '''
        Input:
        either: euler: [B,3]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        gripper_pc = utils.control_points_from_rot_and_trans(eulers, trans, config_path=self.control_point_path)
        return self.evaluate_grasp(pc, gripper_pc)
    
    def forward_with_A_vec(self, A_vec, trans, pc):
        quat = utils.A_vec_to_quat(A_vec)
        return self.forward(quat, trans, pc)
        
    def evaluate_grasp(self, pc, gripper_pc):
        # concatenate gripper_pc with pc
        pc, pc_features = utils.merge_pc_and_gripper_pc2(pc, gripper_pc)
        pc = pc.permute(0,2,1)
        top_feats = self.pointnet_feature_extractor(pc, pc_features)
        top_feats = self.fc1(top_feats)
        top_feats = torch.relu(self.bn1(top_feats))
        

        return top_feats


class SemanticEvaluator(nn.Module):
    def __init__(self):
        super(SemanticEvaluator, self).__init__()
        
        # Latent space
        self.fc1 = nn.Linear(512, 1)
        


    def forward(self, top_feats):
        '''
        Input:
        top_feats: [B,1024]
        '''
        # rotate and translate gripper point clouds
        x = self.fc1(top_feats)   
            
        return x

class SemanticEvaluator(nn.Module):
    def __init__(self):
        super(SemanticEvaluator, self).__init__()
        
        # Latent space
        self.fc1 = nn.Linear(512, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 1)


    def forward(self, top_feats):
        '''
        Input:
        top_feats: [B,1024]
        '''
        # rotate and translate gripper point clouds
        x = self.fc1(top_feats)
        x = torch.relu(self.bn1(x))
        x = self.fc2(x)
            
        return x

class TactileEvaluator(nn.Module):
    def __init__(self, pc, cg_mean_init=0, cg_std_init=1):
        self.mean_estimate = torch.FloatTensor([0.0])
        self.mean_estimate.requires_grad_(True)
        self.std_estimate = torch.FloatTensor([1.0])
        self.std_estimate.requires_grad_(True)
        
        self.likelihood = torch.distributions.Normal(self.mean_estimate, self.std_estimate)
        self.mean_prior = torch.distributions.Normal(cg_mean_init, torch.FloatTensor([10.0]))
        self.std_prior = torch.distributions.Normal(cg_std_init, torch.FloatTensor([1.0]))
        
        self.observations = []
        
    def forward(self, e, t, pc):
        pc = pc[0] # ignore batch, (1024, 3)
        pc_2d = pc[:, :2].numpy() # ignore z-axis
        pc_2d_mean = np.mean(pc_2d, axis=0)
        pca = PCA(n_components=2)
        pca.fit(pc_2d)
        
        comp = pca.components_[0] # choose longest axis
        var = pca.explained_variance_[0]
        comp = comp * var * 0.1
        print(comp)
        cg_belief_2d_point = torch.FloatTensor([
            pc_2d_mean[0] + self.mean_estimate.item() * comp[0], 
            pc_2d_mean[1] + self.mean_estimate.item() * comp[1]
        ])

        unit_vector = torch.zeros([e.shape[0], 3])
        unit_vector[:, 2] = 0.11
        unit_vector = unit_vector.to(self.device)
        rotations = euler2matrix(e, 'XYZ')
        unit_vector = torch.einsum('ik,ijk->ij', unit_vector, rotations)

        dist = (t + unit_vector - cg_belief_2d_point).pow(2).sum(1).sqrt()
        #logit = 3 - (30 * (dist - 0.015))

        logit = 1000 * (0.005 - dist)
        return logit

    def forward_with_euler(self, e, trans, pc):
        return self.forward(e, trans, pc)
    
    def visualize(self):
        print('observations: ', self.observations)
        print('mean: ', self.mean_estimate.item())
        print('std: ', self.std_estimate.item())

    def update(self, grasp_trans, rotation_speed, z_axis_force):
        # Update CG posterior, grasp_trans and z_axis_force not used for now
        self.observations.append(rotation_speed)
        
        prior_ = self.mean_prior.log_prob(self.mean_estimate) + self.std_prior.log_prob(self.std_estimate)
        posterior = torch.mean(self.likelihood.log_prob(torch.FloatTensor(self.observations))) + prior_

        if np.isnan(posterior.data[0]) or np.isnan(prior_.data[0]):
            return

        posterior.backward()
        for param in (self.mean_estimate, self.std_estimate):
            param.data.add_(1 * param.grad.data)
            param.grad.data.zero_()
        


class GraspEvaluatorDistance(nn.Module):
    def __init__(self, z_threshold=0, angle_threshold=0.1, coeff=1):
        super(GraspEvaluatorDistance, self).__init__()
        self.z_threshold = z_threshold
        self.angle_threshold = angle_threshold
        self.coeff = coeff

    def forward(self, trans, quat,pc=None):
        # rotate and translate gripper point clouds
        gripper_pc = utils.transform_gripper_pc_old(quat, trans)            
        return self.evaluate_grasp(gripper_pc)

    def forward_with_eulers(self, trans, eulers, pc=None):
        gripper_pc = utils.control_points_from_rot_and_trans(eulers, trans)
        return self.evaluate_grasp(gripper_pc)
        
    def evaluate_grasp(self, gripper_pc):

        # compute translation distance
        z_mean = torch.mean(gripper_pc[...,2], dim=1, keepdim=True)
        t_dist = z_mean-self.z_threshold 
        
        # compute angle distance
        #upward_direction = torch.FloatTensor([[0,0,1]]).to(quats.device) 
        #g_direction = rot_p_by_quaterion(upward_direction.repeat([quats.shape[0],1]).unsqueeze(0), quats.unsqueeze(0)).squeeze(0)
        #angle = torch.arccos(-g_direction[:,2]).unsqueeze(-1)
        #angle_dist = self.angle_threshold-angle

        ### compute comnined distance
        # dist = torch.stack([t_dist, angle_dist])
        # dist,_ = torch.min(dist, dim=0)
        # dist = torch.mean(dist, dim=0)
        # d = self.coeff*(dist)

       

        d = self.coeff*t_dist# + self.coeff*angle_dist

        return d

# class CollisionDistanceEvaluator(nn.Module):
#     def __init__(self, npoints=10, dist_threshold=0.001, dist_coeff=10000):
#         super(CollisionDistanceEvaluator, self).__init__()
#         '''
#         Evaluator on collision with point cloud.
#         Very naive implementation. Relies on ideal data. Please clean pc very good before using this module.
#         '''
#         self.npoints = npoints
#         self.dist_threshold = dist_threshold
#         self.dist_coeff = dist_coeff

#     def forward(self, trans, quat, pc):
#         '''
#         trans [B,3]
#         quat [B,4]
#         pc [B,1024,3]
#         '''
#         # rotate and translate gripper point clouds
#         gripper_pc = utils.transform_gripper_pc_old(quat, trans) # [B,6,3]

#         return self.evaluate_grasp(gripper_pc, pc)

#     def forward_with_eulers(self, trans, eulers, pc):
#         gripper_pc = utils.control_points_from_rot_and_trans(eulers, trans)
#         return self.evaluate_grasp(gripper_pc, pc)

#     def evaluate_grasp(self, gripper_pc, pc):
#         # calculate gripper mean point
#         gripper_loc = torch.mean(gripper_pc, dim=1, keepdim=True) # [B,1,3]

#         # get distances to all points
#         dist = gripper_loc - pc # [B,1024,3]
#         dist = torch.linalg.norm(dist, dim=2) # [B,1024]

#         # find closest point 
#         dist_min = torch.min(dist, dim=1)[0] # [B]

#         # check distances and make gradients almost zero for non-colliding grasps
#         logit = torch.ones([gripper_pc.shape[0]]).to(pc.device)*self.dist_coeff
#         dist_flag = dist_min < self.dist_threshold
#         logit[dist_flag] = self.dist_coeff*(dist_min[dist_flag]-self.dist_threshold)
#         logit=logit.unsqueeze(-1)

#         return logit


# class CollisionDistanceEvaluator(nn.Module):
#     def __init__(self, npoints=10, dist_threshold=0.001, dist_coeff=10000, approximate=False):
#         super(CollisionDistanceEvaluator, self).__init__()
#         '''
#         Evaluator on collision with point cloud.
#         Very naive implementation. Relies on ideal data. Please clean pc very good before using this module.
#         '''
#         self.npoints = npoints
#         self.dist_threshold = dist_threshold
#         self.dist_coeff = dist_coeff
#         self.approximate=approximate

#     def forward(self, trans, quat, pc_env):
#         '''
#         trans [B,3]
#         quat [B,4]
#         pc [B,n,3]
#         '''

#         gripper_pc = utils.transform_gripper_pc_old(quat, trans) # [B,6,3]

#         # gripper_pc = utils.get_surrogate_grasp_points(quat, trans, is_euler=False)
#         pc = pc_env.clone()
#         pc = pc - trans.unsqueeze(1).repeat([1,pc.shape[1],1])
#         # rotate and translate point clouds to inverse transformation
#         quat_inv = quaternion.quat_inv(quat)
#         quat_inv = quat_inv.unsqueeze(1).repeat([1, pc.shape[1], 1])
#         pc = quaternion.rot_p_by_quaterion(pc, quat_inv)
        
#         return self.evaluate_grasp(pc, gripper_pc, trans, quat)

#     def forward_with_eulers(self, trans, eulers, pc_env):

#         gripper_pc = utils.get_surrogate_grasp_points(eulers, trans, is_euler=True)
#         pc = pc_env.clone()
#         pc = pc - trans.unsqueeze(1).repeat([1,pc.shape[1],1])

#         # rotate and translate point clouds to inverse transformation
#         rotmat = quaternion.euler2matrix(eulers, order='XYZ') # [B,3,3]
#         rotmat = torch.linalg.inv(rotmat)
#         pc = torch.bmm(pc,rotmat) # TODO: NEED TO BE CHECKED
#         quats = quaternion.euler2quat(eulers, order='XYZ')

#         return self.evaluate_grasp(pc, gripper_pc, trans, quats)

#     def evaluate_grasp(self, pc, gripper_pc, trans, quat):

#         #B = pc.shape[0]

#         # big bounding box of the grasp

#         if self.approximate:
#             p_all_center = torch.FloatTensor([-1.78200e-03,1.00500e-05,4.31621e-02]).to(pc.device)
#             half_extends_all = torch.FloatTensor([0.204416/2+self.dist_threshold, 0.0632517/2+self.dist_threshold, 0.1381738/2+self.dist_threshold]).to(pc.device)
#             d = self.sdf_box(points=pc, box_center=p_all_center, box_extends=half_extends_all)
#         else:
#             # left finger
#             p_left_center = torch.FloatTensor([-0.0531353785, 7.649999999998804e-06, 0.085390348]).to(pc.device)
#             half_extends_left = torch.FloatTensor([0.026536043000000002/2+self.dist_threshold, 0.0209743/2+self.dist_threshold, 0.053717304/2+self.dist_threshold]).to(pc.device)
#             d1 = self.sdf_box(points=pc, box_center=p_left_center, box_extends=half_extends_left)

#             # right finger
#             p_right_center = torch.FloatTensor([0.0531353785, -7.649999999997069e-06, 0.085390348]).to(pc.device)
#             half_extends_right = torch.FloatTensor([0.026536043000000002/2+self.dist_threshold, 0.0209743/2+self.dist_threshold, 0.053717304/2+self.dist_threshold]).to(pc.device)
#             d2 = self.sdf_box(points=pc, box_center=p_right_center, box_extends=half_extends_right)

#             # base
#             p_base_center = torch.FloatTensor([-0.0017819999999999989, 1.005000000000103e-05, 0.0200187]).to(pc.device)
#             half_extends_base = torch.FloatTensor([0.204416/2+self.dist_threshold, 0.0632517/2+self.dist_threshold, 0.091887/2+self.dist_threshold]).to(pc.device)
#             d3 = self.sdf_box(points=pc, box_center=p_base_center, box_extends=half_extends_base)

#             d = d1 + d2 + d3

#         # rotate and translate point clouds to inverse transformation
#         quat = quat.unsqueeze(1).repeat([1, pc.shape[1], 1])
#         pc = quaternion.rot_p_by_quaterion(pc, quat)
#         pc = pc + trans.unsqueeze(1).repeat([1,pc.shape[1],1])
    
#         d[d > 0] = 0.0

#         dist = torch.sum(d, dim=1, keepdim=True)
#         dist[dist == 0] = 1.0
#         logit = dist*self.dist_coeff

#         return logit

#     def sdf_box(self, points, box_center, box_extends):
#         '''
#         SDF for box
#         points [B,n,3]
#         box_center [3]
#         box_extends [3]
#         '''

#         all_distances = []
#         for i in range(3):
#             temp_left = points[:,:,0]-box_center[0]-box_extends[0]
#             temp_right = box_center[0]-points[:,:,0]-box_extends[0]
#             temp = torch.stack([temp_left, temp_right])
#             values, idx = torch.max(temp, dim=0)
#             all_distances.append(values)

#         d, idx = torch.max(torch.stack(all_distances), dim=0)
#         return d


class FarGraspEvaluator(nn.Module):
    def __init__(self, threshold=0.01,coeff=10000):
        super(FarGraspEvaluator, self).__init__()
        '''
        Evaluator on far grasp evaluator. Relies on ideal data.
        Very naive implementation. Relies on ideal data. Please clean pc very good before using this module.
        '''
        self.threshold = threshold
        self.coeff = coeff

    def forward(self, trans, quat, pc):
        '''
        trans [B,3]
        quat [B,4]
        pc [B,n,3]
        '''

        # gripper_pc = utils.transform_gripper_pc_old(quat, trans) # [B,6,3]

        quat_new = torch.zeros_like(quat)
        quat_new[:,3] = 1.0

        gripper_pc = utils.get_surrogate_grasp_points(quat_new, trans, is_euler=False)

        # rotate and translate point clouds to inverse transformation
        

        return self.evaluate_grasp(pc, gripper_pc)

    def forward_with_eulers(self, trans, eulers, pc):

        eulers_new = torch.zeros_like(eulers)

        gripper_pc = utils.get_surrogate_grasp_points(eulers_new, trans, is_euler=False)

        return self.evaluate_grasp(pc, gripper_pc)

    def evaluate_grasp(self, pc, gripper_pc):

        pc = torch.mean(pc, dim=1, keepdim=True)
        # print(pc.shape)
        # print(gripper_pc.shape)

        dist = torch.mean((pc-gripper_pc)**2, dim=2)

        logit = self.coeff*(self.threshold-dist)
        # print(logit.shape)
        # print(logit)

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

class CollisionDistanceEvaluator(nn.Module):
    def __init__(self, dist_threshold=0.001, dist_coeff=10000, approximate=False, base_extension=0.0,
                                                   fill_between_fingers=False):
        super(CollisionDistanceEvaluator, self).__init__()
        '''
        Evaluator on collision with point cloud.
        Very naive implementation. Relies on ideal data. Please clean pc very good before using this module.
        '''
        self.dist_threshold = dist_threshold
        self.dist_coeff = dist_coeff
        self.approximate=approximate
        self.fill_between_fingers = fill_between_fingers
        self.base_extension = base_extension

    def forward(self, trans, quat, pc_env):
        '''
        trans [B,3]
        quat [B,4]
        pc [B,n,3]
        '''

        # gripper_pc = utils.transform_gripper_pc_old(quat, trans) # [B,6,3]

        gripper_pc = utils.get_surrogate_grasp_points(quat, trans, is_euler=False)
        pc = pc_env.clone()
        pc = pc - trans.unsqueeze(1).repeat([1,pc.shape[1],1])

        # rotate and translate point clouds to inverse transformation
        quat_inv = quaternion.quat_inv(quat)
        quat_inv = quat_inv.unsqueeze(1).repeat([1, pc.shape[1], 1])
        pc = quaternion.rot_p_by_quaterion(pc, quat_inv)
        

        return self.evaluate_grasp(pc, gripper_pc, trans, quat)

    def forward_with_eulers(self, trans, eulers, pc_env):

        gripper_pc = utils.get_surrogate_grasp_points(eulers, trans, is_euler=True)
        pc = pc_env.clone()
        pc = pc - trans.unsqueeze(1).repeat([1,pc.shape[1],1])

        # rotate and translate point clouds to inverse transformation
        rotmat = quaternion.euler2matrix(eulers, order='XYZ') # [B,3,3]
        rotmat = torch.linalg.inv(rotmat)
        pc = torch.bmm(pc,rotmat) # TODO: NEED TO BE CHECKED
        quats = quaternion.euler2quat(eulers, order='XYZ')

        return self.evaluate_grasp(pc, gripper_pc, trans, quats)

    def evaluate_grasp(self, pc, gripper_pc, trans, quat):

        # B = pc.shape[0]
        # big bounding box of the grasp

        if self.approximate:

            p_all_center = torch.FloatTensor([-1.78200e-03,1.00500e-05,(4.31621e-02-self.base_extension/2)]).to(pc.device)
            half_extends_all = torch.FloatTensor([0.204416/2+self.dist_threshold, 0.0632517/2+self.dist_threshold, (0.1381738+self.base_extension)/2+self.dist_threshold]).to(pc.device)
            mask = self.bb_aligned(pc=pc, p=p_all_center, half_extents=half_extends_all)
        else:
            # left finger
            p_left_center = torch.FloatTensor([-0.0531353785, 7.649999999998804e-06, 0.085390348]).to(pc.device)
            half_extends_left = torch.FloatTensor([0.026536043000000002/2+self.dist_threshold, 0.0209743/2+self.dist_threshold, 0.053717304/2+self.dist_threshold]).to(pc.device)
            mask_left_finger = self.bb_aligned(pc=pc, p=p_left_center, half_extents=half_extends_left)

            # right finger
            p_right_center = torch.FloatTensor([0.0531353785, -7.649999999997069e-06, 0.085390348]).to(pc.device)
            half_extends_right = torch.FloatTensor([0.026536043000000002/2+self.dist_threshold, 0.0209743/2+self.dist_threshold, 0.053717304/2+self.dist_threshold]).to(pc.device)
            mask_right_finger = self.bb_aligned(pc=pc, p=p_right_center, half_extents=half_extends_right)

            # base
            p_base_center = torch.FloatTensor([-0.0017819999999999989, 1.005000000000103e-05, (0.0200187-self.base_extension/2)]).to(pc.device)
            half_extends_base = torch.FloatTensor([0.204416/2+self.dist_threshold, 0.0632517/2+self.dist_threshold, (0.091887+self.base_extension)/2+self.dist_threshold]).to(pc.device)
            mask_base = self.bb_aligned(pc=pc, p=p_base_center, half_extents=half_extends_base)
            
            mask = mask_left_finger | mask_right_finger | mask_base

            if self.fill_between_fingers:
                p_middle_center = torch.FloatTensor([0.0, -7.649999999997069e-06, 0.085390348])
                half_extends_middle = torch.FloatTensor([0.0796714215/2+self.dist_threshold, 0.0209743/2+self.dist_threshold, 0.053717304/2+self.dist_threshold])
                mask_middle = self.bb_aligned(pc=pc, p=p_middle_center, half_extents=half_extends_middle)
                mask = mask | mask_middle
       

        # rotate and translate point clouds to inverse transformation
        quat = quat.unsqueeze(1).repeat([1, pc.shape[1], 1])
        pc = quaternion.rot_p_by_quaterion(pc, quat)
        pc = pc + trans.unsqueeze(1).repeat([1,pc.shape[1],1])
        

        # mask_check = mask.clone()
        mask = mask.long()

        if gripper_pc.shape[1] != 1:
            gripper_loc = torch.mean(gripper_pc, dim=1, keepdim=True)
        else:
            gripper_loc = gripper_pc

        # dist = torch.sum((pc-gripper_loc)**2, dim=2)-1e8

        dist = torch.mean((pc-gripper_loc)**2, dim=2)-1
        dist = dist * mask

        dist = torch.sum(dist, dim=1, keepdim=True)
        dist[dist == 0] = 1.0
        logit = dist*self.dist_coeff

        # if self.reverse:
        #     logit = -1*logit

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


class GraspEvaluator(nn.Module):
    def __init__(self, config_path='configs/pointnet2_GRASPNET6DOF_evaluator4.yaml', control_point_path='configs/panda.npy'):
        super(GraspEvaluator, self).__init__()
        
        # PointNet++ features
        self.pointnet_feature_extractor = PointnetHeader(config_path)

        # Latent space
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)

        # Evaluator
        self.fc2 = nn.Linear(512, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc2_2 = nn.Linear(128, 64)
        self.bn2_2 = nn.BatchNorm1d(64)
        
        self.classification = nn.Linear(64, 1)
        
        # Grasp output
        self.fc3 = nn.Linear(512, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.quat_A_representation = nn.Linear(128, 10)
        self.trans_representation = nn.Linear(128, 3)

        self.control_point_path = control_point_path

    def forward(self, quat, trans, pc):
        '''
        Input:
        either: quat: [B,4]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        # rotate and translate gripper point clouds
        gripper_pc = utils.transform_gripper_pc_old(quat, trans, config_path=self.control_point_path)            
            
        return self.evaluate_grasp(pc, gripper_pc)

    def forward_with_eulers(self, eulers, trans, pc):
        '''
        Input:
        either: euler: [B,3]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        gripper_pc = utils.control_points_from_rot_and_trans(eulers, trans, config_path=self.control_point_path)
        return self.evaluate_grasp(pc, gripper_pc)
    
    def forward_with_A_vec(self, A_vec, trans, pc):
        quat = utils.A_vec_to_quat(A_vec)
        return self.forward(quat, trans, pc)
        
    def evaluate_grasp(self, pc, gripper_pc):
        # concatenate gripper_pc with pc
        pc, pc_features = utils.merge_pc_and_gripper_pc2(pc, gripper_pc)
        pc = pc.permute(0,2,1)
        top_feats = self.pointnet_feature_extractor(pc, pc_features) # 1024
        top_feats = self.fc1(top_feats)
        top_feats = torch.relu(self.bn1(top_feats)) # 512

        # evaluator
        eval_feats = self.fc2(top_feats)
        eval_feats = torch.relu(self.bn2(eval_feats))
        eval_feats = self.fc2_2(eval_feats)
        eval_feats = torch.relu(self.bn2_2(eval_feats))
        eval_out = self.classification(eval_feats) # expected output Bx1

        # grasp: quat with trans
        grasp_feats = self.fc3(top_feats)
        grasp_feats = torch.relu(self.bn3(grasp_feats))
        A_vec = self.quat_A_representation(grasp_feats)
        quaternions = utils.A_vec_to_quat(A_vec) # quaternion
        translations = self.trans_representation(grasp_feats) # translation

        return eval_out, quaternions, translations

class TaskGrasp(nn.Module):
    def __init__(self,):
        super(GraspEvaluator, self).__init__()
        pass

    def forward(self, t, q, pc):
        pass

# class TaskEmbeddingExtractor(nn.Module):
#     def __init__(self):
#         super(TaskEmbeddingExtractor, self).__init__()
#         self.fc1 = nn.Linear(300, 128)
#         self.bn1 = nn.BatchNorm1d(128)
#         self.fc2 = nn.Linear(128, 32)
#         self.bn2 = nn.BatchNorm1d(32)

    
#     def forward(self, task_embedding):
#         task_feats = self.fc1(task_embedding)
#         task_feats = torch.relu(self.bn1(task_feats))
#         task_feats = self.fc2(task_feats)
#         task_feats = torch.relu(self.bn2(task_feats))
#         return task_feats


# class SemanticGraspEvaluator(nn.Module):
#     def __init__(self, config_path='configs/pointnet2_GRASPNET6DOF_evaluator4.yaml', control_point_path='configs/panda.npy'):
#         super(SemanticGraspEvaluator, self).__init__()
        
#         # PointNet++ features
#         self.pointnet_feature_extractor = PointnetHeader(config_path)

#         # Task features
#         self.task_feature_extractor = TaskEmbeddingExtractor()

#         # Latent space
#         self.fc1 = nn.Linear(1024+32, 512) # add task feature
#         self.bn1 = nn.BatchNorm1d(512)

#         # Evaluator
#         self.fc2 = nn.Linear(512, 128)
#         self.bn2 = nn.BatchNorm1d(128)
#         self.fc2_2 = nn.Linear(128, 64)
#         self.bn2_2 = nn.BatchNorm1d(64)
        
#         self.classification = nn.Linear(64, 1)
        
#         # Grasp output
#         self.fc3 = nn.Linear(512, 128)
#         self.bn3 = nn.BatchNorm1d(128)
#         self.quat_A_representation = nn.Linear(128, 10)
#         self.trans_representation = nn.Linear(128, 3)

#         self.control_point_path = control_point_path

#     def forward(self, quat, trans, pc, task_embedding):
#         '''
#         Input:
#         either: quat: [B,4]
#         trans: [B,3]
#         pc: [B,1024,3]
#         '''
#         # rotate and translate gripper point clouds
#         gripper_pc = utils.transform_gripper_pc_old(quat, trans, config_path=self.control_point_path)       
            
#         return self.evaluate_grasp(pc, gripper_pc, task_embedding)

#     # def forward_with_eulers(self, eulers, trans, pc):
#     #     '''
#     #     Input:
#     #     either: euler: [B,3]
#     #     trans: [B,3]
#     #     pc: [B,1024,3]
#     #     '''
#     #     gripper_pc = utils.control_points_from_rot_and_trans(eulers, trans, config_path=self.control_point_path)
#     #     return self.evaluate_grasp(pc, gripper_pc)
    
#     # def forward_with_A_vec(self, A_vec, trans, pc):
#     #     quat = utils.A_vec_to_quat(A_vec)
#     #     return self.forward(quat, trans, pc)
        
#     def evaluate_grasp(self, pc, gripper_pc, task):
#         # get task features
#         task_feature = self.task_feature_extractor(task)     

#         # concatenate gripper_pc with pc
#         pc, pc_features = utils.merge_pc_and_gripper_pc2(pc, gripper_pc)
#         pc = pc.permute(0,2,1)

#         # get pointnet feature
#         pointnet_feature = self.pointnet_feature_extractor(pc, pc_features)

#         # # concatenate task feature and pointnet feature
#         # print("task_feature shape",task_feature.shape)
#         # print("pointnet_feature shape", pointnet_feature.shape)
#         top_feats = torch.cat((task_feature,pointnet_feature),1)
#         top_feats = self.fc1(top_feats)
#         top_feats = torch.relu(self.bn1(top_feats))

#         # evaluator
#         eval_feats = self.fc2(top_feats)
#         eval_feats = torch.relu(self.bn2(eval_feats))
#         eval_feats = self.fc2_2(eval_feats)
#         eval_feats = torch.relu(self.bn2_2(eval_feats))
#         eval_out = self.classification(eval_feats) # expected output Bx1

#         # grasp: quat with trans
#         grasp_feats = self.fc3(top_feats)
#         grasp_feats = torch.relu(self.bn3(grasp_feats))
#         A_vec = self.quat_A_representation(grasp_feats)
#         quaternions = utils.A_vec_to_quat(A_vec)
#         translations = self.trans_representation(grasp_feats)

#         return eval_out, quaternions, translations

class GraspEvaluatorV2(nn.Module):
    def __init__(self, config_path='configs/pointnet2_GRASPNET6DOF_evaluator.yaml'):
        super(GraspEvaluatorV2, self).__init__()
        
        # PointNet++ features
        self.pointnet_feature_extractor = PointnetHeader(config_path)

        # Latent space
        self.fc1 = nn.Linear(512+13, 256)
        self.bn1 = nn.BatchNorm1d(256)

        # Evaluator
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.classification = nn.Linear(64, 1)

    def forward(self, A_vec, trans, pc):
        '''
        Input:
        A_vec: [B,10]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        pc_feats = self.pointnet_feature_extractor(pc, None)
        all_feats = torch.concat([pc_feats, A_vec, trans], dim=1)

        x = torch.relu(self.bn1(self.fc1(all_feats)))
        x = torch.relu(self.bn2(self.fc2(x)))
        x = torch.relu(self.bn3(self.fc3(x)))

        out = self.classification(x)            
        return out
        


class PointnetHeader(nn.Module):
    def __init__(self, path_to_cfg, no_feats=False):
        super(PointnetHeader, self).__init__()

        # load pointnet header configs
        pointnet_params = yaml.safe_load(open(path_to_cfg, 'r'))
        if no_feats:
            pointnet_params[0]['in_channel'] = 0  # ADHOC
        
        self.pointnet_modules = nn.ModuleList()
        for _, params in pointnet_params.items():
            # we use positions as features also
            in_channel = params['in_channel'] + 3
            bias = True
            if 'bias' in params:
                bias = params['bias']
            sa_module = PointNetSetAbstraction(npoint=params['npoint'],
                                            radius=params['radius'],
                                            nsample=params['nsample'],
                                            in_channel=in_channel,
                                            mlp=params['mlp'],
                                            group_all=params['group_all'],
                                            bias=bias)
            self.pointnet_modules.append( sa_module )

    def forward(self, points, point_features):
        for pointnet_layer in self.pointnet_modules:
            #print('here 3', point_features.shape)
            points, point_features = pointnet_layer(points, point_features)
        return point_features.squeeze(-1)



class NNCorrector(torch.nn.Module):
    def __init__(self):
        super(NNCorrector, self).__init__()
        self.fc1 = nn.Linear(1024, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.classification = nn.Linear(64, 1)

    def forward(self, features):
        '''
        Input:
        features: [B,1024]
        '''
        x = torch.relu(self.bn1(self.fc1(features)))
        x = torch.relu(self.bn2(self.fc2(x)))
        out = self.classification(x)
        return out

class GPFeatureExtractor(nn.Module):
    def __init__(self, config_path='configs/pointnet2_GRASPNET6DOF_evaluator4.yaml',
                 control_point_path='configs/panda.npy',
                 path_to_weights='saved_models/PointNet_pretrained/pretrained_mode_62_9287.pt'):
        super(GPFeatureExtractor, self).__init__()
        
        # PointNet++ features
        self.pointnet_feature_extractor = PointnetHeader(config_path)

        # Loading Pretrained PointNet++ weights
        all_weights = torch.load(path_to_weights)
        extract_weights = dict(list(all_weights.items())[:63])
        PN_weights = {k.split(".",1)[1]: v for k, v in extract_weights.items()}

        self.pointnet_feature_extractor.load_state_dict(PN_weights)

        # Latent space
        # self.fc1 = nn.Linear(1024, 512)
        # self.bn1 = nn.BatchNorm1d(512)

        self.control_point_path = control_point_path

    def forward(self, quat, trans, pc):
        '''
        Input:
        either: quat: [B,4]
        trans: [B,3]
        pc: [B,1024,3]
        '''

        gripper_pc = utils.transform_gripper_pc_old(quat, trans, config_path=self.control_point_path)

        return self.get_feature(pc, gripper_pc)

    def forward_with_eulers(self, eulers, trans, pc):
        '''
        Input:
        either: euler: [B,3]
        trans: [B,3]
        pc: [B,1024,3]
        '''
        gripper_pc = utils.control_points_from_rot_and_trans(eulers, trans, config_path=self.control_point_path)
        return self.get_feature(pc, gripper_pc)

    def get_feature(self, pc, gripper_pc):
        # gripper_pc = utils.transform_gripper_pc_old(quat, trans, config_path=self.control_point_path)            
        # concatenate gripper_pc with pc
        pc, pc_features = utils.merge_pc_and_gripper_pc2(pc, gripper_pc)
        pc = pc.permute(0,2,1)
        top_feats = self.pointnet_feature_extractor(pc, pc_features) # 1024
        # top_feats = self.fc1(top_feats)
        # top_feats = torch.relu(self.bn1(top_feats)) # 512

        return top_feats
