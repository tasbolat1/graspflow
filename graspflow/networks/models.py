#!/usr/bin/env python3
# Tasbolat Taunyazov


import torch
import torch.nn as nn

from .pointnet2_utils import PointNetSetAbstraction
import yaml
import networks.utils as utils
import torch.nn.functional as F


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
