from os import link
import numpy as np
import torch
import torch.nn as nn
import time
from robot_ik_model import RobotModel

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel
)

def quaternion_distance(quat1, quat2, keepdim=False):
    '''
    Input Bx4
    Output B
    '''
    return 2*torch.arccos( torch.abs( torch.sum(quat1 * quat2, dim=1, keepdim=keepdim) ) )

class DifferentiableFrankaPanda(DifferentiableRobotModel):
    def __init__(self, device=None, urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf'):
        '''
        Core differentiable franka panda robot model by facebook based on pytorch. Use Only to create learanable pytorch nn.Model
        Input:
            - device: indicate torch.device
            - urdf_path: urdf description of franka robot
        Return:
            - robot_model: differentiable franka panda robot
        '''
        self.urdf_path = urdf_path
        self.learnable_rigid_body_config = None
        self.name = "differentiable_franka_panda"
        super().__init__(self.urdf_path, self.name, device=device)


class PandaRobot(nn.Module):
    def __init__(self, device=None, urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas_without_finger.urdf'):
        # ../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf
        
        '''
        Differentiable franka panda robot model by facebook based on pytorch.
        Input:
            - device: indicate torch.device
            - urdf_path: urdf description of franka robot
        Return:
            - robot_model: differentiable franka panda robot
        '''
        super(PandaRobot, self).__init__()
        self.franka_differentiable_model = DifferentiableFrankaPanda(device=device, urdf_path=urdf_path)

    def forward(self, joint_angles, link_name="panda_gripper_center"):
        '''
        Calculates forward kinematics of the franka robot respect to link_name
        Input:
            - joint_angles [B,7]: joint configuration in radians
            - link_name: name of the link in the urdf
        Return:
            - translation: translation of fk
            - quaternions: quaternions of fk
        '''
        zeros = torch.zeros(joint_angles.shape[0], 2).to(joint_angles.device)
        joint_angles = torch.cat([joint_angles, zeros], 1)
        translations, quaterions = self.franka_differentiable_model.compute_forward_kinematics(joint_angles, link_name=link_name)
        return translations, quaterions

class RobotClassifier(nn.Module):
    def __init__(self, threshold=0.0001, coeff=10000, device=None, urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas_without_finger.urdf'):
        # ../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf
        super(RobotClassifier, self).__init__()
        self.franka_differentiable_model = DifferentiableFrankaPanda(device=device, urdf_path=urdf_path)
        self.threshold = threshold # threshold (later will be taken log of it)
        self.coeff = coeff # larger is better approximation to heavy side
        self.clap_max = 100

    def check_joint_angles(self, joint_angles):
        assert len(joint_angles.shape) == 2, f"joint angles must have shape of [B,7], but given {joint_angles.shape}"
        return


    def forward(self, joint_angles):
        '''
        Computes manipulablity measure of the panda robot taking into account both pre-grasp and grasp points
        Input:
            - joint_angles [B,7]
        Output:
            - logit: [B,1]
        '''

        # compute manipulability_measure
        det_A1, manipulability_measure = self.compute_manipulability_measure(joint_angles, link_name="panda_gripper_center")

        if manipulability_measure.isnan().any():
            # TODO: pull toward the configuration space
            pass

        logit = self.coeff*(manipulability_measure - self.threshold)

        
        

        logit = torch.clamp(logit, max=self.clap_max, min=-self.clap_max)
        return logit.unsqueeze(-1)
    

    def forward_with_pre(self, joint_angles, joint_angles_pre):
        '''
        Computes manipulablity measure of the panda robot taking into account both pre-grasp and grasp points
        Input:
            - joint_angles [B,7]
            - joint_angles_pre [B,7]
        Output:
            - logit: [B,1]
        '''

        # compute manipulability_measure
        det_A1, manipulability_measure = self.compute_manipulability_measure(joint_angles, link_name="panda_gripper_center")
        det_A2, manipulability_measure2 = self.compute_manipulability_measure(joint_angles_pre, link_name="panda_gripper_center")

        manipulability_measure = torch.stack([manipulability_measure, manipulability_measure2])

        if manipulability_measure.isnan().any():
            # TODO: pull toward the configuration space
            pass
        
        min_man_measure, _ = torch.min(manipulability_measure, dim=0)
        logit = self.coeff*(min_man_measure - self.threshold)

        logit = torch.clamp(logit, max=self.clap_max, min=-self.clap_max)
        return logit.unsqueeze(-1)

    def workspace_measure(self, joint_angles):
        return

    def probability(self, joint_angles):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - probability: [B,1]
        '''
        self.check_joint_angles(joint_angles)

        logit = self.forward(joint_angles)

        return torch.sigmoid(logit)

    
    def compute_jacobian(self, joint_angles, link_name="panda_gripper_center"):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - Jacobian: [B,n,k]
        '''

        # fill joint angles
        zeros = torch.zeros(joint_angles.shape[0], 2).to(joint_angles.device)
        joint_angles = torch.cat([joint_angles, zeros], 1)
        self.check_joint_angles(joint_angles)
        
        # compute Jacobian
        J_v, J_w = self.franka_differentiable_model.compute_endeffector_jacobian(joint_angles, link_name=link_name)
        J = torch.hstack([J_v[:,:,:7], J_w[:,:,:7]])

        return J


    def compute_manipulability_measure(self, joint_angles, link_name="panda_gripper_center"):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - probability: [B,1]
        '''

        # compute Jacobian
        J = self.compute_jacobian(joint_angles=joint_angles, link_name=link_name)

        # compute manipulability measure
        A = torch.bmm(J, torch.transpose(J,2,1))
        det_A = torch.linalg.det(A)
        #print('detA')
        #print(det_A.shape)
        manipulability_measure = torch.sqrt(det_A)
        #print(manipulability_measure.shape)

        return det_A, manipulability_measure

    def compute_manipulability_measure_both(self, joint_angles):

        '''
        Computes Manipulability Score (log_det_man_score) for both grasp and pre-grasp
        Input:
            - joint_angles [B,7]
        Output:
            - probability: [B,1]
        '''

        # compute Jacobian
        det_A1, _ = self.compute_manipulability_measure(joint_angles)
        det_A2, _ = self.compute_manipulability_measure(joint_angles)

        log_det_man_score = det_A1+det_A2

        return log_det_man_score


def calc_man_score(joint_angles, link_name='panda_gripper_center'):
    '''
    Calculates manipulability measure force joint state configuration.
    Note: Done in batch operations
    '''
    is_tensor = True
    if not isinstance(joint_angles, torch.Tensor):
        joint_angles = torch.FloatTensor(joint_angles)
        is_tensor = False

    start_t = time.time()
    rc = RobotClassifier()
    print(f'Time taken for init of Robot_class: {time.time()-start_t}')
    
    start_t = time.time()
    with torch.no_grad():
        _, results = rc.compute_manipulability_measure(joint_angles, link_name=link_name)
    print(f'Time taken for comp: {time.time()-start_t}')

    if is_tensor:
        return results
    
    return results.numpy()




class PandaIKModel(nn.Module):
    def __init__(self):
        super(PandaIKModel, self).__init__()
        
        self.fc_t = nn.Linear(3, 16)
        self.fc_q = nn.Linear(4, 16)

        self.learner = nn.Sequential(
          nn.Linear(32,128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Linear(128,512),
          nn.BatchNorm1d(512),
          nn.ReLU(),
          nn.Linear(512,256),
          nn.BatchNorm1d(256),
          nn.ReLU(),
          nn.Linear(256,128),
          nn.BatchNorm1d(128),
          nn.ReLU(),
          nn.Linear(128,64),
          nn.BatchNorm1d(64),
          nn.ReLU(),
          nn.Linear(64,32),
          nn.BatchNorm1d(32),
          nn.ReLU(),
          nn.Linear(32,7)
        )
    def forward(self, t, q):
        '''
        Caculates inverse kinematics of Panda Robot
        Inputs:
        - t [B,3]: translations
        - q [B,4]: quaternions in [x,y,z,w]
        Outputs:
        - theta [B,7]: joint configuration of the robot
        '''

        # concat
        t = torch.relu(self.fc_t(t)) # [B,16] 
        q = torch.relu(self.fc_q(q)) # [B,16]

        x = torch.hstack([t,q])

        # all the rest
        out = self.learner(x)

        return out

class CombinedRobotModel2(nn.Module):
    def __init__(self, mi_threshold = 0.0001, mi_coeff = 10000,
                       t_threshold=0.05, q_threshold=0.2,
                       w_coeff = 10000,
                       device=None,
                       urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas_without_finger.urdf',
                       ik_module_path = "saved_models/IK/167100184736/99000.pt"
                       ):
        super(CombinedRobotModel2, self).__init__()

        self.t_threshold = t_threshold
        self.q_threshold = q_threshold
        # self.t_coeff = t_coeff
        # self.q_coeff = q_coeff
        self.w_coeff = w_coeff
        self.mi_threshold = mi_threshold
        self.mi_coeff = mi_coeff
        self.urdf_path = urdf_path
        self.device = device

        self.robot = DifferentiableFrankaPanda(device=device, urdf_path=urdf_path)

        self.ik = PandaIKModel().to(device)
        self.ik.load_state_dict(torch.load(ik_module_path))
        self.ik.eval()

        self.ik_numpy = RobotModel(30)

    def check_joint_angles(self, theta):
        assert len(theta.shape) == 2, f"joint angles must have shape of [B,7], but given {theta.shape}"
        return

    def fk(self, theta, link_name = "panda_gripper_center"):

        # fill joint angles
        zeros = torch.zeros(theta.shape[0], 2).to(theta.device)
        theta = torch.cat([theta, zeros], 1)

        return self.robot.compute_forward_kinematics(theta, link_name=link_name)

    def forward(self, t, q):

        '''
        Classify the grasp by solving inverse kinematics:
        1. If IK solution is close to true solution -> use mi classifier
        2. If IK solution is bad -> use workspace classifier

        MI classifier shows how grasp closer to singular positions
        Workspace classifier shows how grasp closer to workspace of the robot
        '''

        theta_pred = self.ik(t, q) # [B,7]
        t_pred, q_pred = self.fk(theta_pred) # [B,3] and [B,4]

        # mi threshold
        mi_logit = self.mi_logit(theta_pred, link_name="panda_gripper_center")

        # distance threshold
        t_dist = t - t_pred
        t_dist = torch.norm(t_dist, p=2, dim=1,keepdim=True)
        q_dist = quaternion_distance(q, q_pred,keepdim=True)
        w_logit = self.w_coeff*(self.t_threshold-t_dist) + self.w_coeff*(self.q_threshold-q_dist)
        #w_logit = self.w_coeff*( (self.t_threshold-t_dist) *(self.q_threshold-q_dist) )

        # choose between distance and MI ->
        ts = t.clone().detach().cpu().numpy()
        qs = q.clone().detach().cpu().numpy()
        res1, res2 = self.ik_numpy.solve_ik_batch(ts, qs)

        reachable_idx_refined = ~np.isnan(res1).any(axis=1)
        reachable_idx_sampled = ~np.isnan(res2).any(axis=1)
        reachable_mask = reachable_idx_refined & reachable_idx_sampled

        reachable_mask = torch.BoolTensor(reachable_mask).to(t.device).unsqueeze(-1)

        # finalize
        logit = w_logit

        logit[reachable_mask == True] = mi_logit[reachable_mask == True]
        return logit

    def mi_logit(self, theta, link_name="panda_gripper_center"):
        # compute manipulability_measure
        mi_score = self.mi(theta, link_name=link_name)
        mi_logit = self.mi_coeff*(mi_score - self.mi_threshold)
        return mi_logit.unsqueeze(-1)


    def jacobian(self, theta, link_name="panda_gripper_center"):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - Jacobian: [B,n,k]
        '''

        # fill joint angles
        zeros = torch.zeros(theta.shape[0], 2).to(theta.device)
        theta = torch.cat([theta, zeros], 1)

        # compute Jacobian
        J_v, J_w = self.robot.compute_endeffector_jacobian(theta, link_name=link_name)
        J = torch.hstack([J_v[:,:,:7], J_w[:,:,:7]])

        return J


    def mi(self, theta, link_name="panda_gripper_center"):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - probability: [B,1]
        '''

        # compute Jacobian
        J = self.jacobian(theta=theta, link_name=link_name)

        # compute manipulability measure
        A = torch.bmm(J, torch.transpose(J,2,1))
        det_A = torch.linalg.det(A)
        mi_score = torch.sqrt(det_A)

        return mi_score

class CombinedRobotModel(nn.Module):
    def __init__(self, mi_threshold = 0.0001, mi_coeff = 10000,
                       t_threshold=0.05, q_threshold=0.2,
                       #t_coeff=10000, q_coeff=10000,
                       w_coeff = 10000,
                       device=None,
                       urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas_without_finger.urdf',
                       ik_module_path = "saved_models/IK/167100184736/99000.pt"):
        super(CombinedRobotModel, self).__init__()

        self.t_threshold = t_threshold
        self.q_threshold = q_threshold
        # self.t_coeff = t_coeff
        # self.q_coeff = q_coeff
        self.w_coeff = w_coeff
        self.mi_threshold = mi_threshold
        self.mi_coeff = mi_coeff
        self.urdf_path = urdf_path
        self.device = device

        self.robot = DifferentiableFrankaPanda(device=device, urdf_path=urdf_path)

        self.ik = PandaIKModel().to(device)
        self.ik.load_state_dict(torch.load(ik_module_path))
        self.ik.eval()

    def check_joint_angles(self, theta):
        assert len(theta.shape) == 2, f"joint angles must have shape of [B,7], but given {theta.shape}"
        return

    def fk(self, theta, link_name = "panda_gripper_center"):

        # fill joint angles
        zeros = torch.zeros(theta.shape[0], 2).to(theta.device)
        theta = torch.cat([theta, zeros], 1)

        return self.robot.compute_forward_kinematics(theta, link_name=link_name)

    def forward(self, t, q):

        '''
        Classify the grasp by solving inverse kinematics:
        1. If IK solution is close to true solution -> use mi classifier
        2. If IK solution is bad -> use workspace classifier

        MI classifier shows how grasp closer to singular positions
        Workspace classifier shows how grasp closer to workspace of the robot
        '''

        theta_pred = self.ik(t, q) # [B,7]
        t_pred, q_pred = self.fk(theta_pred) # [B,3] and [B,4]

        mi_logit = self.mi_logit(theta_pred, link_name="panda_gripper_center")

        # compute workspace measures
        t_dist = t - t_pred
        t_dist = torch.norm(t_dist, p=2, dim=1,keepdim=True)
        q_dist = quaternion_distance(q, q_pred,keepdim=True)
        #w_logit = self.t_coeff*(self.t_threshold-t_dist) + self.q_coeff*(self.q_threshold-q_dist)
        w_logit = self.w_coeff*( (self.t_threshold-t_dist) *(self.q_threshold-q_dist) )
        w_flag = w_logit >= 0

        logit = w_logit
        logit[w_flag] = mi_logit[w_flag]

        return logit

    def mi_logit(self, theta, link_name="panda_gripper_center"):
        # compute manipulability_measure
        mi_score = self.mi(theta, link_name=link_name)
        mi_logit = self.mi_coeff*(mi_score - self.mi_threshold)
        return mi_logit.unsqueeze(-1)


    def jacobian(self, theta, link_name="panda_gripper_center"):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - Jacobian: [B,n,k]
        '''

        # fill joint angles
        zeros = torch.zeros(theta.shape[0], 2).to(theta.device)
        theta = torch.cat([theta, zeros], 1)

        # compute Jacobian
        J_v, J_w = self.robot.compute_endeffector_jacobian(theta, link_name=link_name)
        J = torch.hstack([J_v[:,:,:7], J_w[:,:,:7]])

        return J


    def mi(self, theta, link_name="panda_gripper_center"):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - probability: [B,1]
        '''

        # compute Jacobian
        J = self.jacobian(theta=theta, link_name=link_name)

        # compute manipulability measure
        A = torch.bmm(J, torch.transpose(J,2,1))
        det_A = torch.linalg.det(A)
        mi_score = torch.sqrt(det_A)

        return mi_score

if __name__ == "__main__":
    # TEST ROBOT MODEL CLASSIFIER
    robot_classifier = RobotClassifier()
    # q1 = torch.FloatTensor([[-2.17801821, -0.18684093,  1.69959079, -1.37832062,  0.79249172,  1.16310331, -1.74444444]])
    # # theoretical part
    # q2 = torch.FloatTensor([[-2.17801821, -0.03684093,  1.69959079, -1.37832062,  0.79249172,  1.16310331, -1.74444444]])
    # q = torch.cat([q1, q2], 0)
    # print(q.shape)
    # q.requires_grad_(True)
    # print('Joint states: ')
    # print(q)
    # logit = robot_classifier(q)
    # # det_A, manipulability_measure = robot_classifier.compute_manipulability_measure(q)
    # # print('det_A: ')
    # # print(det_A)
    # # print('manipulability measure:')
    # # print(manipulability_measure)
    # print('Logit:')
    # print(logit)
    # # print('Classifier output: ')
    # # print(torch.sigmoid(logit))

    # grad = logit.backward(torch.ones_like(logit))
    # print(q.grad)
    
    # a = robot_classifier.franka_differentiable_model.get_joint_limits()
    # print(a)



    q1 = torch.FloatTensor([[0.40085,1.69405,-2.43301,-0.94400,1.86013,0.68215,-0.84749]])
    q2 = torch.FloatTensor([[-0.20081667506783032, 0.6051039348652488, 0.2193740141649681, -1.4658499204199438, -0.27244959128565255, 1.637356524242295, 2.896813425071393]])

    # q1 = torch.FloatTensor([[0.50863,1.60906,-2.14228,-1.18964,1.64501,0.77869,-0.90183],])
    # q2 = torch.FloatTensor([[-0.3369170307779354, 0.513355975995984, 0.2499895794225034, -1.6123361847856283, -0.10975952648012274, 1.7844145518276424, 2.896703596091105]])

    _, a1 = robot_classifier.compute_manipulability_measure(q1)
    _, a2 = robot_classifier.compute_manipulability_measure(q2)

    print(a1)
    print(a2)

    panda_robot = PandaRobot()
    A1,B1 = panda_robot(q1)
    A2,B2 = panda_robot(q2)

    print(A1, B1)
    print(A2, B2)
