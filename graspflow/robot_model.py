from operator import is_
import numpy as np
import torch
import torch.nn as nn
import warnings

from differentiable_robot_model.robot_model import (
    DifferentiableRobotModel
)



class DifferentiableFrankaPanda(DifferentiableRobotModel):
    def __init__(self, device=None, urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf'):
        #rel_urdf_path = "panda_description/urdf/panda_no_gripper.urdf"
        #self.urdf_path = os.path.join(robot_description_folder, rel_urdf_path)
        #self.urdf_path = '/home/crslab/GRASP/differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf'
        self.urdf_path = urdf_path
        self.learnable_rigid_body_config = None
        self.name = "differentiable_franka_panda"
        super().__init__(self.urdf_path, self.name, device=device)




class PandaRobot(nn.Module):
    def __init__(self, device=None, urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf'):
        super(PandaRobot, self).__init__()
        self.franka_differentiable_model = DifferentiableFrankaPanda(device=device, urdf_path=urdf_path)

    def forward(self, joint_angles, link_name="panda_gripper_center"):
        zeros = torch.zeros(joint_angles.shape[0], 2).to(joint_angles.device)
        joint_angles = torch.cat([joint_angles, zeros], 1)
        translations, quaterions = self.franka_differentiable_model.compute_forward_kinematics(joint_angles, link_name=link_name)
        return translations, quaterions

class RobotClassifier(nn.Module):
    def __init__(self, threshold=0.01, coeff=10000, device=None, urdf_path='../differentiable-robot-model/diff_robot_data/panda_description/urdf/panda_hand_arm_modified_tas.urdf'):
        super(RobotClassifier, self).__init__()
        self.franka_differentiable_model = DifferentiableFrankaPanda(device=device, urdf_path=urdf_path)
        self.threshold = threshold
        self.coeff = coeff
        self.clap_max = 100

    def check_joint_angles(self, joint_angles):
        assert len(joint_angles.shape) == 2, f"joint angles must have shape of [B,7], but given {joint_angles.shape}"
        return

    def forward(self, joint_angles):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - logit: [B,1]
        '''

        # compute manipulability_measure
        det_A, manipulability_measure = self.compute_manipulability_measure(joint_angles, link_name="panda_gripper_center")
        det_A2, manipulability_measure2 = self.compute_manipulability_measure(joint_angles, link_name="fake_target")

        #print(manipulability_measure.shape)
        a = torch.stack([manipulability_measure, manipulability_measure2], dim=1)
        #print('a')
        #print(a.shape)
        manipulability_measure, _ = torch.min(a, dim=1)
        #print(manipulability_measure)

        # print('det_A:')
        # print(det_A)
        # print('manipulability_measure:')
        # print(manipulability_measure)

        # logit
        logit = self.coeff*(manipulability_measure - self.threshold)
        logit = torch.clamp(logit, max=self.clap_max)
        return logit.unsqueeze(-1)

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

    def compute_manipulability_measure(self, joint_angles, link_name="panda_gripper_center"):
        '''
        Input:
            - joint_angles [B,7]
        Output:
            - probability: [B,1]
        '''

        # fill joint angles
        zeros = torch.zeros(joint_angles.shape[0], 2).to(joint_angles.device)
        joint_angles = torch.cat([joint_angles, zeros], 1)
        self.check_joint_angles(joint_angles)
        
        # compute Jacobian
        J_v, J_w = self.franka_differentiable_model.compute_endeffector_jacobian(joint_angles, link_name=link_name)
        J = torch.hstack([J_v[:,:,:7], J_w[:,:,:7]])
        #print(J.shape)

        # compute manipulability measure
        A = torch.bmm(J, torch.transpose(J,2,1))
        det_A = torch.linalg.det(A)
        #print('detA')
        #print(det_A.shape)
        manipulability_measure = torch.sqrt(det_A)
        #print(manipulability_measure.shape)

        return det_A, manipulability_measure
        

def calc_man_score(joint_angles, link_name='panda_gripper_center'):
    '''
    Calculates manipulability measure fore joint state configuration.
    Note: Done in batch operations
    '''
    is_tensor = True
    if not isinstance(joint_angles, torch.Tensor):
        joint_angles = torch.FloatTensor(joint_angles)
        is_tensor = False

    rc = RobotClassifier()
    with torch.no_grad():
        _, results = rc.compute_manipulability_measure(joint_angles, link_name=link_name)

    if is_tensor:
        return results
    
    return results.numpy()



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

    q1 = torch.FloatTensor([[0.50863,1.60906,-2.14228,-1.18964,1.64501,0.77869,-0.90183],])
    q2 = torch.FloatTensor([[-0.3369170307779354, 0.513355975995984, 0.2499895794225034, -1.6123361847856283, -0.10975952648012274, 1.7844145518276424, 2.896703596091105]])

    _, a1 = robot_classifier.compute_manipulability_measure(q1)
    _, a2 = robot_classifier.compute_manipulability_measure(q2)

    print(a1)
    print(a2)

    panda_robot = PandaRobot()
    A1,B1 = panda_robot(q1)
    A2,B2 = panda_robot(q2)

    print(A1, B1)
    print(A2, B2)
