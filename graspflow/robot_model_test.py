import numpy as np
import torch
from robot_model import PandaRobot, RobotClassifier
from networks import losses
from scipy.spatial.transform import Rotation as R
from networks.quaternion import euler2quat, quat2euler, quaternion_mult


from robot_model import PandaIKModel, CombinedRobotModel

ik = PandaIKModel()

# trans_pre:
def move_backward_grasp(quat,t, standoff):
    '''
    Moves the grasp to standoff translation in q direction
    '''
    trans = np.eye(4)
    trans[:3,:3] = R.from_quat(quat).as_matrix()
    trans[:3,3] = t

    standoff_mat = np.eye(4)
    standoff_mat[2] = -standoff
    new_transform = np.matmul(trans,standoff_mat)

    return new_transform[:3,3]


# vars
HOME_POSE = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]
# HOME_POSE = [2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973]

# robot_model = CombinedRobotModel()


# fk_solver = PandaRobot()
# # robot_classifier = RobotClassifier(coeff=10000, threshold=0.01)

# # # theta = torch.FloatTensor([[0.60085,1.69405,-2.43301,-0.94400,1.86013,0.68215,-0.84749]])
# theta = torch.FloatTensor([HOME_POSE])

# t, q = fk_solver(theta)

# print(t)
# print(q)
# t = torch.FloatTensor([[0.70, 0.0, 0],
#                         [0.222, 0.3333, 0.4444]])
# q = torch.FloatTensor([[1, 0, 0, 0], [1, 0, 0, 0]])


# q = R.random().as_quat()
_q = R.random().as_quat()
for sss in ['xyz','yzx','zxy','xzy','zyx','yxz'] :
    sss = sss.upper()
    print(sss)


    e = R.from_quat(_q).as_euler(seq=sss)
    #print('Truth')
    print(e)
    print(_q)
    #print(e)

    #print('Torch:')
    e = torch.FloatTensor(e).unsqueeze(0)
    q = euler2quat(e, order=sss)
    print(q.squeeze(0).numpy())
    e = quat2euler(q, order='XYZ')
    print(e)

# logit = robot_model(t,q)
# print(logit)
# print(torch.sigmoid(logit))

# ik.load_state_dict(torch.load("saved_models/IK/167095731284/38000.pt"))
# ik.eval()

# print(t)
# print(q)

# with torch.no_grad():
#     theta_pred = ik(t, q)

# print(theta_pred)

# t_pred, q_pred = fk_solver(theta_pred)

# t_loss = torch.nn.functional.mse_loss(t_pred, t)
# q_loss = losses.quat_chordal_squared_loss(q_pred, q)

# print(t_pred)
# print(q_pred)

# print(t_loss)
# print(q_loss)


# uncertainty -> how?


# theta = torch.FloatTensor([[torch.nan]*7])

# print(theta)

# t, q = fk_solver.forward(theta, link_name="panda_gripper_center")
# # _, mi = robot_classifier.compute_manipulability_measure(theta)
# # logit = robot_classifier(theta)
# print(t)
# print(q)
# t, q = fk_solver.forward(theta, link_name="fake_target")

# print(t)
# print(q)
# # print(t)
# print(q)
# print(mi)
# print(logit)

# n_dof = robot_classifier.franka_differentiable_model._n_dofs - 2 # for Franka only
# lower_limits = []
# upper_limits = []

# for i in range(n_dof):
#     lower_limits.append( robot_classifier.franka_differentiable_model.get_joint_limits()[i]['lower'] )
#     upper_limits.append( robot_classifier.franka_differentiable_model.get_joint_limits()[i]['upper'] )
    
# print(lower_limits)
# print(upper_limits)



# trans_target = torch.FloatTensor([[0.307222490662, -0.000271922003633, 0.590204374766]])
# quats_target = torch.FloatTensor([[-0.999999865143, 9.54258731421e-05, -0.000480403248173, 0.000172685233533]])
# alpha_trans = 0.5
# theta_home = torch.FloatTensor([HOME_POSE])

# theta = torch.nn.Parameter(theta_home.clone())
# theta_pre = torch.nn.Parameter(theta_home.clone())


# # optimizer = torch.optim.SGD(params = [theta, theta_pre], lr=0.1)
# optimizer = torch.optim.SGD(params = [theta], lr=0.1)

# for epoch in range(100):

#     optimizer.zero_grad()

#     trans, quats = fk_solver.forward(theta)
    
#     # define loss
#     loss_quat = losses.quat_chordal_squared_loss(quats, quats_target)
#     loss_trans = torch.nn.functional.mse_loss(trans, trans_target)
#     loss = alpha_trans*loss_trans + (1-alpha_trans)*loss_quat
#     loss.backward()

#     optimizer.step()


# with torch.no_grad():
#     _, mi1 = robot_classifier.compute_manipulability_measure(theta)
#     print(mi1)

# print(theta)
# print(loss)

# trans_pre = move_backward_grasp(quats.cpu().numpy(), trans.cpu().numpy(), standoff=0.1)
    

# for epoch in range(100):

#     optimizer.zero_grad()

#     trans, quats = fk_solver.forward(theta)
    
#     # define loss
#     loss_quat = losses.quat_chordal_squared_loss(quats, quats_target)
#     loss_trans = torch.nn.functional.mse_loss(trans, trans_pre)
#     loss = alpha_trans*loss_trans + (1-alpha_trans)*loss_quat
#     loss.backward()

#     optimizer.step()

# trans_target_pre = move_backward_grasp(quats_target.cpu().numpy(), trans_target.cpu().numpy(), standoff=0.1)
# trans_target_pre = torch.FloatTensor(trans_target_pre).unsqueeze(0)
# for epoch in range(100):

#     optimizer.zero_grad()

#     trans, quats = fk_solver.forward(theta, link_name="panda_gripper_center")
#     trans_pre, quats_pre = fk_solver.forward(theta_pre, link_name="fake_target")
    
#     # define loss
#     loss_quat = losses.quat_chordal_squared_loss(quats, quats_target)
#     loss_quat_pre = losses.quat_chordal_squared_loss(quats_pre, quats_target)

#     loss_trans = torch.nn.functional.mse_loss(trans, trans_target)
#     loss_trans_pre = torch.nn.functional.mse_loss(trans_pre, trans_target_pre)


#     loss = alpha_trans*loss_trans + (1-alpha_trans)*loss_quat + alpha_trans*loss_trans_pre + (1-alpha_trans)*loss_quat_pre
#     loss.backward()

#     optimizer.step()


# print(theta)
# print(theta_pre)



# print(trans)
# print(quats)

# print(trans_pre)
# print(quats_pre)

# _t = torch.sqrt( torch.sum((trans_target - trans_target_pre)**2) )
# _t = torch.sqrt( torch.sum((trans - trans_pre)**2) )


# print(_t)



