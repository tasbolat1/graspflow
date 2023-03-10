from curses import tparm
from socketserver import ThreadingTCPServer
from termios import TIOCCONS
import numpy as np
import torch
from robot_model import PandaRobot, RobotClassifier
from networks.losses import quat_chordal_squared_loss
import pandas as pd
import matplotlib.pyplot as plt
import franka_ik_pybind

MSE = torch.nn.MSELoss(reduction='none')

def load_data(fname='temp_data/data_to_test.npz'):
    data = np.load(fname)
    print(list(data.keys()))
    qs = data['q']
    ts = data['t']
    thetas = data['theta']
    torques = data['torque']
    return torch.FloatTensor(qs), torch.FloatTensor(ts), torch.FloatTensor(thetas), torch.FloatTensor(torques)


if __name__ == "__main__":

    # ######## TO TEST TORQUE SENSOR PROBLEM ########################

    # df = pd.read_csv('temp_data/robot_torque_test/robot2.csv', header=None)
    
    # q = df.values[:,0:7]
    # dq = df.values[:,7:14]
    # tau_J = df.values[:,14:21]
    # dtau_J = df.values[:,21:28]

    # print('q')
    # print(np.std(q, axis=0))
    # print('dq')
    # print(np.std(dq, axis=0))
    # print('tau_J')
    # print(np.std(tau_J, axis=0))
    # print('dtau_J')
    # print(np.std(dtau_J, axis=0))


    # fig, ax = plt.subplots(nrows=2, ncols=2)
    # ax[0,0].plot(q)
    # ax[0,0].set_title('q')
    # ax[1,0].plot(dq)
    # ax[1,0].set_title('dq')
    # ax[0,1].plot(tau_J)
    # ax[0,1].set_title('tau_J')
    # ax[1,1].plot(dtau_J)
    # ax[1,1].set_title('dtau_J')
    # fig.tight_layout()
    # fig.savefig('robot_robotiq.pdf')
    # # plt.show()


    ####### TO TEST SIM2REAL GAP ########################

    txt = 'good'
    folder_name = 'conf_no_spacing' #'ik_spacing' # conf_no_spacing
    # load the data
    qs, ts, thetas, torques = load_data(fname=f'temp_data/data_to_test_{txt}.npz')

    # define robot model
    robot = PandaRobot()
    robot_classifier = RobotClassifier()
    robot_limits = robot.franka_differentiable_model.get_joint_limits()
    
    J = robot_classifier.compute_jacobian(thetas, link_name='panda_gripper_center')
    print(J.shape)

    # check forward kinematics
    with torch.no_grad():
        ts_fk, qs_fk = robot.forward(thetas, link_name='panda_gripper_center')

        for i in range(ts_fk.shape[0]):
            ts_L1 = MSE(ts_fk[i], ts[i])
            qs_L2 = quat_chordal_squared_loss(qs_fk[i], qs[i])
            print(ts_L1, qs_L2)
        

    print(f'L1 loss for translations: {ts_L1}')
    print(f'Chordal Squared Loss for quaternions: {qs_L2}')

    # get manipulability index
    
    det_A, man_score = robot_classifier.compute_manipulability_measure(thetas)
    print(man_score)

    # check inverse kinematics
    initial_joint_angles = np.array([-0.00034485234427340453, -0.7847331501140928,
                                        -0.00048777872079729497, -2.3551600892113274,
                                        -0.0009046530319716893, 1.5725385250250496,
                                        0.5])

    print('---------------------')

    for i in range(len(ts)):
        _t = ts[i].numpy().astype(np.float64)
        _q = qs[i].numpy().astype(np.float64)
        _thetas = thetas[i].numpy().astype(np.float64)
        _q7 = _thetas[6]
        initial_joint_angles[6] = _q7
        joint_angles1 = franka_ik_pybind.franka_IK(_t, _q, _q7, initial_joint_angles)

        joint_angles2 = franka_ik_pybind.franka_IKCC(
                        _t, _q, _q7, initial_joint_angles)
        print(_t)
        print(_q)
        print(joint_angles1)
        print(_thetas)
        break
        # print(f'pose{i}')
        # print('IK')
        # print(np.sum((_thetas-joint_angles1)**2, axis=1))
        # print('IKCC')
        # print(joint_angles2)
        # print(np.sum((_thetas-joint_angles2)**2))

#     # check simulation controllers
#     sim_theta = []
#     sim_trans = []
#     sim_quat = []
#     ddd = {
#         'poses': [],
#         'trans_mse':[],
#         'quats_mse': [],
#         'theta_mse': [],
#         'torque_mse': [],
#         'jacobian_mse': [],
#     }
#     data = np.load(f'temp_data/{folder_name}/data_to_test_{txt}_0.npz')
#     print(list(data.keys()))

#     _thetas = torch.Tensor(data['thetas']).squeeze(-1)
#     _torques = torch.Tensor(data['forces']).squeeze(-1)[:,:,:7]
#     _trans = torch.Tensor(data['trans']).squeeze(-1)
#     _quats = torch.Tensor(data['quats']).squeeze(-1)
#     _J = torch.Tensor(data['jacobians']).squeeze(-1)[:,:,12,:,:7]
#     print(_J.shape)


#     for j in range(len(_trans)):
#         print(f'Pose: {j+1}')
#         q_target = qs[j].repeat(_trans.shape[1], 1)

#         t_target = ts[j].repeat(_trans.shape[1], 1)
#         theta_target = thetas[j].repeat(_trans.shape[1], 1)
#         force_target = torques[j].repeat(_trans.shape[1], 1)

#         J_target = J[j].repeat(_J.shape[1], 1, 1)


#         theta_mse = MSE(_thetas[j], theta_target)
#         #total_theta_mse.append(theta_mse)

#         torque_mse = MSE(_torques[j], force_target)
#         #total_torque_mse.append(torque_mse)

#         trans_mse = MSE(_trans[j], t_target)
#         #total_trans_mse.append(trans_mse)

#         J_mse = MSE(_J[j], J_target)

#         quat_mse = quat_chordal_squared_loss(_quats[j], q_target, reduce=False)
#         #total_quats_mse.append(quat_mse)

#         theta_mse = torch.sum(theta_mse, dim=1)
#         torque_mse = torch.sum(torque_mse, dim=1)
#         trans_mse = torch.sum(trans_mse, dim=1)
#         J_mse = torch.sum(torch.sum(J_mse, dim=1), dim=1)

#         ddd['theta_mse'].append(f'{torch.mean(theta_mse):.02} ({torch.std(theta_mse):.02})')
#         ddd['quats_mse'].append(f'{torch.mean(quat_mse):.02} ({torch.std(quat_mse):.02})')
#         ddd['torque_mse'].append(f'{torch.mean(torque_mse):.02} ({torch.std(torque_mse):.02})')
#         ddd['trans_mse'].append(f'{torch.mean(trans_mse):.02} ({torch.std(trans_mse):.02})')
#         ddd['jacobian_mse'].append(f'{torch.mean(J_mse):.02} ({torch.std(J_mse):.02})')
#         ddd['poses'].append(j+1)


# df = pd.DataFrame(data=ddd)

# df.to_csv('ddd.csv')