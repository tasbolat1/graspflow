import numpy as np
import torch
import franka_ik_pybind
from scipy.spatial.transform import Rotation as R


PRE_GRASP_Z_OFFSET = 0.05 # [cm]

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

class RobotModel():
    def __init__(self, angle_iterations=10):
        self.initialJointPosition = np.array([-0.00034485234427340453, -0.7847331501140928,
                                    -0.00048777872079729497, -2.3551600892113274,
                                    -0.0009046530319716893, 1.5725385250250496,
                                    0.0])
        self.angle_iterations = angle_iterations

        self.none_var = np.array([np.nan]*7)

    def solve_ik(self, t, q):

        q7_angles = np.linspace(-3.12, 3.12, self.angle_iterations, dtype=np.float64)

        for q7 in q7_angles:
            self.initialJointPosition[6] = q7
            
            jointPositionAnalytical = franka_ik_pybind.franka_IK(
                    t, q, 
                    q7, self.initialJointPosition) # [4,7]

            pre_t = move_backward_grasp(q, t, standoff=PRE_GRASP_Z_OFFSET)


            for theta_ik in jointPositionAnalytical:
                if np.sum(np.isnan(theta_ik)) > 0:
                    continue

                jointPositionAnalyticalIKCC = franka_ik_pybind.franka_IKCC(
                    pre_t.astype(np.float64), q.astype(np.float64), 
                    q7, theta_ik.astype(np.float64))

                
                if np.sum(np.isnan(jointPositionAnalyticalIKCC)) == 0:
                    return theta_ik, jointPositionAnalyticalIKCC
            
        return self.none_var, self.none_var

    def solve_ik_pre_only(self, t, q, theta_ik):

        pre_t = move_backward_grasp(q, t, standoff=PRE_GRASP_Z_OFFSET) # 10 cm back
        q7 = theta_ik[6]

        jointPositionAnalyticalIKCC = franka_ik_pybind.franka_IKCC(
                    pre_t.astype(np.float64), q.astype(np.float64), 
                    q7, theta_ik.astype(np.float64))
        
        if np.sum(np.isnan(jointPositionAnalyticalIKCC)) == 0:
            return jointPositionAnalyticalIKCC

        return self.none_var

    def solve_ik_pre_only_batch(self, t, q, theta_ik):

        t=t.astype(np.float64)
        q=q.astype(np.float64)
        theta_ik = theta_ik.astype(np.float64)

        if not t.flags['F_CONTIGUOUS']:
            t=np.asfortranarray(t, dtype=np.float64)

        res_ik_fake_target = []
        for i in range(t.shape[0]):
            
            _t = t[i].copy()
            _q = q[i].copy()
            _theta_ik = theta_ik[i].copy()
            
            _ik_fake_target = self.solve_ik_pre_only(_t,_q,_theta_ik)
            
            res_ik_fake_target.append(_ik_fake_target)

        res_ik_fake_target = np.array(res_ik_fake_target)
        
        return res_ik_fake_target


    def solve_ik_batch(self, t, q):

        t=t.astype(np.float64)
        q=q.astype(np.float64)

        if t.flags['F_CONTIGUOUS']:
            t=np.asfortranarray(t, dtype=np.float64)
   
        res_ik = []
        res_ik_fake_target = []
        for i in range(t.shape[0]):
            
            _t = t[i].copy()
            _q = q[i].copy()
            
            _ik, _ik_fake_target = self.solve_ik(_t,_q)
            
            #print(_ik.shape, _ik_fake_target.shape)
            res_ik.append(_ik)
            res_ik_fake_target.append(_ik_fake_target)

        res_ik = np.array(res_ik)
        res_ik_fake_target = np.array(res_ik_fake_target)
        
        return res_ik, res_ik_fake_target

    def solve_ik_batch2(self, H):

        t = H[:,:3,3].astype(np.float64)
        q = R.from_matrix(H[:,:3,:3]).as_quat().astype(np.float64)

        if t.flags['F_CONTIGUOUS']:
            t=np.asfortranarray(t, dtype=np.float64)

        t = t.astype(np.float64)
        q = q.astype(np.float64)

        return self.solve_ik_batch(t,q)


if __name__ == "__main__":
    # translation = np.array([0.5588969230651855,  0.0063073597848415375, 0.2631653845310211])
    # quaternion = np.array([-0.5149032426188047, 0.7359848969492421, 0.31914526066961735, -0.302236967949614]) # xyzw

    import time

    qs = np.array([ [-0.999999865143, 9.54258731421e-05, -0.000480403248173, 0.000172685233533],
                [-0.738318296604, 0.674231590111, 0.0171578341705, -0.00186132305569],
                [-0.858459739539, 0.512764924469, -0.00519832061567, -0.00959089607715],
                [-0.960702225441, 0.277499661288, 0.00221213580555, -0.0063465324108],
                [-0.783115366439, 0.621639886936, -0.00241345964736, -0.0169808430356]
    ])

    ts = np.array([ [0.307222490662, -0.000271922003633, 0.590204374766],
                    [0.468916078118, -0.0269388202601, 0.354734500724],
                    [0.414851960638, -0.266023402216, 0.348325552842],
                    [0.488354602468, -0.00608585584512, 0.338575462306],
                    [0.573745182558, 0.0444970969179, 0.336848838267]
    ])


    qs = np.array([[ 0.76147866,  0.56797338,  0.29964933, -0.08812968], [ 0.73960179,  0.60007006,  0.2843841,  -0.10968477]])
    ts = np.array([[0.67488712, 0.34649354, 0.59040132], [0.6590572,  0.33425328, 0.58144983]])
    robot_model = RobotModel(30)

    # print(robot_model.solve_ik(qs[0], ts[0]))
    start_time = time.time()
    res1, res2 = robot_model.solve_ik_batch(ts, qs)
    end_time = time.time() - start_time
    print(end_time)
    print(res1)
    print(res2)
    