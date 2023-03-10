#!/usr/bin/env python3


from ast import AnnAssign
import numpy as np
from utils.auxilary import get_transform
from utils.visualization import gripper_bd
import trimesh
import rospy
from std_msgs.msg import String
from pathlib import Path
from scipy.spatial.transform import Rotation as R

class GraspFlow(object):
    def __init__(self, sampler, idx, classifier, method='GraspFlow', grasp_space='SO3', grasp_folder="../experiments/generated_grasps"):

        # initialize node
        rospy.init_node('execute_grasps', anonymous=False)
        
        # subscibe to robot control
        rospy.Subscriber('graspflow/move_to_results', String, self.resultCallback, queue_size=10)

        # publish to robot control
        self.robot_control_pub = rospy.Publisher('graspflow/move_to', String, queue_size=10)

        # local variables
        self.prefix = f'{sampler}_{classifier}_{method}_{grasp_space}'
        self.grasps_file = f'{grasp_folder}/{idx:003}_{sampler}.npz'

        self.load_grasps()

    def run(self):
         while not rospy.is_shutdown():

            # visualize next grasp

            for i in range(self.n):
                t,q,s = self.ts[i], self.qs[i], self.scores[i]

                self.visualize_grasp(t,q,s)

                print("Enter 1 to grasp, 2 to skip:")
                command = int(input())
                print(f'Commmand: {command}')

                if command == 1:
                    self.send_to_robot_control(t,q)
                    return
                elif command == 2:
                    print('Grasp is skipped')

            print("No grasp left.")

                
    def load_grasps(self, path='/home/crslab/catkin_graspflow/src/processed_data/current_data'):
        data_dir = Path(path)
        pc = np.load(data_dir / 'pc_combined_in_world_frame.npy')
        pc = pc[:, :3] # [B, 1024, 3]
        self.pc = pc
        self.pc_mesh = trimesh.points.PointCloud(self.pc)
        data = np.load(self.grasps_file)
        self.ts = data[f'{self.prefix}_grasps_translations'][0]
        self.qs = data[f'{self.prefix}_grasps_quaternions'][0]
        self.scores = data[f'{self.prefix}_scores'][0]

        # sort grasps from 
        sorted_idx = np.argsort(self.scores)[::-1]
        self.ts = self.ts[sorted_idx]
        self.qs = self.qs[sorted_idx]
        self.scores = self.scores[sorted_idx]
        self.n = self.ts.shape[0]

        print(self.scores.shape)

    def offset_grasp(self, trans, quat):
        r = R.from_quat(quat)
        rotations = r.as_matrix().reshape([1, 3, 3])

        unit_vector = np.zeros([1, 3])
        unit_vector[0, 2] = -0.11 # 127
        unit_vector = np.einsum('ik,ijk->ij', unit_vector, rotations)
        new_trans = trans + unit_vector[0]
        new_trans[2] = max(0.339, new_trans[2])

        new_quat = quat
        return new_trans, new_quat

    def send_to_robot_control(self, trans, quat):
        trans, quat = self.offset_grasp(trans, quat)

        msg = f'1 0 [[],[{trans[0]},{trans[1]},{trans[2]}],[{quat[0]},{quat[1]},{quat[2]},{quat[3]}]]'
        self.robot_control_pub.publish(msg)
        print('Following graps is sent to robot control:')
        print(f'Translation: {trans}')
        print(f'Quaternion: {quat}')

    def resultCallback(self, msg):
        print('Received from robot control:')
        print(msg.data)

    def visualize_grasp(self, trans, quat, score):
        grasp = get_transform(quat, trans)
        scene = trimesh.Scene()
        scene.add_geometry(self.pc_mesh)
        scene.add_geometry(gripper_bd(min(1, score)), transform=grasp)
        scene.show()



if __name__ == '__main__':
    graspflow = GraspFlow(sampler="graspnet", idx=1, classifier='ST')
    graspflow.run()
    rospy.spin()

