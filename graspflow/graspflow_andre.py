import torch
import numpy as np
from utils.auxilary import InfoHolder, quaternions2eulers, tensor_nans_like
import time
from tqdm.auto import tqdm
from networks.models import GraspEvaluatorDistance as GraspEvaluator
from networks.quaternion import sample_norm, quat2euler, quaternion_mult, quat_normalize, slerp, quaternion_add
from theseus.geometry import SO3

class GraspFlow():
    def __init__(self, args):
        '''
        GraspFlow - a framework that refines grasps using different discriminators.
        '''
        self.args = args
        self.device = args.device
        if args.device == -1:
            self.device = 'cpu'
        self.batch_size=args.batch_size

        self.info = InfoHolder() # records trajectory of refined grasps and other relevant infos.

    def refine_grasps_SO3(self, q, t):
        '''
        TODO to ANDRE: can u check why this is slow?
        Refines grasp based on GraspFlow framework.
        Inputs:
            - q [B,N,4]: quaternions of grasps
            - t [B,N,3]: translations of grasps
            - pc [B,N,1024]: pointcloud of the object
        Returns:
            - q_refined [B,N,4]: refined quaternions of the grasps
            - t_refined [B,N,3]: refined translations of the grasps
            - info: additional information regarding to refinement
        '''

        #self.info.set_isEuler(True)

        B = q.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        # refine using euler angles

        print(f'Start to refine using GraspFlow ... ')
        start_time = time.time()
        for indcs in tqdm(split_indcs):
            _q, _t = q[indcs], t[indcs]


            self.info.batch_init()
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient

                q_logmap = SO3(quaternion=_q).log_map()
                q_logmap_v, t_v, success = self._velocity_SO3(q_logmap,_t) # q_v in log map
                # update info

                self.info.batch_update(_q, q_logmap_v, _t, t_v, success)

                delta_q_logmap = self.args.eta_eulers*q_logmap_v

                q_logmap_noise = SO3().randn(q_logmap.shape[0], device=self.device, dtype=_q.dtype).log_map()

                delta_q_logmap = torch.nan_to_num(delta_q_logmap, nan=0) # thesus error

                q_logmap = q_logmap.data + delta_q_logmap + np.sqrt(2*self.args.eta_eulers) * self.args.noise_factor * q_logmap_noise

                _q = SO3().exp_map(q_logmap).to_quaternion()
                
                _q = quat_normalize(_q) # add-hoc: doesn't hurt much, usually very close to 1 anyways
                

                delta_t = self.args.eta_trans * t_v + np.sqrt(2*self.args.eta_trans) * self.args.noise_factor * torch.randn_like(_t)
                _t = _t.data + delta_t

            self.info.update()

        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)

        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {B} samples.')

        # show stats
        print(f'Average initial success: {self.info.compute_init_success()}.')
        print(f'Average final success: {self.info.compute_final_success()}.')

        return self.info.get_refined_grasp()

    def refine_grasps_joint_space(self, thetas):
        '''
        Refines grasp based on GraspFlow framework.
        Inputs:
            - q [B,N,4]: quaternions of grasps
            - t [B,N,3]: translations of grasps
            - pc [B,N,1024]: pointcloud of the object
        Returns:
            - q_refined [B,N,4]: refined quaternions of the grasps
            - t_refined [B,N,3]: refined translations of the grasps
            - info: additional information regarding to refinement
        '''

        self.info.set_isEuler(True)

        B = thetas.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        # refine using euler angles

        print(f'Start to refine using GraspFlow ... ')
        start_time = time.time()
        for indcs in tqdm(split_indcs):
            _thetas, = thetas[indcs]

            e_v = None
            t_v = None
            self.info.batch_init()
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient
                thetas_v, success = self._velocity_joint(_thetas, isEuler=True, isGraspnet=False)
                # update info

                self.info.batch_update(_thetas, thetas_v, tensor_nans_like(thetas), tensor_nans_like(_thetas), success)

                # update
                delta_theta = self.args.eta * thetas_v + np.sqrt(2*self.args.eta) * self.args.noise_factor * torch.randn_like(_thetas)
                _thetas = _thetas.data + delta_theta

            self.info.update()

        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)

        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {B} samples.')

        # show stats
        print(f'Average initial success: {self.info.compute_init_success()}.')
        print(f'Average final success: {self.info.compute_final_success()}.')

        return self.info.get_refined_grasp()


    def load_evaluator(self):
        model = GraspEvaluator().to(self.device)
        self.evaluator = model.eval()

    def load_evaluator_joint_space(self):
        # TODO to ANDRE: load to joint space
        pass

    def _velocity_SO3(self, q_logmap, t):
        '''
        Modified gradient for GraspFlow
        '''
        q_logmap_t = q_logmap.clone()
        t_t = t.clone()
        q_logmap_t.requires_grad_(True)
        t_t.requires_grad_(True)
        if q_logmap_t.grad is not None:
            q_logmap_t.grad.zero_()
        if t_t.grad is not None:
            t_t.grad.zero_()
        

        q_in_t = SO3().exp_map(q_logmap_t).to_quaternion()
        logit = self.evaluator.forward(q_in_t, t_t)
        
        success = torch.sigmoid(logit) # compute success
        
        s_e = torch.ones_like(q_logmap_t.detach())
        s_t = torch.ones_like(t_t.detach())
        
        s_e.expand_as(q_logmap_t)
        s_t.expand_as(t_t)

    
        logsigmoid = torch.nn.functional.logsigmoid(logit) # compute log P(y=1|x)
        logsigmoid.backward(torch.ones_like(logsigmoid))

        q_logmap_grad = q_logmap_t.grad
        trans_grad = t_t.grad
        q_logmap_velocity = s_e * q_logmap_grad.data
        t_velocity = s_t.data * trans_grad.data
        return q_logmap_velocity, t_velocity, success

    def _velocity_joints_space(self, thetas, isEuler=True, isGraspnet=False):
        '''
        Modified gradient for GraspFlow
        '''
        thetas_t = thetas.clone()
       
        if thetas_t.grad is not None:
            thetas_t.grad.zero_()

        # TODO to ANDRE: forward your evaluator
        logit = self.evaluator_ANDRE.forward(thetas_t)

        
        success = torch.sigmoid(logit) # compute success
        
        s_thetas = torch.ones_like(thetas_t.detach())
        s_thetas.expand_as(thetas_t)

        logsigmoid = torch.nn.functional.logsigmoid(logit) # compute log P(y=1|x)
        logsigmoid.backward(torch.ones_like(logsigmoid))

        thetas_grad = thetas_t.grad
        thetas_velocity = s_thetas * thetas_grad.data
        return thetas_velocity, success