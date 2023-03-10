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

    def refine_grasps_metropolis(self, q, t):

        e = quaternions2eulers(q)
        self.info.set_isEuler(True)

        split_size = int(self.args.n/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(self.args.n), split_size)

        print(f'Start to refine using metropolis ... .')
        start_time = time.time()
        last_success =  None
        with torch.no_grad():
            for indcs in tqdm( split_indcs ):
                _e, _t = e[indcs], t[indcs]
                last_success = None
                self.info.batch_init()

                for k in tqdm( range(0, self.args.max_iterations+1), leave=False):
                    if last_success is None:
                        last_success  = self.evaluator.forward_with_eulers(_e, _t)
                        last_success = torch.sigmoid(last_success)

                    self.info.batch_update(_e, tensor_nans_like(_e), _t, tensor_nans_like(_t), last_success)

                    delta_t = 2 * (torch.rand(_t.shape, dtype=torch.float).to(self.device) - 0.5) * 0.02
                    delta_euler_angles = (torch.rand(_e.shape, dtype=torch.float).to(self.device) - 0.5) * 2
                    perturbed_t = _t.data + delta_t.float()
                    perturbed_e = _e.data + delta_euler_angles.float()
                    
                    logit  = self.evaluator.forward_with_eulers(perturbed_e, perturbed_t)
                    perturbed_success = torch.sigmoid(logit)
                    ratio = perturbed_success / torch.max(last_success,torch.tensor(0.0001).to(self.device))
                    mask = torch.rand(ratio.shape).to(self.device) <= ratio
                    #mask = torch.ones_like(perturbed_success) <= ratio # this works like magic for simple distributions (smooth space)

                    ind = torch.where(mask)[0]
                    last_success[ind] = perturbed_success[ind]
                
                    # update
                    _e[ind] = perturbed_e.data[ind].float()
                    _t[ind] = perturbed_t.data[ind].float()

                self.info.update() 
                    
        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)

        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {self.args.n} samples.')

        # show stats
        print(f'Average initial success: {self.info.compute_init_success()}.')
        print(f'Average final success: {self.info.compute_final_success()}.')

        return self.info.get_refined_grasp()

    def refine_grasps_graspnet(self, q, t):


        e = quaternions2eulers(q)
        self.info.set_isEuler(True)


        split_size = int(self.args.n/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(self.args.n), split_size)

        print(f'Start to refine using grapsnet ... .')
        start_time = time.time()
        for indcs in tqdm( split_indcs ):
            _e, _t = e[indcs], t[indcs]

            self.info.batch_init()
            for t in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient
                e_v, t_v, success = self._velocity(_e,_t, isEuler=True, isGraspnet=True)
                
                # update info
                self.info.batch_update(_e, e_v, _t, t_v, success)

                # update: ad-hoc scaling from graspnet?
                norm_t = torch.norm(t_v, p=2, dim=-1).to(self.device)
                alpha = torch.min(0.01 / norm_t, torch.tensor(1, dtype=torch.float).to(self.device)).float()
                
                _e = _e.data + e_v * alpha[:, None]
                _t = _t.data + t_v * alpha[:, None]

            self.info.update()

        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)


        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {self.args.n} samples.')

        # show stats
        print(f'Average initial success: {self.info.compute_init_success()}.')
        print(f'Average final success: {self.info.compute_final_success()}.')

        return self.info.get_refined_grasp()

    def refine_grasps_SO3(self, q, t):
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
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False, disable=True):

                # get gradient
                tic = time.time()
                q_logmap = SO3(quaternion=_q).log_map()
                toc = time.time()
                print(f"Initialization time: {toc-tic}")

                tic = time.time()
                q_logmap_v, t_v, success = self._velocity_SO3(q_logmap,_t) # q_v in log map
                toc = time.time()
                print(f"Gradient time: {toc-tic}")
                # update info

                self.info.batch_update(_q, q_logmap_v, _t, t_v, success)

                tic = time.time()
                delta_q_logmap = self.args.eta_eulers*q_logmap_v
                delta_q_logmap = torch.nan_to_num(delta_q_logmap, nan=0) # thesus error
                
                q_logmap_noise = torch.randn(q_logmap.shape[0], device=self.device, dtype=_q.dtype)
                q_logmap = q_logmap.data + delta_q_logmap + np.sqrt(2*self.args.eta_eulers) * self.args.noise_factor * q_logmap_noise
                toc = time.time()
                print(f"Update time: {toc-tic}")

                tic = time.time()
                _q = SO3().exp_map(q_logmap).to_quaternion()
                toc = time.time()
                print(f"exp_map time: {toc-tic}")
                
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

    def refine_grasps_euler(self, q, t):
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

        B = q.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        # refine using euler angles
        e = quat2euler(q, order='XYZ')

        print(f'Start to refine using GraspFlow ... ')
        start_time = time.time()
        for indcs in tqdm(split_indcs):
            _e, _t = e[indcs], t[indcs]

            e_v = None
            t_v = None
            self.info.batch_init()
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient
                e_v, t_v, success = self._velocity(_e,_t, isEuler=True, isGraspnet=False)
                # update info

                self.info.batch_update(_e, e_v, _t, t_v, success)

                # update
                delta_e = self.args.eta_eulers * e_v + np.sqrt(2*self.args.eta_eulers) * self.args.noise_factor * sample_norm(_e.shape[0], return_euler=True).to(self.device)
                _e = _e.data + delta_e

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

    def refine_grasps(self, q, t):
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

        # split into batches
        split_size = int(self.args.n/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(self.args.n), split_size)
        # refine using euler angles

        print(f'Start to refine using GraspFlow ... ')
        start_time = time.time()
        for indcs in split_indcs: #tqdm( split_indcs ):
            _q, _t = q[indcs], t[indcs]

            q_v = None
            t_v = None
            self.info.batch_init()
            for k in range(0, self.args.max_iterations+1):# tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient
                print(f'step {k} ----------------')
                q_v, t_v, success = self._velocity(_q,_t, isEuler=False, isGraspnet=False)
                q_temp, q_v = quaternion_add(_q, q_v, self.args.eta_eulers, self.args.noise_factor)
                self.info.batch_update(_q, q_v, _t, t_v, success)

                _q = q_temp

                delta_t = self.args.eta_trans * t_v + np.sqrt(2*self.args.eta_trans) * self.args.noise_factor * torch.randn_like(_t)
                _t = _t.data + delta_t

            self.info.update()

        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)

        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {self.args.n} samples.')

        # show stats
        print(f'Average initial success: {self.info.compute_init_success()}.')
        print(f'Average final success: {self.info.compute_final_success()}.')

        return self.info.get_refined_grasp()


    def evaluate_all_grasps(self, quaternions, translations, pc):
        # TODO
        pass


    def load_evaluator(self):
        model = GraspEvaluator().to(self.device)
        self.evaluator = model.eval()

    # def _velocity(self, e, t):
    #     '''
    #     Modified gradient for GraspFlow
    #     '''
    #     e_t = e.clone()
    #     t_t = t.clone()
    #     e_t.requires_grad_(True)
    #     t_t.requires_grad_(True)
    #     if e_t.grad is not None:
    #         e_t.grad.zero_()
    #     if t_t.grad is not None:
    #         t_t.grad.zero_()
            
    #     #logit = self.evaluator.forward_with_eulers(e_t, t_t, pc)
    #     logit = self.evaluator.forward(e_t, t_t)
        
    #     success = torch.sigmoid(logit) # compute success
    #     logsigmoid = torch.nn.functional.logsigmoid(logit) # compute log P(y=1|x)
        
    #     s_e = torch.ones_like(e_t.detach())
    #     s_t = torch.ones_like(t_t.detach())
        
    #     s_e.expand_as(e_t)
    #     s_t.expand_as(t_t)
    #     logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))
    #     e_grad = e_t.grad
    #     trans_grad = t_t.grad
    #     e_velocity = s_e * e_grad.data
    #     t_velocity = s_t.data * trans_grad.data
    #     return e_velocity, t_velocity, success


    # def _velocity_euler(self, e, t):
    #     '''
    #     Modified gradient for GraspFlow
    #     '''
    #     e_t = e.clone()
    #     t_t = t.clone()
    #     e_t.requires_grad_(True)
    #     t_t.requires_grad_(True)
    #     if e_t.grad is not None:
    #         e_t.grad.zero_()
    #     if t_t.grad is not None:
    #         t_t.grad.zero_()
            
    #     logit = self.evaluator.forward_with_eulers(e_t, t_t)
        
    #     success = torch.sigmoid(logit) # compute success
    #     logsigmoid = torch.nn.functional.logsigmoid(logit) # compute log P(y=1|x)
        
    #     s_e = torch.ones_like(e_t.detach())
    #     s_t = torch.ones_like(t_t.detach())
        
    #     s_e.expand_as(e_t)
    #     s_t.expand_as(t_t)
    #     logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))
    #     e_grad = e_t.grad
    #     trans_grad = t_t.grad
    #     e_velocity = s_e * e_grad.data
    #     t_velocity = s_t.data * trans_grad.data
    #     return e_velocity, t_velocity, success

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

    def _velocity(self, e, t, isEuler=True, isGraspnet=False):
        '''
        Modified gradient for GraspFlow
        '''
        e_t = e.clone()
        t_t = t.clone()
        e_t.requires_grad_(True)
        t_t.requires_grad_(True)
        if e_t.grad is not None:
            e_t.grad.zero_()
        if t_t.grad is not None:
            t_t.grad.zero_()
            
        if isEuler:
            logit = self.evaluator.forward_with_eulers(e_t, t_t)
        else:
            logit = self.evaluator.forward(e_t, t_t)

        
        success = torch.sigmoid(logit) # compute success
        
        s_e = torch.ones_like(e_t.detach())
        s_t = torch.ones_like(t_t.detach())
        
        s_e.expand_as(e_t)
        s_t.expand_as(t_t)

        if isGraspnet:
            logit.backward(torch.ones_like(logit).to(self.device))
        else:
            logsigmoid = torch.nn.functional.logsigmoid(logit) # compute log P(y=1|x)
            logsigmoid.backward(torch.ones_like(logsigmoid))

        e_grad = e_t.grad
        trans_grad = t_t.grad
        e_velocity = s_e * e_grad.data
        t_velocity = s_t.data * trans_grad.data
        return e_velocity, t_velocity, success