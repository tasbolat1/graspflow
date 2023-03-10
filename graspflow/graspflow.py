from distutils.log import info
from gettext import translation
from posixpath import split
from re import A, I
from sre_constants import SUCCESS
import torch
import numpy as np
from utils.auxilary import InfoHolder, quaternions2eulers, tensor_nans_like, eulers2quaternions
from scipy.spatial.transform import Rotation as R
import time
from tqdm.auto import tqdm
from networks.models import GraspEvaluator
from networks.quaternion import quat_normalize, euler2matrix
from robot_model import PandaRobot, RobotClassifier

class GraspFlow():
    def __init__(self, args, include_robot=False):
        '''
        GraspFlow - a framework that refines grasps using different discriminators.
        '''
        self.args = args
        self.device = args.device
        if args.device == -1:
            self.device = 'cpu'
        self.batch_size=self.args.batch_size 

        # self.info = InfoHolder() # records trajectory of refined grasps and other relevant infos.

        self.include_robot = include_robot
        if include_robot:
            self.fk_solver = PandaRobot(device=self.device)
            self.robot_classifier = RobotClassifier(device=self.device, coeff=args.robot_coeff, threshold=args.robot_threshold)
    
    def refine_grasps_metropolis(self, q, t, pc):

        e = quaternions2eulers(q)
        
        info = InfoHolder(var_names_with_grad=["translations", "quaternions", "eulers"],
                          var_names_without_grad=["grasp_scores"])


        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        print(f'Start to refine using metropolis ... .')
        
        exec_time = 0
        with torch.no_grad():
            for indcs in tqdm( split_indcs ):
                _pc, _e, _t = pc[indcs], e[indcs], t[indcs]
                last_scores = None
                info.batch_init()

                for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                    tic = time.time()
                    if last_scores is None:
                        last_scores, _, _  = self.evaluator.forward_with_eulers(_e, _t, _pc)
                        last_scores = torch.sigmoid(last_scores)
                    exec_time += time.time()-tic    
                    
                    #self.info.batch_update(_e, tensor_nans_like(_e), _t, tensor_nans_like(_t), tensor_nans_like(_e), tensor_nans_like(_e), last_scores)
                    _q = eulers2quaternions(_e.detach().cpu().numpy().copy())
                    info.batch_update(translations=_t, translations_grad=tensor_nans_like(_t),
                                     eulers = _e, eulers_grad = tensor_nans_like(_t),
                                     quaternions=_q, quaternions_grad=tensor_nans_like(torch.FloatTensor(_q)),
                                     grasp_scores=last_scores)

                    tic = time.time()
                    delta_t = 2 * (torch.rand(_t.shape, dtype=torch.float).to(self.device) - 0.5) * 0.02
                    delta_euler_angles = (torch.rand(_e.shape, dtype=torch.float).to(self.device) - 0.5) * 0.02
                    perturbed_t = _t.data + delta_t.float()
                    perturbed_e = _e.data + delta_euler_angles.float()
                    
                    logit, _, _ = self.evaluator.forward_with_eulers(perturbed_e, perturbed_t, _pc)
                    perturbed_scores = torch.sigmoid(logit)

                    ratio = perturbed_scores / torch.max(last_scores,torch.tensor(0.0001).to(self.device))
                    mask = torch.rand(ratio.shape).to(self.device) <= ratio
                    ind = torch.where(mask)[0]
                    last_scores[ind] = perturbed_scores[ind]
                
                    # update
                    _e[ind] = perturbed_e.data[ind].float()
                    _t[ind] = perturbed_t.data[ind].float()

                    exec_time += time.time() - tic

                info.update() 
                    

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data_holder['translations'][-1,...]
        qs_refined=info.data_holder['quaternions'][-1,...]
        final_scores=info.data_holder['grasp_scores'][-1,...].squeeze(-1)
        init_scores=info.data_holder['grasp_scores'][0,...].squeeze(-1)

        return ts_refined, qs_refined, init_scores, final_scores, info


    def refine_grasps_graspnet(self, q, t, pc):

        e = quaternions2eulers(q)


        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        info = InfoHolder(var_names_with_grad=["translations", "quaternions", "eulers"],
                          var_names_without_grad=["grasp_scores"])

        print(f'Start to refine using grapsnet ... .')

        exec_time = 0
        for indcs in tqdm( split_indcs ):

            _pc, _e, _t = pc[indcs], e[indcs], t[indcs]

            info.batch_init()
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient
                tic = time.time()
                e_v, t_v, scores = self._velocity(_e,_t, _pc, isGraspnet=True)
                exec_time += time.time()-tic

                # update info
                _q = eulers2quaternions(_e.detach().cpu().numpy().copy())
                info.batch_update(translations=_t, translations_grad=t_v,
                    eulers = _e, eulers_grad = e_v,
                    quaternions=_q, quaternions_grad=tensor_nans_like(torch.FloatTensor(_q)),
                    grasp_scores=scores)
                
                # update: ad-hoc scaling from graspnet?
                tic = time.time()
                norm_t = torch.norm(t_v, p=2, dim=-1).to(self.device)
                alpha = torch.min(0.01 / norm_t, torch.tensor(1, dtype=torch.float).to(self.device)).float()
                
                _e = _e.data + e_v * alpha[:, None]
                _t = _t.data + t_v * alpha[:, None]
                exec_time += time.time()-tic

            info.update()

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data_holder['translations'][-1,...]
        qs_refined=info.data_holder['quaternions'][-1,...]
        final_scores=info.data_holder['grasp_scores'][-1,...].squeeze(-1)
        init_scores=info.data_holder['grasp_scores'][0,...].squeeze(-1)

        return ts_refined, qs_refined, init_scores, final_scores, info



    def refine_grasps_theta(self, theta, pc, pc_mean):
        '''
        Refines grasp based on GraspFlow framework.
        Inputs:
            - theta [B,N,7]: theta configuration of grasps
            - pc [B,N,1024]: normalized pointcloud of the object
            - pc_mean [B,3,1024]: mean of pointcloud of the object
        Returns:
            - q_refined [B,N,4]: refined quaternions of the grasps
            - t_refined [B,N,3]: refined translations of the grasps
            - info: additional information regarding to refinement
        '''
        B = theta.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        # refine using euler angles

        info = InfoHolder(var_names_with_grad=["translations", "quaternions", "theta", "theta_robot"],
                          var_names_without_grad=["grasp_scores", "robot_scores"])

        print(f'Start to refine using GraspFlow ... ')

        exec_time = 0
        for indcs in tqdm(split_indcs):
            _pc, _pc_mean, _theta = pc[indcs], pc_mean[indcs], theta[indcs]

            info.batch_init()
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # compute gradients
                tic = time.time()
                theta_v, scores = self._velocity_theta(theta=_theta, pc=_pc, pc_mean=_pc_mean, isRobotModel=False)

                theta_robot_v = tensor_nans_like(theta_v)
                if self.include_robot:
                    theta_robot_v, robot_scores = self._velocity_theta(theta=_theta, pc=_pc, pc_mean=_pc_mean, isRobotModel=True)
                exec_time += time.time() - tic

                # update info
                with torch.no_grad():
                    _t, _q = self.fk_solver(_theta)
                    _t = _t - pc_mean
                    robot_scores = torch.sigmoid(self.robot_classifier(_theta))


                info.batch_update(translations=_t, translations_grad=tensor_nans_like(_t),
                                  quaternions=_q, quaternions_grad=tensor_nans_like(_q),
                                  theta=_theta, theta_grad=theta_v,
                                  theta_robot=_theta, theta_robot_grad=theta_robot_v,
                                  grasp_scores=scores, robot_scores=robot_scores)

                tic = time.time()
                delta_theta = self.args.eta_theta_s * theta_v + np.sqrt(2*self.args.eta_theta_s)*self.args.noise_theta*torch.randn_like(_theta)

                if self.include_robot:
                    delta_theta += self.args.eta_theta_e*theta_robot_v

                _theta = _theta.data + delta_theta
                exec_time += time.time() - tic

            info.update()

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data_holder['translations'][-1,...]
        qs_refined=info.data_holder['quaternions'][-1,...]
        final_scores=info.data_holder['grasp_scores'][-1,...].squeeze(-1)
        init_scores=info.data_holder['grasp_scores'][0,...].squeeze(-1)

        return ts_refined, qs_refined, init_scores, final_scores, info

    def refine_grasps_euler(self, q, t, pc):

        e = quaternions2eulers(q)

        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        info = InfoHolder(var_names_with_grad=["translations", "quaternions", "eulers"],
                          var_names_without_grad=["grasp_scores"])

        print(f'Start to refine using GraspFlow ... .')

        exec_time = 0
        for indcs in tqdm( split_indcs ):

            _pc, _e, _t = pc[indcs], e[indcs], t[indcs]

            info.batch_init()
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient
                tic = time.time()
                e_v, t_v, scores = self._velocity(_e,_t, _pc, isEuler=True)
                exec_time += time.time()-tic

                # update info
                _q = eulers2quaternions(_e.detach().cpu().numpy().copy())
                info.batch_update(translations=_t, translations_grad=t_v,
                    eulers = _e, eulers_grad = e_v,
                    quaternions=_q, quaternions_grad=tensor_nans_like(torch.FloatTensor(_q)),
                    grasp_scores=scores)
                
                # update:

                tic = time.time()
                delta_e = self.args.eta_e * e_v + np.sqrt(2*self.args.eta_e) * self.args.noise_e * torch.randn_like(_e)
                _e = _e.data + delta_e
                delta_t = self.args.eta_t * t_v + np.sqrt(2*self.args.eta_t) * self.args.noise_t * torch.randn_like(_t)
                _t = _t.data + delta_t
                exec_time += time.time()-tic

            info.update()

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data_holder['translations'][-1,...]
        qs_refined=info.data_holder['quaternions'][-1,...]
        final_scores=info.data_holder['grasp_scores'][-1,...].squeeze(-1)
        init_scores=info.data_holder['grasp_scores'][0,...].squeeze(-1)

        return ts_refined, qs_refined, init_scores, final_scores, info

    def refine_grasps_tactile(self, q, t, pc):

        e = quaternions2eulers(q)

        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        info = InfoHolder(var_names_with_grad=["translations", "quaternions", "eulers"],
                          var_names_without_grad=["grasp_scores"])

        print(f'Start to refine using GraspFlow ... .')

        exec_time = 0
        for indcs in tqdm( split_indcs ):

            _pc, _e, _t = pc[indcs], e[indcs], t[indcs]

            info.batch_init()
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # get gradient
                tic = time.time()
                e_v, t_v, scores = self._velocity_tactile(_e,_t, _pc)
                exec_time += time.time()-tic

                # update info
                _q = eulers2quaternions(_e.detach().cpu().numpy().copy())
                info.batch_update(translations=_t, translations_grad=t_v,
                    eulers = _e, eulers_grad = e_v,
                    quaternions=_q, quaternions_grad=tensor_nans_like(torch.FloatTensor(_q)),
                    grasp_scores=scores)
                
                # update:

                tic = time.time()
                delta_e = self.args.eta_e * e_v + np.sqrt(2*self.args.eta_e) * self.args.noise_e * torch.randn_like(_e)
                _e = _e.data + delta_e
                delta_t = self.args.eta_t * t_v + np.sqrt(2*self.args.eta_t) * self.args.noise_t * torch.randn_like(_t)
                _t = _t.data + delta_t
                exec_time += time.time()-tic

            info.update()

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data_holder['translations'][-1,...]
        qs_refined=info.data_holder['quaternions'][-1,...]
        final_scores=info.data_holder['grasp_scores'][-1,...].squeeze(-1)
        init_scores=info.data_holder['grasp_scores'][0,...].squeeze(-1)

        return ts_refined, qs_refined, init_scores, final_scores, info

    def evaluate_all_grasps(self, q, t, pc):
        '''
        Assess grasps using evaluator.
        Inputs:
            - q [B,N,4]: quaternions of grasps
            - t [B,N,3]: translations of grasps
            - pc [B,N,1024]: pointcloud of the object
        Returns:
            - success [B]: success vector
        '''


        B = q.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        all_success = []
        with torch.no_grad():
            for indcs in tqdm(split_indcs, disable=True):
                _pc, _q, _t = pc[indcs], q[indcs], t[indcs]

                success_logit, _, _  = self.evaluator.forward(_q, _t, _pc)
                success = torch.sigmoid(success_logit)
                all_success.append(success.cpu().numpy().copy())

        success = np.concatenate(all_success, axis=1).squeeze(-1)
        return success


    def evaluate_man_score_classifier(self, thetas):
        B = thetas.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        all_success = []
        with torch.no_grad():
            for indcs in tqdm(split_indcs, disable=True):
                _thetas = thetas[indcs]
                logit = self.robot_classifier(_thetas)
                success = torch.sigmoid(logit)
                all_success.append(success.cpu().numpy().copy())
        success = np.concatenate(all_success, axis=1).squeeze(-1)

        return np.nan_to_num(success)


    def evaluate_grasps(self, angles, translations, pc, eulers=True):

        with torch.no_grad():
            if eulers:
                    success, _, _ = self.evaluator.forward_with_eulers(angles, translations, pc)
                    success = torch.sigmoid(success)
            else:
                    success, _, _ = self.evaluator(angles, translations, pc)
                    success = torch.sigmoid(success)
        return success

    def load_evaluator(self, path_to_model):
        model = GraspEvaluator()
        model.load_state_dict(torch.load(path_to_model))
        model = model.to(self.device)
        self.evaluator = model.eval()

    def _velocity_theta(self, theta, pc, pc_mean, isRobotModel=False):

        pc.requires_grad_(False)
        pc_mean.requires_grad_(False)
        theta_t = theta.clone()
        theta_t.requires_grad_(True)
        if theta_t.grad is not None:
            theta_t.grad.zero_()

        if isRobotModel:
            logit = self.robot_classifier(theta_t)
        else:
            t_t, q_t = self.fk_solver(theta_t)
            t_t = t_t - pc_mean # normalize 
            logit, _, _ = self.evaluator.forward(q_t,t_t,pc)

        success = torch.sigmoid(logit) # compute success
        logsigmoid = torch.nn.functional.logsigmoid(logit)
  
        s_theta = torch.ones_like(theta_t.detach())
        s_theta.expand_as(s_theta)

        logsigmoid.backward(torch.ones_like(logit).to(self.device))
        
        theta_grad = theta_t.grad
        theta_velocity = s_theta * theta_grad.data
        return theta_velocity, success

    def _velocity(self, e, t, pc, isSO3=False, isGraspnet=False, isEuler=False):
        pc.requires_grad_(False)
        e_t = e.clone()
        t_t = t.clone()
        e_t.requires_grad_(True)
        t_t.requires_grad_(True)
        if e_t.grad is not None:
            e_t.grad.zero_()
        if t_t.grad is not None:
            t_t.grad.zero_()
            
        if isGraspnet or isEuler:
            logit, _, _ = self.evaluator.forward_with_eulers(e_t, t_t, pc)
        else:
            raise ValueError('No such configuration exists')
        
        success = torch.sigmoid(logit) # compute success
        
        s_e = torch.ones_like(e_t.detach())
        s_t = torch.ones_like(t_t.detach())
        
        s_e.expand_as(e_t)
        s_t.expand_as(t_t)

        if isGraspnet:
            logit.backward(torch.ones_like(logit).to(self.device))
        else:
            logsigmoid = torch.nn.functional.logsigmoid(logit) # compute log P(y=1|x)
            logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))

        e_grad = e_t.grad
        trans_grad = t_t.grad
        e_velocity = s_e * e_grad.data
        t_velocity = s_t.data * trans_grad.data
        print(success.shape)
        return e_velocity, t_velocity, success


    def _velocity_tactile(self, e, t, pc):
        pc.requires_grad_(False)
        cg_belief = pc.mean(1) # TODO update this based on some axis along the table plane

        e_t = e.clone()
        e_t.requires_grad_(True)
        if e_t.grad is not None:
            e_t.grad.zero_()
        t_t = t.clone()
        t_t.requires_grad_(True)
        if t_t.grad is not None:
            t_t.grad.zero_()
        
        unit_vector = torch.zeros([e.shape[0], 3])
        unit_vector[:, 2] = 0.11
        unit_vector = unit_vector.to(self.device)
        rotations = euler2matrix(e_t, 'XYZ')
        unit_vector = torch.einsum('ik,ijk->ij', unit_vector, rotations)
            
        dist = (t_t + unit_vector - cg_belief).pow(2).sum(1).sqrt()
        logit = 1000 * (0.005 - dist)
        
        success = torch.sigmoid(logit) # compute success
        
        s_e = torch.ones_like(e_t.detach())
        s_t = torch.ones_like(t_t.detach())
        
        s_e.expand_as(e_t)
        s_t.expand_as(t_t)

        logsigmoid = torch.nn.functional.logsigmoid(logit) # compute log P(y=1|x)
        logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))

        e_grad = e_t.grad
        e_velocity = s_e * e_grad.data
        trans_grad = t_t.grad
        t_velocity = s_t.data * trans_grad.data
        return e_velocity, t_velocity, success.reshape((success.shape[0], 1))
