
import torch
import numpy as np
from utils.auxilary import InfoHolder, quaternions2eulers, tensor_nans_like, eulers2quaternions
import time
from tqdm.auto import tqdm
from networks.models import GraspEvaluator, GraspEvaluatorDistance
from networks.quaternion import quat_normalize, euler2quat
from theseus.geometry import SO3
from robot_model import CombinedRobotModel
from svgd import SVGD 
import gpytorch
from mtadam import MTAdam

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

        self.include_robot = include_robot
        self.robot = CombinedRobotModel(device=self.device, mi_coeff=args.robot_coeff, mi_threshold=args.robot_threshold)
        self.SO3_holder = SO3() # this makes faster
        self.table = GraspEvaluatorDistance(z_threshold=0.175, angle_threshold=1.2, coeff=1000).to(self.device)

    def refine_grasps_metropolis(self, q, t, pc):

        e = quaternions2eulers(q)

        info = InfoHolder(var_names=["translations", "quaternions", "eulers", "grasp_scores"])

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
                    info.batch_update(translations=_t, eulers = _e, quaternions=_q, grasp_scores=last_scores)

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
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]
        final_scores=info.data['grasp_scores'][-1,...].squeeze(-1)
        init_scores=info.data['grasp_scores'][0,...].squeeze(-1)

        return ts_refined, qs_refined, init_scores, final_scores, info


    def refine_grasps_graspnet(self, q, t, pc):

        e = quaternions2eulers(q)


        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        info = InfoHolder(var_names=["translations", "quaternions", "eulers",
            "translations_grad", "eulers_grad", "grasp_scores"])

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
                    eulers = _e, eulers_grad = e_v, quaternions=_q,
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
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]
        final_scores=info.data['grasp_scores'][-1,...].squeeze(-1)
        init_scores=info.data['grasp_scores'][0,...].squeeze(-1)

        return ts_refined, qs_refined, init_scores, final_scores, info

    def refine_grasps_GrapsFlow(self, q, t, pc, pc_mean, classifiers="S"):

        e = quaternions2eulers(q)

        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        info = InfoHolder(var_names=["translations", "quaternions", "rots",
                                     "translation_S_grad", "translation_E_grad", "translation_T_grad",
                                     "rots_S_grad", "rots_E_grad", "rots_T_grad",
                                     "grasp_scores", "robot_scores", "table_scores",
                                     "rots_S_grad_norm", "translation_S_grad_norm",
                                     "rots_T_grad_norm", "translation_T_grad_norm"])

        print(f'Start to refine using GraspFlow ... .')
       
        K_rots = gpytorch.kernels.RBFKernel().to(self.device)
        K_rots.lengthscale = torch.FloatTensor([[0.05]]).to(self.device)

        K_trans = gpytorch.kernels.RBFKernel().to(self.device)
        K_trans.lengthscale = torch.FloatTensor([[0.05]]).to(self.device)

        exec_time = 0
        for indcs in tqdm( split_indcs ):

            _pc, _e, _t, _q, _pc_mean = pc[indcs], e[indcs], t[indcs], q[indcs], pc_mean[indcs]

            if self.args.grasp_space == 'SO3':
                tic = time.time()
                _e = SO3(quaternion=_q).log_map()
                exec_time +=time.time() -tic

            e_S, t_S, e_E, t_E, e_T, t_T = None, None, None, None, None, None  
            e_S_grad, t_S_grad, e_E_grad, t_E_grad, e_T_grad, t_T_grad = 0, 0, 0, 0, 0, 0 # init gradients
            num_classifiers = len(classifiers)
            s_sign, e_sign, t_sign = 0, 0, 0
             

            e_S = torch.nn.Parameter(_e.clone())
            t_S = torch.nn.Parameter(_t.clone())
            e_E = torch.nn.Parameter(_e.clone())
            t_E = torch.nn.Parameter(_t.clone())
            e_T = torch.nn.Parameter(_e.clone())
            t_T = torch.nn.Parameter(_t.clone())

            optimizer = torch.optim.SGD([
                    {'params': [t_S], 'lr': num_classifiers*self.args.eta_t, 'momentum': 0.0},
                    {'params': [e_S], 'lr': num_classifiers*self.args.eta_e, 'momentum': 0.0},
                    {'params': [e_E], 'lr': num_classifiers*self.args.eta_e, 'momentum': 0.0},
                    {'params': [t_E], 'lr': num_classifiers*self.args.eta_t, 'momentum': 0.0},
                    {'params': [e_T], 'lr': num_classifiers*self.args.eta_table_e, 'momentum': 0.0},
                    {'params': [t_T], 'lr': num_classifiers*self.args.eta_table_t, 'momentum': 0.0}
                ], maximize=True)

            info.batch_init()

            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # init scores
                scores, robot_scores, table_scores = None, None, None
                
                # zero gradients
                optimizer.zero_grad()
                
                # get gradient
                tic = time.time()

                logsigmoid_S = logsigmoid_E = logsigmoid_T = 0

                if "S" in classifiers:
                    scores, logsigmoid_S = self.compute_logit(e_S, t_S, _pc, _pc_mean, classifier="S")
                    s_sign = 1
                if "E" in classifiers:
                    robot_scores, logsigmoid_E = self.compute_logit(e_E, t_E, _pc, _pc_mean, classifier="E")
                    e_sign = 1
                if "T" in classifiers:
                    table_scores, logsigmoid_T = self.compute_logit(e_T, t_T, _pc, _pc_mean, classifier="T")
                    t_sign = 1


                logsigmoid = logsigmoid_S + logsigmoid_E + logsigmoid_T
                logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))


                # compute phi function for particles

                if self.args.use_svgd:
                    e_S_grad, t_S_grad, K_rr, K_tt, _temp_rr, _temp_tt, grad_K_r, grad_K_t = self.compute_phi(K_rots, K_trans, e_S, t_S, classifier="S")
            
                # normalize gradients:
                if "S" in classifiers:
                    e_S_grad, t_S_grad, e_S_grad_norm, t_S_grad_norm = self.normalize_gradients(e_S, t_S)
                if "E" in classifiers:
                    e_E_grad, t_E_grad  = self.normalize_gradients(e_E, t_E)
                if "T" in classifiers:
                    e_T_grad, t_T_grad, e_T_grad_norm, t_T_grad_norm = self.normalize_gradients(e_T, t_T)

                exec_time += time.time()-tic

                # update info

                if self.args.grasp_space == 'SO3':
                    q_t = self.SO3_holder.exp_map(e_S).to_quaternion()
                if self.args.grasp_space == 'Euler':
                    q_t = eulers2quaternions(e_S.detach().cpu().numpy().copy())

                if robot_scores is None:
                    robot_scores = torch.sigmoid(self.robot(t_S, q_t))
                if table_scores is None:
                    table_scores = torch.sigmoid(self.table(q_t, t_S))
                
                info.batch_update(translations=t_S, quaternions=q_t, rots=e_S,
                                  translation_S_grad = t_S_grad, rots_S_grad = e_S_grad,
                                  translation_E_grad = t_E_grad, rots_E_grad = e_E_grad,
                                  translation_T_grad = t_T_grad, rots_T_grad = e_T_grad,
                                  grasp_scores=scores,
                                  robot_scores=robot_scores,
                                  table_scores=table_scores,
                                  rots_S_grad_norm=e_S_grad_norm, translation_S_grad_norm=t_S_grad_norm,
                                  rots_T_grad_norm=e_T_grad_norm, translation_T_grad_norm=t_T_grad_norm)
                
                # update variables and add noise
                tic = time.time()
                optimizer.step() # updates gradients everywhere
                

                with torch.no_grad():

                    # add noise
                    t_noise_term = num_classifiers*np.sqrt(2*self.args.eta_t) * self.args.noise_t * torch.randn_like(t_S)
                    e_noise_term = num_classifiers*np.sqrt(2*self.args.eta_e) * self.args.noise_e * torch.randn_like(e_S)

                    t_temp=(s_sign*t_S + e_sign*t_E + t_sign*t_T + t_noise_term)/num_classifiers
                    e_temp=(s_sign*e_S + e_sign*e_E + t_sign*e_T + e_noise_term)/num_classifiers

                    e_S.copy_(e_temp)
                    e_E.copy_(e_temp)
                    e_T.copy_(e_temp)
                    t_S.copy_(t_temp)
                    t_E.copy_(t_temp)
                    t_T.copy_(t_temp)

                exec_time += time.time()-tic

                

            info.update()

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]

        return ts_refined, qs_refined, info


    def refine_grasps_GrapsFlow_test(self, q, t, pc, pc_mean, classifiers="S"):

        e = quaternions2eulers(q)

        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        info = InfoHolder(var_names=["translations", "quaternions", "rots",
                                     "translation_S_grad", "translation_E_grad", "translation_T_grad",
                                     "rots_S_grad", "rots_E_grad", "rots_T_grad",
                                     "grasp_scores", "robot_scores", "table_scores"])

        print(f'Start to refine using GraspFlow ... .')

       

        exec_time = 0
        for indcs in tqdm( split_indcs ):

            _pc, _e, _t, _q, _pc_mean = pc[indcs], e[indcs], t[indcs], q[indcs], pc_mean[indcs]

            if self.args.grasp_space == 'SO3':
                tic = time.time()
                _e = SO3(quaternion=_q).log_map()
                exec_time +=time.time() -tic

            e_S, t_S, e_E, t_E, e_T, t_T = None, None, None, None, None, None  
            e_S_grad, t_S_grad, e_E_grad, t_E_grad, e_T_grad, t_T_grad = 0, 0, 0, 0, 0, 0 # init gradients
            num_classifiers = len(classifiers)
            s_sign, e_sign, t_sign = 0, 0, 0
             

            e_S = torch.nn.Parameter(_e.clone())
            t_S = torch.nn.Parameter(_t.clone())
            e_E = torch.nn.Parameter(_e.clone())
            t_E = torch.nn.Parameter(_t.clone())
            e_T = torch.nn.Parameter(_e.clone())
            t_T = torch.nn.Parameter(_t.clone())

            optimizer = torch.optim.Adam([
                    {'params': [e_S], 'lr': num_classifiers*self.args.eta_e},
                    {'params': [t_S], 'lr': num_classifiers*self.args.eta_t},
                    {'params': [e_E], 'lr': num_classifiers*self.args.eta_e},
                    {'params': [t_E], 'lr': num_classifiers*self.args.eta_t},
                    {'params': [e_T], 'lr': num_classifiers*self.args.eta_table_e},
                    {'params': [t_T], 'lr': num_classifiers*self.args.eta_table_t}
                ], maximize=True, amsgrad=True)

            info.batch_init()

            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # init scores
                scores, robot_scores, table_scores = None, None, None
                
                # zero gradients
                optimizer.zero_grad()
                
                # get gradient
                tic = time.time()

                logsigmoid_S = logsigmoid_E = logsigmoid_T = 0

                if "S" in classifiers:
                    scores, logsigmoid_S = self.compute_logit(e_S, t_S, _pc, _pc_mean, classifier="S")
                    s_sign = 1
                if "E" in classifiers:
                    robot_scores, logsigmoid_E = self.compute_logit(e_E, t_E, _pc, _pc_mean, classifier="E")
                    e_sign = 1
                if "T" in classifiers:
                    table_scores, logsigmoid_T = self.compute_logit(e_T, t_T, _pc, _pc_mean, classifier="T")
                    t_sign = 1


                logsigmoid = logsigmoid_S + logsigmoid_E + logsigmoid_T
                logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))

                exec_time += time.time()-tic

                # ad hoc: check for nans
                if "S" in classifiers:
                    e_S.grad = torch.nan_to_num(e_S.grad, nan=0) # thesus error: adhoc!
                if "T" in classifiers:
                    e_T.grad = torch.nan_to_num(e_T.grad, nan=0) # thesus error: adhoc!
                if "E" in classifiers:
                    e_E.grad = torch.nan_to_num(e_E.grad, nan=0) # thesus error: adhoc!


                # update info

                if self.args.grasp_space == 'SO3':
                    q_t = self.SO3_holder.exp_map(e_S).to_quaternion()
                if self.args.grasp_space == 'Euler':
                    q_t = eulers2quaternions(e_S.detach().cpu().numpy().copy())

                if robot_scores is None:
                    robot_scores = torch.sigmoid(self.robot(t_S, q_t))
                if table_scores is None:
                    table_scores = torch.sigmoid(self.table(q_t, t_S))
                
                info.batch_update(translations=t_S, quaternions=q_t, rots=e_S,
                                  translation_S_grad = t_S.grad.data, rots_S_grad = e_S.grad.data,
                                  translation_E_grad = t_E_grad, rots_E_grad = e_E_grad,
                                  translation_T_grad = t_T_grad, rots_T_grad = e_T_grad,
                                  grasp_scores=scores,
                                  robot_scores=robot_scores,
                                  table_scores=table_scores)
                
                # update variables and add noise
                tic = time.time()
                optimizer.step() # updates gradients everywhere
                

                with torch.no_grad():

                    # add noise
                    t_noise_term = num_classifiers*np.sqrt(2*self.args.eta_t) * self.args.noise_t * torch.randn_like(t_S)
                    e_noise_term = num_classifiers*np.sqrt(2*self.args.eta_e) * self.args.noise_e * torch.randn_like(e_S)

                    t_temp=(s_sign*t_S + e_sign*t_E + t_sign*t_T + t_noise_term)/num_classifiers
                    e_temp=(s_sign*e_S + e_sign*e_E + t_sign*e_T + e_noise_term)/num_classifiers

                    e_S.copy_(e_temp)
                    e_E.copy_(e_temp)
                    e_T.copy_(e_temp)
                    t_S.copy_(t_temp)
                    t_E.copy_(t_temp)
                    t_T.copy_(t_temp)

                exec_time += time.time()-tic

                

            info.update()

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]

        return ts_refined, qs_refined, info


    def refine_grasps_GrapsFlow_MTAdam(self, q, t, pc, pc_mean, classifiers="S"):

        e = quaternions2eulers(q)
        B = q.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        info = InfoHolder(var_names=["translations", "quaternions", "rots",
                                     "grasp_scores", "robot_scores", "table_scores",
                                     "translations_S_grad", "translations_E_grad", "translations_T_grad",
                                     "rots_S_grad", "rots_E_grad", "rots_T_grad",])

        print(f'Start to refine using GraspFlow MTAdam ... .')

        num_classifiers = len(classifiers)

        exec_time = 0
        for indcs in tqdm( split_indcs ):

            _pc, _e, _t, _q, _pc_mean = pc[indcs], e[indcs], t[indcs], q[indcs], pc_mean[indcs]

            if self.args.grasp_space == 'SO3':
                tic = time.time()
                _e = SO3(quaternion=_q).log_map()
                exec_time +=time.time() -tic

            rots_S_grad, rots_E_grad, rots_T_grad, translations_S_grad, translations_E_grad, translations_T_grad = 0,0,0,0,0,0

            e_t = torch.nn.Parameter(_e.clone())
            t_t = torch.nn.Parameter(_t.clone())

            optimizer = MTAdam([
                    {'params': [t_t], 'lr': self.args.eta_t},
                    {'params': [e_t], 'lr': self.args.eta_e},
                ], betas=(0.9, 0.999, 0.0), maximize=True, amsgrad=True)

            info.batch_init()

            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):

                # init scores
                scores, robot_scores, table_scores = None, None, None
                
                # zero gradients
                optimizer.zero_grad()
                
                # get gradient
                tic = time.time()

                losses = []
                if "S" in classifiers:
                    scores, logsigmoid_S = self.compute_logit(e_t, t_t, _pc, _pc_mean, classifier="S")
                    losses.append(logsigmoid_S)
                if "E" in classifiers:
                    robot_scores, logsigmoid_E = self.compute_logit(e_t, t_t, _pc, _pc_mean, classifier="E")
                    losses.append(logsigmoid_E)
                if "T" in classifiers:
                    table_scores, logsigmoid_T = self.compute_logit(e_t, t_t, _pc, _pc_mean, classifier="T")
                    losses.append(logsigmoid_T)

                exec_time += time.time()-tic

                # update info
                e_temp = e_t.detach().clone()
                t_temp = t_t.detach().clone()
                if self.args.grasp_space == 'SO3':
                    q_t = self.SO3_holder.exp_map(e_temp).to_quaternion()
                if self.args.grasp_space == 'Euler':
                    q_t = euler2quat(e_temp)

                
                with torch.no_grad():
                    if robot_scores is None:
                        robot_scores = torch.sigmoid(self.robot(t_temp, q_t))
                    if table_scores is None:
                        table_scores = torch.sigmoid(self.table(q_t, t_temp))
                    

                # update variables and add noise
                tic = time.time()
                _, states = optimizer.step(loss_array=losses, ranks=[1,1]) # updates gradients everywhere

                # print(f'param groups: {len(states)}')
                # print(f'params: {len(states[0])}')
                # print(f'shape: {states[0][0].shape}')

                if "S" in classifiers:
                    S_idx = classifiers.index('S')
                    translations_S_grad = states[0][0][S_idx, ...]
                    rots_S_grad = states[1][0][S_idx, ...]
                if "E" in classifiers:
                    E_idx = classifiers.index('E')
                    translations_E_grad = states[0][0][E_idx, ...]
                    rots_E_grad = states[1][0][E_idx, ...]
                if "T" in classifiers:
                    T_idx = classifiers.index('T')
                    translations_T_grad = states[0][0][T_idx, ...]
                    rots_T_grad = states[1][0][T_idx, ...]
                    
                info.batch_update(translations=t_t, quaternions=q_t, rots=e_t,
                    grasp_scores=scores, robot_scores=robot_scores, table_scores=table_scores,
                    translations_S_grad=translations_S_grad, rots_S_grad=rots_S_grad,
                    translations_E_grad=translations_E_grad, rots_E_grad=rots_E_grad,
                    translations_T_grad=translations_T_grad, rots_T_grad=rots_T_grad,)


                # with torch.no_grad():

                #     # add noise
                #     t_noise_term = num_classifiers*np.sqrt(2*self.args.eta_t) * self.args.noise_t * torch.randn_like(t_S)
                #     e_noise_term = num_classifiers*np.sqrt(2*self.args.eta_e) * self.args.noise_e * torch.randn_like(e_S)

                #     t_temp=(s_sign*t_S + e_sign*t_E + t_sign*t_T + t_noise_term)/num_classifiers
                #     e_temp=(s_sign*e_S + e_sign*e_E + t_sign*e_T + e_noise_term)/num_classifiers

                #     e_S.copy_(e_temp)
                #     e_E.copy_(e_temp)
                #     e_T.copy_(e_temp)
                #     t_S.copy_(t_temp)
                #     t_E.copy_(t_temp)
                #     t_T.copy_(t_temp)

                exec_time += time.time()-tic

                

            info.update()

        # conclude update
        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]

        return ts_refined, qs_refined, info

    def compute_phi(self, K_r, K_t, r, t , classifier="S"):

        # save gradients
        r_grad = r.grad.clone()
        t_grad = t.grad.clone()

        # initialize torch variables
        _r = r.clone()
        _r.requires_grad_(True)
        if _r.grad is not None:
            _r.grad.zero_()

        _t = t.clone()
        _t.requires_grad_(True)
        if _t.grad is not None:
            _t.grad.zero_()

        # compute covariances
        K_rr = K_r(_r)
        K_tt = K_t(_t)

        # compute gradients
        grad_K_r = torch.autograd.grad(K_rr.sum(), _r)[0]
        grad_K_t = torch.autograd.grad(K_tt.sum(), _t)[0]

        # add and average them
        _temp_rr = K_rr.matmul(r_grad)
        _temp_tt = K_tt.matmul(t_grad)
        phi_r = (_temp_rr + grad_K_r) / _r.size(0)
        phi_t = (_temp_tt + grad_K_t) / _t.size(0)

        # assign gradients
        r.grad = phi_r
        t.grad = phi_t

        return r.grad.data, t.grad.data, K_rr, K_tt, _temp_rr, _temp_tt, grad_K_r, grad_K_t




    def compute_logit(self, r, t, pc, pc_mean, classifier="S"):
        
        if classifier == "S":
            if self.args.grasp_space == 'SO3':
                r = self.SO3_holder.exp_map(r).to_quaternion()
                r = quat_normalize(r)
                logit, _ , _= self.evaluator.forward(r, t, pc)
            elif self.args.grasp_space == 'Euler':
                logit, _, _  = self.evaluator.forward_with_eulers(r, t, pc)
            
        elif classifier == "E":
            t = t + pc_mean # denormalize 
            if self.args.grasp_space == 'SO3':
                r = self.SO3_holder.exp_map(r).to_quaternion()
                r = quat_normalize(r)
                logit = self.robot(t, r)

            elif self.args.grasp_space == 'Euler':
                r=euler2quat(r, order="XYZ")
                logit = self.robot(t, r)
            
        elif classifier == "T":
            t = t + pc_mean # denormalize 
            if self.args.grasp_space == 'SO3':
                r = self.SO3_holder.exp_map(r).to_quaternion()
                r = quat_normalize(r)
                logit = self.table(r, t)
            elif self.args.grasp_space == 'Euler':
                logit = self.table.forward_with_eulers(r, t)
        else:
            raise ValueError()

        logsigmoid = torch.nn.functional.logsigmoid(logit)
        scores = torch.sigmoid(logit) # compute success

        return scores, logsigmoid

    def normalize_gradients(self, r, t):
        
        r.grad = torch.nan_to_num(r.grad, nan=0) # thesus error

        # Normalize gradients
        t_t_grad_norm = torch.norm(t.grad, p=2, dim=1, keepdim = True)#.clamp(min=-100, max=1000)
        r_t_grad_norm = torch.norm(r.grad, p=2, dim=1, keepdim = True)#.clamp(min=-100, max=1000)

        # Do not allow to move more than eta_t and eta_e
        t_t_grad_coeff = torch.min(1.0 / t_t_grad_norm, torch.tensor(1, dtype=torch.float).to(self.device))
        r_t_grad_coeff = torch.min(1.0 / r_t_grad_norm, torch.tensor(1, dtype=torch.float).to(self.device))

        t.grad = t.grad * t_t_grad_coeff
        r.grad = r.grad * r_t_grad_coeff

        return r.grad.data, t.grad.data, r_t_grad_norm, t_t_grad_norm


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
                logit = self.robot.mi_logit(_thetas)
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

    def _velocity(self, e, t, pc, isSO3=False, isGraspnet=False, isEuler=False):

        e_t = torch.nn.Parameter(e.clone())
        t_t = torch.nn.Parameter(t.clone())
            
        if isSO3:
            q_in_t = SO3().exp_map(e_t).to_quaternion()
            logit, _, _ = self.evaluator.forward(q_in_t, t_t, pc)
        elif isGraspnet or isEuler:
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
        return e_velocity, t_velocity, success

if __name__ == "__main__":
    print('Test')
