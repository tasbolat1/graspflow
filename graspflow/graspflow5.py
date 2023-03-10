import os
import torch
import numpy as np
from utils.auxilary import InfoHolder, quaternions2eulers, tensor_nans_like, eulers2quaternions
import time
from tqdm.auto import tqdm
from networks.quaternion import quat_normalize, euler2quat
from theseus.geometry import SO3
from svgd import SVGD 
from mtadam import MTAdam
import yaml
from graspflow_classifiers import S_classifier, E_classifier, T_classifier, C_classifier, N_classifier, D_classifier


class _my_sigmoid(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        out=torch.sigmoid(x)
        ctx.save_for_backward(x)
        return out
    
    @staticmethod
    def backward(ctx, grad_ouput):
        x, = ctx.saved_tensors
        # x = x.clamp(min=-650, max=650) # 45
        # stretch_coeff = 100000 # 1000
        # scale_coeff = 50
        # grad_input = torch.exp(-x**2/stretch_coeff)*scale_coeff

        grad_input = (-x)**2
        return grad_ouput

def my_sigmoid(x):
    return _my_sigmoid.apply(x)

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
        self.SO3_holder = SO3() # this makes faster

    def load_cfg(self, path_to_cfg):
        '''
        Load Classifier cfgs to the model
        '''
        cfg = yaml.safe_load(open(path_to_cfg, 'r'))

        if 'S' in cfg['classifiers']:
            self.S = S_classifier(device=self.device, batch_size=self.batch_size, 
                                  path_to_model=cfg['classifiers']['S']['path_to_model'],
                                  dist_coeff=cfg['classifiers']['S']['coeff'],
                                  dist_threshold=cfg['classifiers']['S']['threshold'],
                                  approximate=cfg['classifiers']['S']['approximate'])
            print("S classifier is loaded.")
        if 'C' in cfg['classifiers']:
            self.C = C_classifier(device=self.device, batch_size=self.batch_size, 
                                  threshold=cfg['classifiers']['C']['threshold'],
                                  coeff=cfg['classifiers']['C']['coeff'],
                                  isTable=cfg['classifiers']['C']['isTable'],
                                  approximate=cfg['classifiers']['C']['approximate'])
            print("C classifier is loaded.")
        if 'E' in cfg['classifiers']:
            self.E = E_classifier(device=self.device, batch_size=self.batch_size, 
                                  mi_coeff=cfg['classifiers']['E']['mi_coeff'],
                                  mi_threshold=cfg['classifiers']['E']['mi_threshold'],
                                  t_threshold=cfg['classifiers']['E']['t_threshold'],
                                  q_threshold=cfg['classifiers']['E']['q_threshold'],
                                  w_coeff=cfg['classifiers']['E']['w_coeff'])
            print("E classifier is loaded.")
        if 'T' in cfg['classifiers']:
            self.T = T_classifier(device=self.device, batch_size=self.batch_size)
            print("T classifier is loaded.")

        if 'N' in cfg['classifiers']:
            self.N = N_classifier(device=self.device, batch_size=self.batch_size)
            print("N classifier is loaded.")

        if 'D' in cfg['classifiers']:
            self.D = D_classifier(device=self.device, batch_size=self.batch_size,
                                alpha_threshold=cfg['classifiers']['D']['alpha_threshold'])


        # load optimizer
        if cfg['optimizer']['name'] == 'SGD':
            self.optimizer_function = torch.optim.SGD
            self.eta_t = 1.0
            self.eta_r = 1.0
            if 'GraspOpt' in cfg.keys():
                self.eta_t = cfg['optimizer']['eta_t']
                self.eta_r = cfg['optimizer']['eta_r']
            self.isMultiTerm = False
            self.grad_normalize = cfg['optimizer']['grad_normalize']
        elif cfg['optimizer']['name'] == 'Adam':
            self.optimizer_function = torch.optim.Adam
            self.eta_t = 1.0
            self.eta_r = 1.0
            if 'GraspOpt' in cfg.keys():
                self.eta_t = cfg['optimizer']['eta_t']
                self.eta_r = cfg['optimizer']['eta_r']
            self.isMultiTerm = False
            self.grad_normalize = cfg['optimizer']['grad_normalize']
        elif cfg['optimizer']['name'] == 'MTAdam':
            self.optimizer_function = MTAdam
            self.eta_t = cfg['optimizer']['eta_t']
            self.eta_r = cfg['optimizer']['eta_r']
            self.isMultiTerm = True
            self.grad_normalize = False # makes no sense to set it cfg['optimizer']['grad_normalize']
        else:
            raise NotImplementedError()


        # set GraspOpt parameters

        if 'GraspOpt' in cfg.keys():
            self.N_rules = len(cfg['GraspOpt'])
            self.graspopt_classifiers = []
            for i in range(1, self.N_rules+1):
                self.graspopt_classifiers.append(cfg['GraspOpt'][i]['classifiers'])

            self.graspopt_all_classifiers = ''.join(self.graspopt_classifiers)

        self.cfg = cfg

    def normalize_gradients(self, t, r):
        '''
        Normalizes gradients for given either tensor (requires_grad=True) or Parameter.
        Returns gradients
        '''
        
        r.grad = torch.nan_to_num(r.grad, nan=0) # thesus error

        # Normalize gradients
        t_t_grad_norm = torch.norm(t.grad, p=2, dim=1, keepdim = True)#.clamp(min=-100, max=1000)
        r_t_grad_norm = torch.norm(r.grad, p=2, dim=1, keepdim = True)#.clamp(min=-100, max=1000)

        # Do not allow to move more than eta_t and eta_e
        t_t_grad_coeff = torch.min(1.0 / t_t_grad_norm, torch.tensor(1, dtype=torch.float).to(self.device))
        r_t_grad_coeff = torch.min(1.0 / r_t_grad_norm, torch.tensor(1, dtype=torch.float).to(self.device))

        t.grad = t.grad * t_t_grad_coeff
        r.grad = r.grad * r_t_grad_coeff

        return t.grad.data, r.grad.data 


    def calculate_delta_and_set_default(self, t_init, r_init):
        '''
        Calculates "applied" grad modification on step function and sets to default value
        '''

        delta_t = self.optimizer.param_groups[0]['params'][0] - t_init
        delta_r = self.optimizer.param_groups[1]['params'][0] - r_init
        
        with torch.no_grad():
            self.optimizer.param_groups[0]['params'][0].copy_(t_init)
            self.optimizer.param_groups[1]['params'][0].copy_(r_init)

        return t_init, r_init, delta_t, delta_r

    def refine_grasps_GraspOpt(self, qs, ts, pcs, pc_means, queries=None, pc_envs=None):

        # convert to euler
        es = quaternions2eulers(qs)

        # prepare infoholder
        info_var_names = ["translations", "quaternions", "rots", "utility", "translations_grad", "rots_grad"]
        for classifier in self.graspopt_all_classifiers:
            info_var_names.append(f'{classifier}_scores')

        info = InfoHolder(var_names=info_var_names)

        B = qs.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        print(f'Start to refine using GraspOpt ... for {self.graspopt_all_classifiers}.')

        exec_time = 0
        for indcs in tqdm( split_indcs ):
            
            # select batch
            pc, e, t, q, pc_mean = pcs[indcs], es[indcs], ts[indcs], qs[indcs], pc_means[indcs]

            if pc_envs is not None:
                pc_env = pc_envs[indcs]
            if queries is not None:
                query = queries[indcs]

            if self.args.grasp_space == 'SO3':
                e = SO3(quaternion=q).log_map()

            # set optimizer
            t = torch.nn.Parameter(t.clone())
            e = torch.nn.Parameter(e.clone())
            self.optimizer = self.optimizer_function([
                    {'params': [t], 'lr': self.eta_t},
                    {'params': [e], 'lr': self.eta_r},
                ], maximize=True)

            info.batch_init()
            
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False, disable=False):
                tic = time.time()

                # Step 0: zero gradients
                self.optimizer.zero_grad()

                kword_data = {}
                # Step 1: calculate total utility function
                total_util = 1
                for i in range(1, self.N_rules+1):
                    sub_util = 1
                    for classifier in self.graspopt_classifiers[i-1]:
                        if classifier == "S":
                            logit, scores = self.S.forward(t, e, pc, grasp_space=self.args.grasp_space)
                            kword_data['S_scores'] = scores
                        if classifier == "E":
                            logit, scores = self.E.forward(t, e, pc_mean, grasp_space=self.args.grasp_space)
                            kword_data['E_scores'] = scores
                        if classifier == "C":
                            logit, scores = self.C.forward(t, e, pc_mean, pc_env=pc_env, grasp_space=self.args.grasp_space)
                            # print(logit)
                            kword_data['C_scores'] = scores
                        if classifier == "N":
                            logit, scores = self.N.forward(t, e, pc=pc, pc_mean=pc_mean, query=query, grasp_space=self.args.grasp_space)
                            # print(logit)
                            kword_data['N_scores'] = scores
                            
                        # clap flag is true
                        if self.cfg['GraspOptCfg']['clip']['use']:
                            logit = logit.clamp(min=self.cfg['GraspOptCfg']['clip']['min'], max=self.cfg['GraspOptCfg']['clip']['max'])
                            #print(logit)
                            # logit_scale = torch.abs(logit.clone().detach())
                            # logit = logit / logit_scale

                            # print(logit)


                        if int(self.cfg['GraspOptCfg']['lower_bound']['num_iterations']) >= k:
                            sub_util = sub_util + torch.torch.nn.functional.logsigmoid(logit)
      
                        else:
                            sub_util = sub_util*my_sigmoid(logit)
                    
                    
                    if int(self.cfg['GraspOptCfg']['lower_bound']['num_iterations']) >= k:
                        total_util = total_util + sub_util
                    else:
                        coeff = 2**(self.N_rules-i)
                        print(f'coeff: {coeff}')
                        total_util = total_util + coeff * sub_util
                    


                # Step 2: calculate gradients
                # print('BACKWARD JUST NOW TO BE RUN!')
                total_util.backward(torch.ones_like(total_util))
                # print('BACKWARD SUCCESSFULL!')

                # print('Gradients are here')
                # print(t.grad)

                t.grad = torch.nan_to_num(t.grad, nan=0)
                e.grad = torch.nan_to_num(e.grad, nan=0)

                # print(total_util)
                # print(e.grad)
                # print(t.grad)

                if self.grad_normalize:
                    t.grad, e.grad = self.normalize_gradients(t, e)

                exec_time += time.time()-tic

                # update infoholders
                with torch.no_grad():
                    if self.args.grasp_space == 'SO3':
                        _q = self.SO3_holder.exp_map(e).to_quaternion()
                        info.batch_update(translations=t, quaternions=_q, rots=e)
                    elif self.args.grasp_space == 'Euler':
                        _q = eulers2quaternions(e)
                        info.batch_update(translations=t, quaternions=_q, rots=e)

                    kword_data['translations_grad'] = t.grad
                    kword_data['rots_grad'] = e.grad
                    kword_data['utility'] = total_util
                
                

                # Step 3: update grasps
                tic = time.time()
                self.optimizer.step()
                exec_time += time.time()-tic

                info.batch_update(**kword_data)

            info.update()

        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]
                
        return ts_refined, qs_refined, info 

    def refine_grasps_GraspOptES(self, qs, ts, pcs, pc_means, queries=None, pc_envs=None):

        # convert to euler
        es = quaternions2eulers(qs)

        # prepare infoholder
        info_var_names = ["translations", "quaternions", "rots", "utility", "translations_grad", "rots_grad"]
        for classifier in self.graspopt_all_classifiers:
            info_var_names.append(f'{classifier}_scores')

        info = InfoHolder(var_names=info_var_names)

        B = qs.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        print(f'Start to refine using GraspOpt ... for {self.graspopt_all_classifiers}.')

        exec_time = 0
        for indcs in tqdm( split_indcs ):
            
            # select batch
            pc, e, t, q, pc_mean = pcs[indcs], es[indcs], ts[indcs], qs[indcs], pc_means[indcs]

            # TEMP INCREASE NUM OF INPUTS:
            pc = pc.repeat([2, 1, 1])
            pc_mean = pc_mean.repeat([2, 1])

            if pc_envs is not None:
                pc_env = pc_envs[indcs]
                pc_env = pc_env.repeat([2, 1, 1])
            if queries is not None:
                query = queries[indcs]
                # TODO: query

            if self.args.grasp_space == 'SO3':
                e = SO3(quaternion=q).log_map()

            # set optimizer
            t_temp = torch.nn.Parameter(t.clone())
            e_temp = torch.nn.Parameter(e.clone())
            self.optimizer = self.optimizer_function([
                    {'params': [t_temp], 'lr': self.eta_t},
                    {'params': [e_temp], 'lr': self.eta_r},
                ], maximize=True)

            info.batch_init()
            
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False, disable=False):

                tic = time.time()

                # step -1: sample additional grasps
                initial_sample_count = t.shape[0]
                _new_t = t + self.cfg['GraspOptESCfg']['t_std_dev'] * torch.randn_like(t)
                _new_e = e + self.cfg['GraspOptESCfg']['e_std_dev'] * torch.randn_like(e)

                del self.optimizer.param_groups[0]

                t = torch.nn.Parameter( torch.concat([t,_new_t], dim=0).clone() )
                e = torch.nn.Parameter( torch.concat([e,_new_e], dim=0).clone() )

                self.optimizer.add_param_group({'params': [t], 'lr': self.eta_t })
                self.optimizer.add_param_group({'params': [e], 'lr': self.eta_r })

                # # assign to params
                # with torch.no_grad():
                #     t.copy_(t)
                #     e.copy_(e)

                # Step 0: zero gradients
                self.optimizer.zero_grad()

                kword_data = {}
                # Step 1: calculate total utility function
                total_util = 1
                lb_utility = 0
                for i in range(1, self.N_rules+1):
                    sub_util = 1
                    lb_sub_utility = 0
                    for classifier in self.graspopt_classifiers[i-1]:
                       
                        all_args = (t,e,pc,pc_mean,pc_env,query)
                        logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                        kword_data[f'{classifier}_scores'] = scores
                        
                        lb_sub_utility = lb_sub_utility + torch.torch.nn.functional.logsigmoid(logit)
      
                        with torch.no_grad():
                            sub_util = sub_util*torch.sigmoid(logit)
                    
                    lb_utility = lb_utility + lb_sub_utility
                    with torch.no_grad():
                        coeff = 2**(self.N_rules-i)
                        total_util = total_util + coeff * sub_util




                # # assign to params
                # with torch.no_grad():
                #     t.copy_(t)
                #     e.copy_(e)

                # Step 2: calculate gradients
                lb_utility.backward(torch.ones_like(lb_utility))

                t.grad = torch.nan_to_num(t.grad, nan=0)
                e.grad = torch.nan_to_num(e.grad, nan=0)


                if self.grad_normalize:
                    t.grad, e.grad = self.normalize_gradients(t, e)

                exec_time += time.time()-tic     
                

                # Step 3: update grasps
                tic = time.time()
                self.optimizer.step()
                exec_time += time.time()-tic

                # sort and pop only top N ones

                kword_data['translations_grad'] = t.grad
                kword_data['rots_grad'] = e.grad


                topK = torch.argsort(total_util, dim=0, descending=True)[:initial_sample_count]
                topK = topK.squeeze(-1)
                t = t[topK,...]
                e = e[topK,...]
                total_util = total_util[topK, ...]
                lb_utility = lb_utility[topK, ...]

                for classifier in self.graspopt_all_classifiers:
                    kword_data[f'{classifier}_scores'] = kword_data[f'{classifier}_scores'][topK, ...]        
      

                # update infoholders
                with torch.no_grad():
                    if self.args.grasp_space == 'SO3':
                        _q = self.SO3_holder.exp_map(e).to_quaternion()
                        info.batch_update(translations=t, quaternions=_q, rots=e)
                    elif self.args.grasp_space == 'Euler':
                        _q = eulers2quaternions(e)
                        info.batch_update(translations=t, quaternions=_q, rots=e)

                    kword_data['utility'] = total_util
                

                info.batch_update(**kword_data)

            info.update()

        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]
                
        return ts_refined, qs_refined, info 


    def split_batch(self, current_size, batch_size):
        if current_size <= batch_size:
            split_size = 1
        else:
            split_size = int(current_size/(batch_size)) + 1
        split_indcs = np.array_split(np.arange(current_size), split_size)
        return split_indcs

    def refine_grasps_GraspOptES2(self, qs, ts, pcs, pc_means, queries=None, pc_envs=None):

        # convert to euler
        es = quaternions2eulers(qs)

        # prepare infoholder
        info_var_names = ["translations", "quaternions", "rots", "utility", "translations_grad", "rots_grad"]
        for classifier in self.graspopt_all_classifiers:
            info_var_names.append(f'{classifier}_scores')

        info = InfoHolder(var_names=info_var_names)

        split_indcs = self.split_batch(qs.shape[0], self.batch_size)

        print(f'Start to refine using GraspOpt ... for {self.graspopt_all_classifiers}.')

        exec_time = 0
        for indcs in tqdm( split_indcs ):
            
            # select batch
            pc, e, t, q, pc_mean = pcs[indcs], es[indcs], ts[indcs], qs[indcs], pc_means[indcs]


            if pc_envs is not None:
                pc_env = pc_envs[indcs]
            if queries is not None:
                query = queries[indcs]

            if self.args.grasp_space == 'SO3':
                e = SO3(quaternion=q).log_map()

            # set optimizer
            _t_temp = torch.nn.Parameter(t.clone())
            _e_temp = torch.nn.Parameter(e.clone())
            self.optimizer = self.optimizer_function([
                    {'params': [_t_temp], 'lr': self.eta_t},
                    {'params': [_e_temp], 'lr': self.eta_r},
                ], maximize=True)

            info.batch_init()            

            es_freq = self.cfg['GraspOptESCfg']['es_freq']
            num_samples_per_grasp = self.cfg['GraspOptESCfg']['num_samples_per_grasp']
            
            if self.cfg['GraspOptESCfg']['S_warmup_iterations'] !=-1:
                t, e = self.S_warmup(t,e,pc)

            for k in tqdm( range(0, self.args.max_iterations+1), leave=False, disable=False):

                tic = time.time()

                # STEP 1 (ES): no grad - calculate fitness function
                
                if ((k+1) % es_freq == 0):
                    with torch.no_grad():
                        initial_sample_count = t.shape[0]
                        
                        t_temp = t.repeat([num_samples_per_grasp, 1])
                        e_temp = e.repeat([num_samples_per_grasp, 1])
                        pc_temp = pc.repeat([num_samples_per_grasp, 1, 1])
                        pc_mean_temp = pc_mean.repeat([num_samples_per_grasp, 1])

                        if pc_env is not None:
                            pc_env_temp = pc_env.repeat([num_samples_per_grasp, 1, 1])
                        if query is not None:
                            query_temp = np.tile(query, [num_samples_per_grasp, 1]) # TODO: Check


                        t_temp = t_temp + self.cfg['GraspOptESCfg']['t_std_dev'] * torch.randn_like(t_temp)
                        e_temp = e_temp + self.cfg['GraspOptESCfg']['e_std_dev'] * torch.randn_like(e_temp)

                        total_util = 1
                        for i in range(1, self.N_rules+1):
                            sub_util = 1
                            for classifier in self.graspopt_classifiers[i-1]:
                                all_args = (t_temp,e_temp,pc_temp,pc_mean_temp,pc_env_temp,query_temp)
                                logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                                sub_util = sub_util*torch.sigmoid(logit)

                            coeff = 2**(self.N_rules-i)
                            total_util = total_util + coeff * sub_util

                        # fitness function
                        topK = torch.argsort(total_util, dim=0, descending=True)[:initial_sample_count]
                        topK = topK.squeeze(-1)
                        t = t_temp[topK,...].clone()
                        e = e_temp[topK,...].clone()
                        pc = pc_temp[topK,...].clone()
                        if pc_env is not None:
                            pc_env = pc_env_temp[topK, ...].clone()

                        if query is not None:
                            query = query_temp[topK.detach().cpu().numpy(), ...] # TODO: Test

                    
                # STEP 2: Calculate Lower Bound with gradients
                del self.optimizer.param_groups[0]
                t = torch.nn.Parameter( t.clone() )
                e = torch.nn.Parameter( e.clone() )

                self.optimizer.add_param_group({'params': [t], 'lr': self.eta_t })
                self.optimizer.add_param_group({'params': [e], 'lr': self.eta_r })

                self.optimizer.zero_grad()

                kword_data = {}
                # Step 1: calculate total utility function
                total_util = 1
                lb_utility = 0
                for i in range(1, self.N_rules+1):
                    sub_util = 1
                    lb_sub_utility = 0
                    for classifier in self.graspopt_classifiers[i-1]:
                       
                        all_args = (t,e,pc,pc_mean,pc_env,query)
                        logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                        kword_data[f'{classifier}_scores'] = scores
                        
                        lb_sub_utility = lb_sub_utility + torch.torch.nn.functional.logsigmoid(logit)
      
                        with torch.no_grad():
                            sub_util = sub_util*torch.sigmoid(logit)
                    
                    lb_utility = lb_utility + lb_sub_utility
                    with torch.no_grad():
                        coeff = 2**(self.N_rules-i)
                        total_util = total_util + coeff * sub_util


                # Step 2: calculate gradients
                lb_utility.backward(torch.ones_like(lb_utility))

                t.grad = torch.nan_to_num(t.grad, nan=0)
                e.grad = torch.nan_to_num(e.grad, nan=0)

                if self.grad_normalize:
                    t.grad, e.grad = self.normalize_gradients(t, e)

                exec_time += time.time()-tic     
                
                # Step 3: update grasps
                tic = time.time()
                self.optimizer.step()
                exec_time += time.time()-tic

                # sort and pop only top N ones
                kword_data['translations_grad'] = t.grad
                kword_data['rots_grad'] = e.grad

                for classifier in self.graspopt_all_classifiers:
                    kword_data[f'{classifier}_scores'] = kword_data[f'{classifier}_scores']        
      
                # update infoholders
                with torch.no_grad():
                    if self.args.grasp_space == 'SO3':
                        _q = self.SO3_holder.exp_map(e).to_quaternion()
                        info.batch_update(translations=t, quaternions=_q, rots=e)
                    elif self.args.grasp_space == 'Euler':
                        _q = eulers2quaternions(e)
                        info.batch_update(translations=t, quaternions=_q, rots=e)

                    kword_data['utility'] = total_util
                
                info.batch_update(**kword_data)

            info.update()

        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]
                
        return ts_refined, qs_refined, info

    def S_warmup(self, t, e, pc):
        t = torch.nn.Parameter( t.clone() )
        e = torch.nn.Parameter( e.clone() )

        local_optimizer = self.optimizer_function([
                    {'params': [t], 'lr': self.eta_t},
                    {'params': [e], 'lr': self.eta_r},
                ], maximize=True)

        S_warmup_iterations = self.cfg['GraspOptESCfg']['S_warmup_iterations']
        for k in tqdm( range(0, S_warmup_iterations), leave=False, disable=False):
            
            local_optimizer.zero_grad()

            all_args = (t,e,pc,None,None,None)
            logit, scores = self.classifier_func(all_args=all_args, classifier='S')

            local_loss = torch.torch.nn.functional.logsigmoid(logit)
            local_loss.backward(torch.ones_like(local_loss))

            t.grad = torch.nan_to_num(t.grad, nan=0)
            e.grad = torch.nan_to_num(e.grad, nan=0)

            if self.grad_normalize:
                    t.grad, e.grad = self.normalize_gradients(t, e)

            local_optimizer.step()

        return t.clone(), e.clone()

        
         

    def refine_grasps_GraspFlow_Sample(self, qs, ts, pcs, pc_means, pc_envs, queries=None, classifiers="S"):
        '''
        Refines grasps given as (q,t) tuples using GraspFlow method.
        '''
        
        es = quaternions2eulers(qs)

        # prepare infoholder
        info_var_names = ["translations", "quaternions", "rots"]
        for classifier in classifiers:
            info_var_names.append(f'translations_{classifier}_grad')
            info_var_names.append(f'rots_{classifier}_grad')
            info_var_names.append(f'{classifier}_scores')

        info = InfoHolder(var_names=info_var_names)

        B = qs.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        
        print(f'Start to refine using GraspFlow ... for {classifiers}.')

        if "T" in classifiers and os.path.exists("/home/crslab/GRASP/graspflow_models/experiments/tactile_observations.csv"):
            self.T.update_all()

        exec_time = 0
        for indcs in tqdm( split_indcs ):
            
            # select batch
            pc, e, t, q, pc_mean = pcs[indcs], es[indcs], ts[indcs], qs[indcs], pc_means[indcs]

            if pc_envs is not None:
                pc_env = pc_envs[indcs]
            if queries is not None:
                query = queries[indcs]

            if self.args.grasp_space == 'SO3':
                e = SO3(quaternion=q).log_map()

            # set optimizer
            t = torch.nn.Parameter(t.clone())
            e = torch.nn.Parameter(e.clone())
            self.optimizer = self.optimizer_function([
                    {'params': [t], 'lr': self.eta_t},
                    {'params': [e], 'lr': self.eta_r},
                ], maximize=True)

            info.batch_init()

            
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):
                    
                # STEP 1: Forward pass and gradient collection
                tic = time.time()
                kword_data = {}
                if "S" in classifiers:
                    t_init = t.clone()
                    e_init = e.clone()
                    self.optimizer.zero_grad()
                    logit, scores = self.S.forward(t, e, pc, grasp_space=self.args.grasp_space)
                    logsigmoid = torch.nn.functional.logsigmoid(logit)

                    logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))
                    if self.grad_normalize:
                        self.optimizer.param_groups[0]['params'][0].grad, self.optimizer.param_groups[1]['params'][0].grad = self.normalize_gradients(self.optimizer.param_groups[0]['params'][0], self.optimizer.param_groups[1]['params'][0])

                    self.optimizer.step()
                    kword_data["S_scores"] = scores
                    t, e, delta_t, delta_r = self.calculate_delta_and_set_default(t_init, e_init)
                    kword_data[f'translations_S_grad'] = delta_t
                    kword_data[f'rots_S_grad'] = delta_r

                if "E" in classifiers:
                    t_init = t.clone()
                    e_init = e.clone()
                    self.optimizer.zero_grad()
                    logit, scores = self.E.forward(t, e, pc_mean, grasp_space=self.args.grasp_space)
                    logsigmoid = torch.nn.functional.logsigmoid(logit)
                    logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))
                    if self.grad_normalize:
                        self.optimizer.param_groups[0]['params'][0].grad, self.optimizer.param_groups[1]['params'][0].grad = self.normalize_gradients(self.optimizer.param_groups[0]['params'][0], self.optimizer.param_groups[1]['params'][0])

                    self.optimizer.step()
                    kword_data["E_scores"] = scores
                    t, e, delta_t, delta_r = self.calculate_delta_and_set_default(t_init, e_init)
                    kword_data[f'translations_E_grad'] = delta_t
                    kword_data[f'rots_E_grad'] = delta_r

                if "C" in classifiers:
                    t_init = t.clone()
                    e_init = e.clone()
                    self.optimizer.zero_grad()
                    logit, scores = self.C.forward(t, e, pc_mean, pc_env=pc_env, grasp_space=self.args.grasp_space)
                    logsigmoid = torch.nn.functional.logsigmoid(logit)
                    logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))
                    if self.grad_normalize:
                        self.optimizer.param_groups[0]['params'][0].grad, self.optimizer.param_groups[1]['params'][0].grad = self.normalize_gradients(self.optimizer.param_groups[0]['params'][0], self.optimizer.param_groups[1]['params'][0])

                    self.optimizer.step()
                    kword_data["C_scores"] = scores
                    t, e, delta_t, delta_r = self.calculate_delta_and_set_default(t_init, e_init)
                    kword_data[f'translations_C_grad'] = delta_t
                    kword_data[f'rots_C_grad'] = delta_r

                if "N" in classifiers:
                    t_init = t.clone()
                    e_init = e.clone()
                    self.optimizer.zero_grad()
                    logit, scores = self.N.forward(t, e, pc_mean, pc_env=pc_env, grasp_space=self.args.grasp_space)
                    logsigmoid = torch.nn.functional.logsigmoid(logit)
                    logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))
                    if self.grad_normalize:
                        self.optimizer.param_groups[0]['params'][0].grad, self.optimizer.param_groups[1]['params'][0].grad = self.normalize_gradients(self.optimizer.param_groups[0]['params'][0], self.optimizer.param_groups[1]['params'][0])

                    self.optimizer.step()
                    kword_data["N_scores"] = scores
                    t, e, delta_t, delta_r = self.calculate_delta_and_set_default(t_init, e_init)
                    kword_data[f'translations_N_grad'] = delta_t
                    kword_data[f'rots_N_grad'] = delta_r

                if "T" in classifiers:
                    t_init = t.clone()
                    e_init = e.clone()
                    self.optimizer.zero_grad()
                    logit, scores = self.T.forward(t, e, pc, grasp_space=self.args.grasp_space)
                    logsigmoid = torch.nn.functional.logsigmoid(logit)
                    logsigmoid.backward(torch.ones_like(logsigmoid).to(self.device))
                    if self.grad_normalize:
                        self.optimizer.param_groups[0]['params'][0].grad, self.optimizer.param_groups[1]['params'][0].grad = self.normalize_gradients(self.optimizer.param_groups[0]['params'][0], self.optimizer.param_groups[1]['params'][0])
                    self.optimizer.step()

                    kword_data["T_scores"] = scores
                    t, e, delta_t, delta_r = self.calculate_delta_and_set_default(t_init, e_init)
                    kword_data[f'translations_T_grad'] = delta_t
                    kword_data[f'rots_T_grad'] = delta_r

               
                exec_time += time.time()-tic

                # update infoholders
                with torch.no_grad():
                    if self.args.grasp_space == 'SO3':
                        _q = self.SO3_holder.exp_map(e).to_quaternion()
                        info.batch_update(translations=t, quaternions=_q, rots=e)
                    elif self.args.grasp_space == 'Euler':
                        _q = eulers2quaternions(e)
                        info.batch_update(translations=t, quaternions=_q, rots=e)


                # STEP 2: Update grasps with obtained gradients from each classifier
                tic = time.time()
                t_new = t.clone()
                e_new = e.clone()

           

                for idx, classifier in enumerate(classifiers):
                    t_new.add_(self.cfg['classifiers'][classifier]['eta_t']*kword_data[f'translations_{classifier}_grad'])
                    e_new.add_(self.cfg['classifiers'][classifier]['eta_r']*kword_data[f'rots_{classifier}_grad'])


                # STEP 3: Add noise
                t_new.add_(self.cfg['noise']['eta_t'] * torch.randn_like(t))
                e_new.add_(self.cfg['noise']['eta_r'] * torch.randn_like(e))

                # assign to params
                with torch.no_grad():
                    t.copy_(t_new)
                    e.copy_(e_new)

                exec_time += time.time()-tic

                info.batch_update(**kword_data)

            info.update()

        info.conclude(exec_time=exec_time)
        # print(exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]

        return ts_refined, qs_refined, info 


    def refine_grasps_GraspFlow_MTAdam(self, qs, ts, pcs, pc_means, pc_envs=None, queries=None, classifiers="S"):
        '''
        Refines grasps given as (q,t) tuples using GraspFlow method.
        '''
        
        es = quaternions2eulers(qs)

        # prepare infoholder
        info_var_names = ["translations", "quaternions", "rots"]
        for classifier in classifiers:
            info_var_names.append(f'translations_{classifier}_grad')
            info_var_names.append(f'rots_{classifier}_grad')
            info_var_names.append(f'{classifier}_scores')

        info = InfoHolder(var_names=info_var_names)

        B = qs.shape[0]
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        if "T" in classifiers and os.path.exists("/home/crslab/GRASP/graspflow_models/experiments/tactile_observations.csv"):
            self.T.update_all()
        
        print(f'Start to refine using GraspFlow ... .')

        for indcs in tqdm( split_indcs ):
            
            # select batch
            pc, e, t, q, pc_mean = pcs[indcs], es[indcs], ts[indcs], qs[indcs], pc_means[indcs]

            if pc_envs is not None:
                pc_env = pc_envs[indcs]
            if queries is not None:
                query = queries[indcs]

            if self.args.grasp_space == 'SO3':
                e = SO3(quaternion=q).log_map()

            # set optimizer
            t = torch.nn.Parameter(t.clone())
            e = torch.nn.Parameter(e.clone())
            self.optimizer = self.optimizer_function([
                    {'params': [t], 'lr': self.eta_t},
                    {'params': [e], 'lr': self.eta_r},
                ], maximize=True)

            info.batch_init()

            exec_time = 0
            for k in tqdm( range(0, self.args.max_iterations+1), leave=False):
                    
                # print(f'Iteration {k}')
                tic = time.time()

                with torch.autograd.set_detect_anomaly(True):

                    losses = []
                    kword_data = {}
                    if "S" in classifiers:
                        S_logit, S_scores = self.S.forward(t, e, pc, grasp_space=self.args.grasp_space)
                        S_logsigmoid = torch.nn.functional.logsigmoid(S_logit)
                        losses.append(S_logsigmoid)
                        kword_data["S_scores"] = S_scores
 

                    if "E" in classifiers:
                        E_logit, E_scores = self.E.forward(t, e, pc_mean, grasp_space=self.args.grasp_space)
                        E_logsigmoid = torch.nn.functional.logsigmoid(E_logit)
                        losses.append(E_logsigmoid)
                        kword_data["E_scores"] = E_scores

                    if "C" in classifiers:
                        C_logit, C_scores = self.C.forward(t, e, pc_mean, pc_env=pc_env, grasp_space=self.args.grasp_space)
                        C_logsigmoid = torch.nn.functional.logsigmoid(C_logit)
                        losses.append(C_logsigmoid)
                        kword_data["C_scores"] = C_scores

                    if "T" in classifiers:
                        #self.T.update(None, 0.9, None)
                        T_logit, T_scores = self.T.forward(t, e, pc, grasp_space=self.args.grasp_space)
                        T_logsigmoid = torch.nn.functional.logsigmoid(T_logit)
                        losses.append(T_logsigmoid)
                        kword_data["T_scores"] = T_scores

                    if "N" in classifiers:
                        N_logit, N_scores = self.N.forward(t, e, pc, pc_mean, query, grasp_space=self.args.grasp_space)
                        N_logsigmoid = torch.nn.functional.logsigmoid(N_logit)
                        losses.append(N_logsigmoid)
                        kword_data["N_scores"] = N_scores

                    # update weights
                    t_init = t.clone()
                    e_init = e.clone()

                    ranks = [1]*len(classifiers)
                    states = self.optimizer.step(losses, ranks=ranks)
                    t_grads = states[1][0][0]
                    r_grads = states[1][1][0]

                    exec_time += time.time()-tic

                    # update infoholders
                    with torch.no_grad():
                        if self.args.grasp_space == 'SO3':
                            _q = self.SO3_holder.exp_map(e_init).to_quaternion()
                            info.batch_update(translations=t_init, quaternions=_q, rots=e_init)
                        elif self.args.grasp_space == 'Euler':
                            _q = eulers2quaternions(e_init)
                            info.batch_update(translations=t_init, quaternions=_q, rots=e_init)
                    
                    for idx, classifier in enumerate(classifiers):
                        kword_data[f'translations_{classifier}_grad'] = t_grads[idx]
                        kword_data[f'rots_{classifier}_grad'] = r_grads[idx]

                    info.batch_update(**kword_data)

            info.update()

        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]

        return ts_refined, qs_refined, info

    


    def classifier_func(self, all_args, classifier='S'):
        t, e, pc, pc_mean, pc_env, query = all_args
        if classifier == "S":
            logit, scores = self.S.forward(t, e, pc, grasp_space=self.args.grasp_space)
        if classifier == "E":
            logit, scores = self.E.forward(t, e, pc_mean, grasp_space=self.args.grasp_space)
        if classifier == "C":
            logit, scores = self.C.forward(t, e, pc_mean, pc_env=pc_env, grasp_space=self.args.grasp_space)
        if classifier == "N":
            logit, scores = self.N.forward(t, e, pc=pc, pc_mean=pc_mean, query=query, grasp_space=self.args.grasp_space)
        if classifier == "D":
            logit, scores = self.D.forward(t, e, grasp_space=self.args.grasp_space)
        return logit, scores


    def refine_grasps_GraspFlow(self, qs, ts, pcs, pc_means, pc_envs, queries, classifiers="S"):
    
        if self.isMultiTerm:
            return self.refine_grasps_GraspFlow_MTAdam(qs, ts, pcs, pc_means, pc_envs, queries=queries, classifiers=classifiers)
        else:
            return self.refine_grasps_GraspFlow_Sample(qs, ts, pcs, pc_means, pc_envs, queries=queries, classifiers=classifiers)



if __name__ == "__main__":
    print('Test')
