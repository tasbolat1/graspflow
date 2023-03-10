import os
import torch
import numpy as np
from utils.auxilary import InfoHolder, quaternions2eulers, tensor_nans_like, eulers2quaternions
import time
from tqdm.auto import tqdm
from theseus.geometry import SO3
from mtadam import MTAdam
import yaml
from graspflow_classifiers import S_classifier, E_classifier, T_classifier, C_classifier, N_classifier, D_classifier
from utils.auxilary import construct_grasp_matrix
from networks.quaternion import rot_p_by_quaterion

class GraspFlow():
    def __init__(self, args, include_robot=False):
        '''
        GraspFlow - a framework that refines grasps using different discriminators.
        '''
        self.args = args
        self.device = args.device
        if args.device == -1:
            self.device = 'cpu'
        self.batch_size = self.args.batch_size 
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
                                  approximate=cfg['classifiers']['C']['approximate'],
                                  base_extension=cfg['classifiers']['C']['base_extension'],
                                  fill_between_fingers=cfg['classifiers']['C']['fill_between_fingers'])
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
            print("D classifier is loaded.")


        # load optimizer
        if cfg['optimizer']['name'] == 'SGD':
            self.optimizer_function = torch.optim.SGD
            self.isMultiTerm = False
            self.grad_normalize = cfg['optimizer']['grad_normalize']
        elif cfg['optimizer']['name'] == 'Adam':
            self.optimizer_function = torch.optim.Adam
            self.isMultiTerm = False
            self.grad_normalize = cfg['optimizer']['grad_normalize']
        elif cfg['optimizer']['name'] == 'MTAdam':
            self.optimizer_function = MTAdam
            self.isMultiTerm = True
            self.grad_normalize = False # makes no sense to set it cfg['optimizer']['grad_normalize']
        else:
            raise NotImplementedError()

        self.eta_t = cfg['optimizer']['eta_t']
        self.eta_r = cfg['optimizer']['eta_r']
        # set GraspOpt parameters

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
        
        # Normalize gradients
        t_t_grad_norm = torch.norm(t.grad, p=2, dim=1, keepdim = True)#.clamp(min=-100, max=1000)
        r_t_grad_norm = torch.norm(r.grad, p=2, dim=1, keepdim = True)#.clamp(min=-100, max=1000)

        # Do not allow to move more than eta_t and eta_e
        t_t_grad_coeff = torch.min(1.0 / t_t_grad_norm, torch.tensor(1, dtype=torch.float).to(self.device))
        r_t_grad_coeff = torch.min(1.0 / r_t_grad_norm, torch.tensor(1, dtype=torch.float).to(self.device))

        t.grad = t.grad * t_t_grad_coeff
        r.grad = r.grad * r_t_grad_coeff

        return t.grad.data, r.grad.data

    def refine_grasps_GraspOptES(self, qs, ts, pcs, pc_means, queries=None, pc_envs=None):

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
                query_i = queries[indcs]

            if self.args.grasp_space == 'SO3':
                e = SO3(quaternion=q).log_map()

            
            # Intents for N classifier:
            if 'N' in self.graspopt_all_classifiers:
                query = self.N.get_intents(query_i)


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

            # run first iteration
            tic = time.time()
            with torch.no_grad():
                kword_data = {}
                all_args = (t,e,pc,pc_mean,pc_env,query)
                
                

                _q = self.SO3_holder.exp_map(e).to_quaternion()
                info.batch_update(translations=t, quaternions=_q, rots=e)


                # for classifier in self.graspopt_all_classifiers:
                #     logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                #     kword_data[f'{classifier}_scores'] = scores

                total_util = 1
                for i in range(1, self.N_rules+1):
                    sub_util = 1
                    for classifier in self.graspopt_classifiers[i-1]:
                        logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                        kword_data[f'{classifier}_scores'] = scores
                        sub_util = sub_util*torch.sigmoid(logit)

                    coeff = 2**(self.N_rules-i)
                    total_util = total_util + coeff * sub_util

                # add nan gradiets for first run
                kword_data['translations_grad'] = tensor_nans_like(t)
                kword_data['rots_grad'] = tensor_nans_like(e)
                kword_data['utility'] = total_util

                info.batch_update(**kword_data)
            exec_time += time.time()-tic


            # Note: this is not added to the info
            t, e = self.init_warmup(t,e,pc)

            # Note: this is not added to the info
            if self.cfg['GraspOptESCfg']['S_warmup_iterations'] !=-1:
                t, e = self.S_warmup(t,e,pc)

            for k in tqdm( range(1, self.args.max_iterations+1), leave=False, disable=False):
                tic = time.time()

                # STEP 1 (ES): no grad - calculate fitness function
                if (k % es_freq == 0):
                    with torch.no_grad():
                        initial_sample_count = t.shape[0]
                        t_temp = t.repeat([num_samples_per_grasp, 1])
                        e_temp = e.repeat([num_samples_per_grasp, 1])
                        pc_temp = pc.repeat([num_samples_per_grasp, 1, 1])
                        pc_mean_temp = pc_mean.repeat([num_samples_per_grasp, 1])


                        if pc_env is not None:
                            pc_env_temp = pc_env.repeat([num_samples_per_grasp, 1, 1])

                        query_temp = None
                        if query is not None: # TODO
                            query_temp0 = np.tile(query[0], [num_samples_per_grasp])
                            query_temp1 = np.tile(query[1], [num_samples_per_grasp]) # TODO: Check
                            query_temp = [query_temp0, query_temp1]

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

                        if (query is not None) and  ('N' in self.graspopt_all_classifiers) :
                            query = [query_temp[0][topK.detach().cpu().numpy()], query_temp[1][topK.detach().cpu().numpy()] ] # TODO: Test


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

    def refine_grasps_GraspOptES2(self, qs, ts, pcs, pc_means, queries=None, pc_envs=None):

        # convert to euler
        es = quaternions2eulers(qs)

        # prepare infoholder
        info_var_names = ["translations", "quaternions", "utility"]
        for classifier in self.graspopt_all_classifiers:
            info_var_names.append(f'{classifier}_scores')

        info = InfoHolder(var_names=info_var_names)
        split_indcs = self.split_batch(qs.shape[0], self.batch_size)

        if pc_envs is None:
            pc_envs = torch.zeros_like(pc)
        if queries is None:
            queries = np.zeros([ts.shape[0], 1])

        print(f'Start to refine using GraspOpt ... for {self.graspopt_all_classifiers} for trial {self.args.grasp_folder}.')
        exec_time = 0

        for indcs in tqdm( split_indcs ):
            
            # select batch
            pc, e, t, q, pc_mean, pc_env, query_i = pcs[indcs], es[indcs], ts[indcs], qs[indcs], pc_means[indcs], pc_envs[indcs], queries[indcs]

            if self.args.grasp_space == 'SO3':
                e = SO3(quaternion=q).log_map()

            # Intents for N classifier:
            if 'N' in self.graspopt_all_classifiers:
                query = self.N.get_intents(query_i)
            else:
                query = query_i

            # set optimizer
            _t_temp = torch.nn.Parameter(t.clone())
            _e_temp = torch.nn.Parameter(e.clone())
            self.optimizer = self.optimizer_function([
                    {'params': [_t_temp], 'lr': self.eta_t},
                    {'params': [_e_temp], 'lr': self.eta_r},
                ], maximize=True)

            info.batch_init()            

            grad_iterations = self.cfg['GraspOptESCfg']['grad_iterations']
            num_samples_per_grasp = self.cfg['GraspOptESCfg']['num_samples_per_grasp']

            # run first iteration
            tic = time.time()
            with torch.no_grad():
                kword_data = {}
                all_args = (t,e,pc,pc_mean,pc_env,query)
                
                _q = self.SO3_holder.exp_map(e).to_quaternion()
                info.batch_update(translations=t, quaternions=_q)

                total_util = 1
                for i in range(1, self.N_rules+1):
                    sub_util = 1
                    for classifier in self.graspopt_classifiers[i-1]:
                        logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                        kword_data[f'{classifier}_scores'] = scores
                        sub_util = sub_util*torch.sigmoid(logit)

                    coeff = 2**(self.N_rules-i)
                    total_util = total_util + coeff * sub_util

                kword_data['utility'] = total_util

                info.batch_update(**kword_data)

            exec_time += time.time()-tic

            # Note: this is not added to the info
            t, e = self.init_warmup(t,e,pc)

            # Note: this is not added to the info
            if self.cfg['GraspOptESCfg']['S_warmup_iterations'] !=-1:
                t, e = self.S_warmup(t,e,pc)

            for k in tqdm( range(1, self.args.max_iterations+1), leave=False, disable=False):
                

                ############################### STEP 1 (Grad): grad optimization #######################################
                initial_sample_count = t.shape[0]
                t_temp = t.repeat([num_samples_per_grasp, 1])
                e_temp = e.repeat([num_samples_per_grasp, 1])
                pc_temp = pc.repeat([num_samples_per_grasp, 1, 1])
                pc_mean_temp = pc_mean.repeat([num_samples_per_grasp, 1])
                pc_env_temp = pc_env.repeat([num_samples_per_grasp, 1, 1])
                query_temp = np.tile(query, [num_samples_per_grasp])

                t_temp = t_temp + self.cfg['GraspOptESCfg']['t_std_dev'] * torch.randn_like(t_temp)
                e_temp = e_temp + self.cfg['GraspOptESCfg']['e_std_dev'] * torch.randn_like(e_temp)

                # locally optimize

                del self.optimizer.param_groups[0]
                t_temp = torch.nn.Parameter( t_temp.clone() )
                e_temp = torch.nn.Parameter( e_temp.clone() )

                self.optimizer.add_param_group({'params': [t_temp], 'lr': self.eta_t })
                self.optimizer.add_param_group({'params': [e_temp], 'lr': self.eta_r })

                tic = time.time()
                for k in range(grad_iterations):
                    
                    self.optimizer.zero_grad()

                    lb_utility = 0
                    for classifier in self.graspopt_all_classifiers:
                        all_args = (t_temp,e_temp,pc_temp,pc_mean_temp,pc_env_temp,query_temp)
                        logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                        lb_utility = lb_utility + torch.torch.nn.functional.logsigmoid(logit)

                    # calculate gradients
                    lb_utility.backward(torch.ones_like(lb_utility))

                    t_temp.grad = torch.nan_to_num(t_temp.grad, nan=0)
                    e_temp.grad = torch.nan_to_num(e_temp.grad, nan=0)

                    if self.grad_normalize:
                        t_temp.grad, e_temp.grad = self.normalize_gradients(t_temp, e_temp)
                        
                    self.optimizer.step()

                exec_time += time.time()-tic
                
                ############################### STEP 2: selection #######################################

                kword_data = {}
                with torch.no_grad():
                    t = torch.concat([t, t_temp], dim=0)
                    e = torch.concat([e, e_temp], dim=0)
                    pc = torch.concat([pc, pc_temp], dim=0)
                    pc_mean = torch.concat([pc_mean, pc_mean_temp], dim=0)
                    pc_env = torch.concat([pc_env, pc_env_temp], dim=0)
                    query = np.concatenate([query, query_temp], axis=0)

                    
                    total_util = 1
                    for i in range(1, self.N_rules+1):
                        sub_util = 1
                        for classifier in self.graspopt_classifiers[i-1]:
                            all_args = (t,e,pc,pc_mean,pc_env,query)
                            logit, scores = self.classifier_func(all_args=all_args, classifier=classifier)
                            sub_util = sub_util*torch.sigmoid(logit)
                            kword_data[f'{classifier}_scores'] = scores

                        coeff = 2**(self.N_rules-i)
                        total_util = total_util + coeff * sub_util

                    exec_time += time.time()-tic

                    topK = torch.argsort(total_util, dim=0, descending=True)[:initial_sample_count]
                    topK = topK.squeeze(-1)

                    

                    t = t[topK,...].clone()
                    e = e[topK,...].clone()
                    pc = pc[topK,...].clone()
                    pc_env = pc_env[topK, ...].clone()
                    pc_mean = pc_mean[topK, ...].clone()
                    total_util = total_util[topK, ...].clone()
                    query = query[topK.detach().cpu().numpy(), ...].copy()
                    
                    for classifier in self.graspopt_all_classifiers:
                        kword_data[f'{classifier}_scores'] = kword_data[f'{classifier}_scores'][topK, ...].clone()

                    if self.args.grasp_space == 'SO3':
                        _q = self.SO3_holder.exp_map(e).to_quaternion()
                        info.batch_update(translations=t, quaternions=_q)
                    elif self.args.grasp_space == 'Euler':
                        _q = eulers2quaternions(e)
                        info.batch_update(translations=t, quaternions=_q)
                    kword_data['utility'] = total_util

                    info.batch_update(**kword_data)

            info.update()

        info.conclude(exec_time=exec_time)

        # extract values
        ts_refined=info.data['translations'][-1,...]
        qs_refined=info.data['quaternions'][-1,...]
                
        return ts_refined, qs_refined, info

    def init_warmup(self, t, e, pc, grasp_space='SO3'):

        offset = 0.025
        t_offset = torch.FloatTensor([0,0,offset]).unsqueeze(0).unsqueeze(0).to(t.device)
        t_offset = t_offset.repeat([t.shape[0], 1, 1])

        if grasp_space == 'SO3':
            q = self.SO3_holder.exp_map(e).to_quaternion()
        else:
            q = e.clone()
            
        t_delta = rot_p_by_quaterion(t_offset, q.unsqueeze(1)).squeeze(1)
        t = t + t_delta

        return t, e

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

    def classifier_func(self, all_args, classifier='S'):
        t, e, pc, pc_mean, pc_env, query = all_args
        if classifier == "S":
            logit, scores = self.S.forward(t, e, pc, grasp_space=self.args.grasp_space)
        if classifier == "E":
            logit, scores = self.E.forward(t, e, pc_mean, grasp_space=self.args.grasp_space)
        if classifier == "C":
            logit, scores = self.C.forward(t, e, pc_mean, pc_env=pc_env, grasp_space=self.args.grasp_space)
        if classifier == "N":
            logit, scores = self.N.forward(t, e, pc=pc, query=query, grasp_space=self.args.grasp_space)
        if classifier == "D":
            logit, scores = self.D.forward(t, e, grasp_space=self.args.grasp_space)
        return logit, scores

    def split_batch(self, current_size, batch_size):
        if current_size <= batch_size:
            split_size = 1
        else:
            split_size = int(current_size/(batch_size)) + 1
        split_indcs = np.array_split(np.arange(current_size), split_size)
        return split_indcs


if __name__ == "__main__":
    print('Test')
