from ftplib import all_errors
import torch
import numpy as np
from networks.models import GraspEvaluator
from utils.points import propose_grasps
from utils.auxilary import quaternions2eulers, InfoHolder
from tqdm.auto import tqdm
import time
from scipy.spatial.transform import Rotation as R
from networks import quaternion as quat_ops
from networks import utils
from utils.auxilary import PandaGripper, construct_grasp_matrix
import trimesh
from networks.models import GPFeatureExtractor, NNCorrector


class GraspNet():
    def __init__(self, args):
        self.args = args
        self.device = args.device
        if args.device == -1:
            self.device = 'cpu'
        self.batch_size=self.args.batch_size 

        self.info = InfoHolder()

    def sample_grasp(self, pc):
        # pc: [B, 1024, 3]
        # return:
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized

        # TODO: incorporate sampling graspnet6dof sampling
        quaternions, translations = propose_grasps(pc, radius=0.01, num_grasps=self.args.n, sampler=self.args.sampler)
        quaternions = torch.FloatTensor(quaternions).to(self.device)
        translations = torch.FloatTensor(translations).to(self.device)
        return quaternions, translations


    def refine_grasps_metropolis(self, quaternions, translations, pc, mode='refinement'):
        # Input:
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized
        # return
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized

        self.info = InfoHolder()


        if self.args.rot_reps == 'euler':
            rots = quaternions2eulers(quaternions)
        elif self.args.rot_reps == 'quaternion':
            rots = quaternions.copy()
        else:
            rots = utils.quat_to_A_vec(quaternions)

        n = translations.shape[0]
        split_size = int(n/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(n), split_size)
        #print(split_indcs)
        # refine by batch

        filter_masks = torch.ones(translations.shape[0]).to(translations.device)

        print(f'Start to refine using metropolis ... .')
        start_time = time.time()
        last_success =  None
        with torch.no_grad():
            for indcs in tqdm( split_indcs ):
                _pc, _rots, _translations = pc[indcs].float(), rots[indcs].float(), translations[indcs].float()
                _filter_masks = filter_masks[indcs]
                last_success = None
                self.info.batch_init()
                for t in tqdm( range(1, self.args.max_iterations+1), leave=False):
                    if last_success is None:
                        last_success, _, _  = self.evaluator.forward_with_eulers(_rots, _translations, _pc)
                        last_success = torch.sigmoid(last_success)

                    delta_t = 2 * (torch.rand(_translations.shape, dtype=torch.float).to(self.device) - 0.5) * 0.02
                    delta_euler_angles = (torch.rand(_rots.shape, dtype=torch.float).to(self.device) - 0.5) * 2
                    perturbed_translation = _translations.data + delta_t.float()
                    perturbed_euler_angles = _rots.data + delta_euler_angles.float()
                    
                    d_score, _, _  = self.evaluator.forward_with_eulers(perturbed_euler_angles, perturbed_translation, _pc)
                    perturbed_success = torch.sigmoid(d_score)
                    ratio = perturbed_success / torch.max(last_success,torch.tensor(0.0001).to(self.device))
                    mask = torch.rand(ratio.shape).to(self.device) <= ratio

                    ind = torch.where(mask)[0]
                    last_success[ind] = perturbed_success[ind]
                
                    # update
                    _rots[ind] = perturbed_euler_angles.data[ind].float()
                    _translations[ind] = perturbed_translation.data[ind].float() 

                    # update info
                    self.info.batch_update(_rots,None, _translations, None, last_success, _filter_masks, self.args.rot_reps)


                # add to the info last updates
                last_batch_success = self.evaluate_grasps(_rots, _translations, _pc)
                self.info.batch_update(_rots,None, _translations, None, last_batch_success, None, self.args.rot_reps)
                self.info.update()

        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)

        # # final filter to get rid off colluding samples
        # collision_filters = self.filter_collision_grasps(self.info.quaternions[-1, ...], self.info.translations[-1,...], pc.cpu().numpy())
        # self.info.filter_mask[-1] = self.info.filter_mask[-1] * collision_filters

        # # check pcs between fingertips
        # grasping_filters = self.filter_pc_existence(self.info.quaternions[-1, ...], self.info.translations[-1,...], pc.cpu().numpy(), num_points=1)
        # self.info.filter_mask[-1] = self.info.filter_mask[-1] * grasping_filters

        # # eliminate out bad score grasps
        # success = self.info.success[-1]
        # success_filter = np.ones_like(success)
        # success_filter[success < self.args.success_threshold] = 0
        # self.info.filter_mask[-1] = self.info.filter_mask[-1] * success_filter

        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {self.args.n} samples.')

        # show stats
        final_samples_count = self.info.filter_mask[-1].sum()
        print(f'Average initial success (filtered): {self.info.compute_init_success(filtered=True)}')
        print(f'Average final success (filtered): {self.info.compute_final_success(filtered=True)} for {final_samples_count} samples.')

        return self.info.get_refined_grasp()

    def refine_grasps_graspnet(self, quaternions, translations, pc, mode='refinement'):
        # Input:
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized
        # return
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized

        self.info = InfoHolder()

        if self.args.rot_reps == 'euler':
            rots = quaternions2eulers(quaternions)
        elif self.args.rot_reps == 'quaternion':
            rots = quaternions.copy()
        else:
            rots = utils.quat_to_A_vec(quaternions)

        n = translations.shape[0]
        split_size = int(n/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(n), split_size)
        #print(split_indcs)
        # refine by batch

        filter_masks = torch.ones(translations.shape[0]).to(translations.device)

        print(f'Start to refine using grapsnet ... .')
        start_time = time.time()
        for indcs in tqdm( split_indcs ):
            _pc, _rots, _translations = pc[indcs], rots[indcs], translations[indcs]
            _filter_masks = filter_masks[indcs]

            rots_v = None
            translations_v = None
            self.info.batch_init()
            for t in tqdm( range(1, self.args.max_iterations+1), leave=False):

                # get gradient
                if mode == 'refinement':
                    rots_v, translations_v, success = self._velocity(_rots,_translations, Nq=self.args.Nq, Np=self.args.Np, pc=_pc)
                elif mode == 'correction':
                    rots_v, translations_v, success = self._velocity_2(_rots,_translations, pc=_pc)
                # rots_v, translations_v, success = self._velocity(_rots,_translations, Nq=self.args.Nq, Np=self.args.Np, pc=_pc)
                
                # update info
                self.info.batch_update(_rots,rots_v, _translations, translations_v, success, _filter_masks, self.args.rot_reps)

                # update
                norm_t = torch.norm(translations_v, p=2, dim=-1).to(self.device)
                alpha = torch.min(0.01 / norm_t, torch.tensor(1, dtype=torch.float).to(self.device)).float()
                
                # update
                _rots = _rots.data + rots_v * alpha[:, None]
                _translations = _translations.data + translations_v * alpha[:, None]

            # add to the info last updates
            last_batch_success = self.evaluate_grasps(_rots, _translations, _pc)
            self.info.batch_update(_rots,None, _translations, None, last_batch_success, None, self.args.rot_reps)
            self.info.update()

        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)

        # final filter to get rid off colluding samples
        collision_filters = self.filter_collision_grasps(self.info.quaternions[-1, ...], self.info.translations[-1,...], pc.cpu().numpy())
        self.info.filter_mask[-1] = self.info.filter_mask[-1] * collision_filters

        # check pcs between fingertips
        grasping_filters = self.filter_pc_existence(self.info.quaternions[-1, ...], self.info.translations[-1,...], pc.cpu().numpy(), num_points=1)
        self.info.filter_mask[-1] = self.info.filter_mask[-1] * grasping_filters

        # # eliminate out bad score grasps
        # success = self.info.success[-1]
        # success_filter = np.ones_like(success)
        # success_filter[success < self.args.success_threshold] = 0
        # self.info.filter_mask[-1] = self.info.filter_mask[-1] * success_filter

        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {self.args.n} samples.')

        # show stats
        final_samples_count = self.info.filter_mask[-1].sum()
        print(f'Average initial success (filtered): {self.info.compute_init_success(filtered=True)}')
        print(f'Average final success (filtered): {self.info.compute_final_success(filtered=True)} for {final_samples_count} samples.')

        return self.info.get_refined_grasp()


    def evaluate_all_grasps(self, quaternions, translations, pc):
        # Input:
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized
        # return
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized
        

        if self.args.rot_reps == 'euler':
            rots = quaternions2eulers(quaternions)
        elif self.args.rot_reps == 'quaternion':
            rots = quaternions.copy()
        else:
            rots = utils.quat_to_A_vec(quaternions)

        n = translations.shape[0]
        split_size = int(n/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(n), split_size)
        #print(split_indcs)
        # refine by batch

        print(f'Start to evaluate ... .')
        start_time = time.time()

        all_scores = []
        for indcs in tqdm( split_indcs ):
            _pc, _rots, _translations = pc[indcs], rots[indcs], translations[indcs]

            # add to the info last updates
            scores = self.evaluate_grasps(_rots, _translations, _pc)
            scores = scores.cpu().numpy()
            all_scores.append(scores)

        all_scores = np.concatenate(all_scores)

        

        exec_time=time.time()-start_time



        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {self.args.n} samples.')

        # show stats

        return all_scores

    def refine_grasps(self, quaternions, translations, pc, mode='refinement'):
        # Input:
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized
        # return
        # quats: [B, 4]
        # trans: [B,3]
        # pc: [B, 1024, 3] - normalized

        self.info = InfoHolder()

        if self.args.rot_reps == 'euler':
            rots = quaternions2eulers(quaternions)
        elif self.args.rot_reps == 'quaternion':
            rots = quaternions.copy()
        else:
            rots = utils.quat_to_A_vec(quaternions)

        n = translations.shape[0]
        split_size = int(n/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(n), split_size)
        #print(split_indcs)
        # refine by batch

        filter_masks = torch.ones(translations.shape[0]).to(translations.device)

        print(f'Start to refine using DGflow ... .')
        start_time = time.time()
        for indcs in tqdm( split_indcs ):
            _pc, _rots, _translations = pc[indcs], rots[indcs], translations[indcs]
            _filter_masks = filter_masks[indcs]

            rots_v = None
            translations_v = None
            self.info.batch_init()
            for t in tqdm( range(1, self.args.max_iterations+1), leave=False):

                # get gradient
                if mode == 'refinement':
                    rots_v, translations_v, success = self._velocity(_rots,_translations, Nq=self.args.Nq, Np=self.args.Np, pc=_pc)
                elif mode == 'correction':
                    rots_v, translations_v, success = self._velocity_2(_rots,_translations, pc=_pc)
                # rots_v, translations_v, success = self._velocity(_rots,_translations, Nq=self.args.Nq, Np=self.args.Np, pc=_pc)

                # apply filter
                _filter_masks = self.filter1(rots_v, translations_v, success, _filter_masks)
                
                # update info
                self.info.batch_update(_rots,rots_v, _translations, translations_v, success, _filter_masks, self.args.rot_reps)

                # update
                delta_rot = self.args.eta_rots * rots_v + np.sqrt(2*self.args.eta_rots) * self.args.noise_factor * torch.randn_like(_rots)
                if mode == 'refinement':
                    delta_rot = torch.mul(_filter_masks, delta_rot.permute(1,0)).permute(1,0)
                _rots = _rots.data + delta_rot

                delta_trans = self.args.eta_trans * translations_v + np.sqrt(2*self.args.eta_trans) * self.args.noise_factor * torch.randn_like(_translations)
                if mode == 'refinement':
                    delta_trans = torch.mul(_filter_masks, delta_trans.permute(1,0)).permute(1,0)
                _translations = _translations.data + delta_trans

            # add to the info last updates
            last_batch_success = self.evaluate_grasps(_rots, _translations, _pc)
            self.info.batch_update(_rots,None, _translations, None, last_batch_success, None, self.args.rot_reps)
            self.info.update()

        exec_time=time.time()-start_time

        # conclude update
        self.info.conclude(exec_time=exec_time)

        # final filter to get rid off colluding samples
        collision_filters = self.filter_collision_grasps(self.info.quaternions[-1, ...], self.info.translations[-1,...], pc.cpu().numpy())
        self.info.filter_mask[-1] = self.info.filter_mask[-1] * collision_filters

        # check pcs between fingertips
        grasping_filters = self.filter_pc_existence(self.info.quaternions[-1, ...], self.info.translations[-1,...], pc.cpu().numpy(), num_points=1)
        self.info.filter_mask[-1] = self.info.filter_mask[-1] * grasping_filters

        # # eliminate out bad score grasps
        # success = self.info.success[-1]
        # success_filter = np.ones_like(success)
        # success_filter[success < self.args.success_threshold] = 0
        # self.info.filter_mask[-1] = self.info.filter_mask[-1] * success_filter


        # output is in [seq, B, :] format
        print(f'Done with refinement in {exec_time} seconds for {self.args.n} samples.')

        # show stats
        final_samples_count = self.info.filter_mask[-1].sum()
        print(f'Average initial success (filtered): {self.info.compute_init_success(filtered=True)}')
        print(f'Average final success (filtered): {self.info.compute_final_success(filtered=True)} for {final_samples_count} samples.')

        return self.info.get_refined_grasp()
        

    def filter_collision_grasps(self, quaternions, translations, pc):
        '''
        Checks collision with Gripper
        returns zero if does not satisfy
        '''
        is_collision_free = []
        
        for i, _pc in enumerate(pc):
            panda_gripper = PandaGripper(root_folder='../grasper').get_combined_obbs()
            panda_gripper.apply_transform(construct_grasp_matrix(quaternions[i], translations[i]).squeeze(axis=0))
            if np.any(panda_gripper.contains(_pc)):
                is_collision_free.append(0)
            else:
                is_collision_free.append(1)

        return np.array(is_collision_free)

    def filter_pc_existence(self, quaternions, translations, pc, num_points=5):
        '''
        Checks collision with Gripper
        returns zero if does not satisfy
        '''
        
        b_trans = np.eye(4)
        b_trans[:3,3] = [0.0, 0.0 , 0.08925]
        
        is_graspable = []
        for i, _pc in enumerate(pc):
            middle_finger_volume = trimesh.primitives.Box(extents=[0.08,0.0225,0.0465], transform=b_trans)
            middle_finger_volume.apply_transform(construct_grasp_matrix(quaternions[i], translations[i]).squeeze(axis=0))
            contains_pc = middle_finger_volume.contains(_pc)
            if np.sum(contains_pc) >= num_points:
                is_graspable.append(1)
            else:
                is_graspable.append(0)

        return np.array(is_graspable)


    def filter1(self, _grad1, _grad2, success, filter_masks):
        '''
        Filters out based on the gradients and success
        returns 0 if grad1 < threshold1, grad2 < threshold2, success < threshold3
        '''
        grad_threshold = 0.5
        success_threshold = 0.3

        mask1 = torch.all(_grad1 <= grad_threshold, dim=1)
        mask2 = torch.all(_grad2 <= grad_threshold, dim=1)

        mask3 = success.squeeze(1) <= success_threshold
        mask = (mask1 & mask2 & mask3)
        filter_masks[mask] = 0.0

        return filter_masks

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


    def _velocity(self, rots, translations, pc, Nq=1, Np=1):
        # eulers [B,3]
        # translations [B,3]
        # pc [B,1024,3] - normalized

        rots_t = rots.clone()
        translations_t = translations.clone()
        rots_t.requires_grad_(True)
        translations_t.requires_grad_(True)
        if rots_t.grad is not None:
            rots_t.grad.zero_()
        if translations_t.grad is not None:
            translations_t.grad.zero_()
            
        pc.requires_grad_(False)
        if self.args.rot_reps == 'euler':
            d_score, _, _ = self.evaluator.forward_with_eulers(rots_t, translations_t, pc)
        elif self.args.rot_reps == 'quaternion':
            d_score, _, _ = self.evaluator.forward(rots_t, translations_t, pc)
        else:
            d_score, _, _ = self.evaluator.forward_with_A_vec(rots_t, translations_t, pc)
            
        success = torch.sigmoid(d_score)

        Nq = torch.FloatTensor([Nq]).to(self.device)
        Np = torch.FloatTensor([Np]).to(self.device)
        bias_term = torch.log(Nq) - torch.log(Np)
        d_score -= bias_term

        if self.args.f == 'KL':
            s_eulers = torch.ones_like(d_score.detach())
            s_translations = torch.ones_like(d_score.detach())

        elif self.args.f == 'logD':
            s_eulers = 1 / (1 + d_score.detach().exp())
            s_translations = 1 / (1 + d_score.detach().exp())

        elif self.args.f == 'JS':
            s_eulers = 1 / (1 + 1 / d_score.detach().exp())
            s_translations = 1 / (1 + 1 / d_score.detach().exp())
        else:
            raise ValueError()

        s_eulers.expand_as(rots_t)
        s_translations.expand_as(translations_t)
        d_score.backward(torch.ones_like(d_score).to(self.device))
        rots_grad = rots_t.grad
        # print(rots_grad)
        trans_grad = translations_t.grad
        # return s_eulers * eulers_grad.data, \
        #        s_translations.data * trans_grad.data, \
        #        success

        
        is_success = torch.ones(success.shape[0]).to(rots.device)
        is_success[success.squeeze(1) >= self.args.success_threshold] = 0.0

        rot_velocity = s_eulers * rots_grad.data
        trans_velocity = s_translations.data * trans_grad.data

        rot_velocity = torch.mul(is_success, rot_velocity.permute(1,0)).permute(1,0)
        trans_velocity = torch.mul(is_success, trans_velocity.permute(1,0)).permute(1,0)

        return rot_velocity, trans_velocity, success


    def _velocity_2(self, rots, translations, pc, Nq_ev=1, Np_ev=1, Nq_corr=1, Np_corr=1):
        # eulers [B,3]
        # translations [B,3]
        # pc [B,1024,3] - normalized
        rots_t = rots.clone()
        translations_t = translations.clone()
        rots_t.requires_grad_(True)
        translations_t.requires_grad_(True)
        if rots_t.grad is not None:
            rots_t.grad.zero_()
        if translations_t.grad is not None:
            translations_t.grad.zero_()
            
        pc.requires_grad_(False)
        if self.args.rot_reps == 'euler':
            d_score_ev, _, _ = self.evaluator.forward_with_eulers(rots_t, translations_t, pc)
        elif self.args.rot_reps == 'quaternion':
            d_score_ev, _, _ = self.evaluator.forward(rots_t, translations_t, pc)
        else:
            d_score_ev, _, _ = self.evaluator.forward_with_A_vec(rots_t, translations_t, pc)
            
        success = torch.sigmoid(d_score_ev)
        # Get d_score of corrector =================================================
        feature_extractor = GPFeatureExtractor().to(self.device)
        features = feature_extractor.forward_with_eulers(rots_t, translations_t, pc)
        # This needs features
        d_score_corr = self.corrector.forward(features)
        # =================================================
        #Nq_ev = torch.FloatTensor([Nq_ev]).to(self.device)
        #Np_ev = torch.FloatTensor([Np_ev]).to(self.device)
        #Nq_corr = torch.FloatTensor([Nq_corr]).to(self.device)
        #Np_corr = torch.FloatTensor([Np_corr]).to(self.device)
        
        # bias_term = torch.log(Nq_ev) - torch.log(Np_ev)
        #bias_term = torch.log(torch.Tensor([Nq_ev*Nq_corr])) - torch.log(torch.Tensor([Np_ev*Np_corr]))
        #log_r = torch.ones_like(d_score_ev).to(self.device)*bias_term
        log_r = + d_score_ev + d_score_corr# + torch.log( 1 + Nq_ev/Np_ev*d_score_ev.exp() + Nq_corr/Np_corr*d_score_corr.exp() )
        # d_score_ev -= bias_term
        
        if self.args.f == 'KL':
            s_eulers = torch.ones_like(d_score_ev.detach())
            s_translations = torch.ones_like(d_score_ev.detach())
        elif self.args.f == 'logD':
            s_eulers = 1 / (1 + d_score_ev.detach().exp())
            s_translations = 1 / (1 + d_score_ev.detach().exp())
        elif self.args.f == 'JS':
            s_eulers = 1 / (1 + 1 / d_score_ev.detach().exp())
            s_translations = 1 / (1 + 1 / d_score_ev.detach().exp())
        else:
            raise ValueError()
        s_eulers.expand_as(rots_t)
        s_translations.expand_as(translations_t)
        # d_score_ev.backward(torch.ones_like(d_score_ev).to(self.device))
        log_r.backward(torch.ones_like(log_r).to(self.device))
        rots_grad = rots_t.grad
        # print(rots_grad)
        trans_grad = translations_t.grad
        
        is_success = torch.ones(success.shape[0]).to(rots.device)
        is_success_mask = success.squeeze(1) >= self.args.success_threshold
        if self.args.method == 'DGflow':
            is_success[is_success_mask] = 0.0
        else:
            is_success[is_success_mask] = 1.0

        rot_velocity = s_eulers * rots_grad.data
        trans_velocity = s_translations.data * trans_grad.data
        rot_velocity = torch.mul(is_success, rot_velocity.permute(1,0)).permute(1,0)
        trans_velocity = torch.mul(is_success, trans_velocity.permute(1,0)).permute(1,0)
        return rot_velocity, trans_velocity, success