import csv
import torch
import numpy as np
from networks.models import GraspEvaluator, GraspEvaluatorDistance, CollisionDistanceEvaluator, FarGraspEvaluator
from theseus.geometry import SO3
from networks.quaternion import quat_normalize, euler2quat, quat2euler, euler2matrix, rot_p_by_quaterion
from tqdm.auto import tqdm
from robot_model import CombinedRobotModel2
from utils.points import regularize_pc_point_count_torch
import sys
from sklearn.decomposition import PCA
sys.path.insert(0, "../graspflow_taskgrasp/gcngrasp")
import taskgrasp_refine, taskgrasp_refine_old


class S_classifier():
    def __init__(self, path_to_model, device='cpu', batch_size=64, dist_threshold=0.1, dist_coeff=10000, approximate=False):
        '''
        S classifier for robust grasps.
        '''
        self.device = device
        self.batch_size = batch_size
        self.SO3_holder = SO3() # this makes faster
        self.load_evaluator(path_to_model=path_to_model)

        self.collision_evaluator = FarGraspEvaluator(threshold=dist_threshold,
                                                    coeff=dist_coeff)

    def load_evaluator(self, path_to_model):
        model = GraspEvaluator()
        model.load_state_dict(torch.load(path_to_model))
        model = model.to(self.device)
        self.evaluator = model.eval()

    def forward(self, t, r, pc, grasp_space="SO3"):


        if pc.shape[1] != 1024:
            pc = regularize_pc_point_count_torch(pc, npoints=1024)
            # pc = torch.FloatTensor(pc).to(t.device)
        if grasp_space == 'SO3':
            r = self.SO3_holder.exp_map(r).to_quaternion()
            r = quat_normalize(r)
            logit = self.collision_evaluator.forward(t, r, pc)
            logit2, _ , _= self.evaluator.forward(r, t, pc)
 
        elif grasp_space == 'Euler':
            logit, _, _  = self.collision_evaluator.forward_with_eulers(r, t, pc)
            logit2, _ , _= self.evaluator.forward_with_eulers(r, t, pc)
        elif grasp_space == 'Quaternion':
            logit, _, _  = self.collision_evaluator.forward(r, t, pc)
            logit2, _ , _= self.evaluator.forward(r, t, pc)
        else:
            raise ValueError("Not implemented!")

        
        logit[logit > 0] = logit2[logit > 0]
        scores = torch.sigmoid(logit)

        return logit, scores

    def evaluate_all(self, t, q, pc, grasp_space="SO3"):

        B = q.shape[0]
        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        all_success = []
        with torch.no_grad():
            for indcs in tqdm(split_indcs, disable=True):
                _pc, _q, _t = pc[indcs], q[indcs], t[indcs]

                _, success = self.forward(_t, _q, _pc, grasp_space=grasp_space)
                all_success.append(success.cpu().numpy().copy())

        success = np.concatenate(all_success, axis=0)
        return success

    def __print__(self):
        print("S classifier for robust grasps.")

# class S_classifier():
#     def __init__(self, path_to_model, device='cpu', batch_size=64):
#         '''
#         S classifier for robust grasps.
#         '''
#         self.device = device
#         self.batch_size = batch_size
#         self.SO3_holder = SO3() # this makes faster
#         self.load_evaluator(path_to_model=path_to_model)

#     def load_evaluator(self, path_to_model):
#         model = GraspEvaluator()
#         model.load_state_dict(torch.load(path_to_model))
#         model = model.to(self.device)
#         self.evaluator = model.eval()

#     def forward(self, t, r, pc, grasp_space="SO3"):


#         if pc.shape[1] != 1024:
#             pc = regularize_pc_point_count_torch(pc, npoints=1024)
#             # pc = torch.FloatTensor(pc).to(t.device)
#         if grasp_space == 'SO3':
#             r = self.SO3_holder.exp_map(r).to_quaternion()
#             r = quat_normalize(r)
#             logit, _ , _= self.evaluator.forward(r, t, pc)
#         elif grasp_space == 'Euler':
#             logit, _, _  = self.evaluator.forward_with_eulers(r, t, pc)
#         elif grasp_space == 'Quaternion':
#             logit, _, _  = self.evaluator.forward(r, t, pc)
#         else:
#             raise ValueError("Not implemented!")

#         scores = torch.sigmoid(logit)

#         return logit, scores

#     def evaluate_all(self, t, q, pc, grasp_space="SO3"):

#         B = q.shape[0]
#         # split into batches
#         split_size = int(B/(self.batch_size+1))+1
#         split_indcs = np.array_split(np.arange(B), split_size)

#         all_success = []
#         with torch.no_grad():
#             for indcs in tqdm(split_indcs, disable=True):
#                 _pc, _q, _t = pc[indcs], q[indcs], t[indcs]

#                 _, success = self.forward(_t, _q, _pc, grasp_space=grasp_space)
#                 all_success.append(success.cpu().numpy().copy())

#         success = np.concatenate(all_success, axis=0)
#         return success

#     def __print__(self):
#         print("S classifier for robust grasps.")


class C_classifier():
    def __init__(self, device='cpu', batch_size=64, threshold=0.15, coeff=1000, isTable=True, approximate=False,
                base_extension=0.2, fill_between_fingers=True):
        '''
        C classifier: Collision classifier with table only
        '''
        self.device = device
        self.batch_size = batch_size
        self.SO3_holder = SO3() # this makes faster
        if isTable:
            evaluator = GraspEvaluatorDistance(z_threshold=threshold, coeff=coeff).to(self.device)
        else:
            evaluator = CollisionDistanceEvaluator(dist_coeff=coeff, dist_threshold=threshold,
                                                   approximate=approximate, base_extension=base_extension,
                                                   fill_between_fingers=fill_between_fingers).to(self.device)

        self.evaluator = evaluator.eval()
    def forward(self, t, r, pc_mean, pc_env, grasp_space="SO3"):
        t = t + pc_mean # denormalize 
        if grasp_space == 'SO3':
            r = self.SO3_holder.exp_map(r).to_quaternion()
            r = quat_normalize(r)
            logit = self.evaluator.forward(t, r, pc_env=pc_env)
        elif grasp_space == 'Euler':
            logit  = self.evaluator.forward_with_eulers(t, r, pc_env=pc_env)
        elif grasp_space == 'Quaternion':
            logit  = self.evaluator.forward(t, r, pc_env=pc_env)
        else:
            raise ValueError("Not implemented!")

        scores = torch.sigmoid(logit)

        return logit, scores

    def evaluate_all(self, t, q, pc_mean, pc_env,  grasp_space="SO3"):

        B = q.shape[0]
        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)

        all_success = []
        with torch.no_grad():
            for indcs in tqdm(split_indcs, disable=True):
                _q, _t, _pc_mean, _pc_env = q[indcs], t[indcs], pc_mean[indcs], pc_env[indcs]

                _, success = self.forward(_t, _q, _pc_mean, _pc_env, grasp_space=grasp_space)
                all_success.append(success.cpu().numpy().copy())

        success = np.concatenate(all_success, axis=0)
        return success

    def __print__(self):
        print("C classifier: classifier that measures collision with the environment.")

class E_classifier():
    def __init__(self, device='cpu', batch_size=64,
                       mi_coeff=10000, mi_threshold=0.01,
                       t_threshold=0.05, q_threshold=0.2,
                       w_coeff=10000):
        '''
        E classifier: 1) Keeps away from singular points. 2) Pushes grasps to robot's workspace
        '''
        self.device = device
        self.batch_size = batch_size
        self.SO3_holder = SO3() # this makes faster
        self.evaluator = CombinedRobotModel2(device=self.device, 
                                            mi_coeff=mi_coeff, mi_threshold=mi_threshold,
                                            t_threshold=t_threshold, q_threshold=q_threshold,w_coeff=w_coeff)

    def forward(self, t, r, pc_mean, grasp_space="SO3"):
        t = t + pc_mean # denormalize 
        if grasp_space == 'SO3':
            r = self.SO3_holder.exp_map(r).to_quaternion()
            r = quat_normalize(r)
            logit = self.evaluator(t, r)
        elif grasp_space == 'Euler':
            r=euler2quat(r, order="XYZ")
            logit = self.evaluator(t, r)
        elif grasp_space == 'Quaternion':
            logit  = self.evaluator(t, r)
        else:
            raise ValueError("Not implemented!")

        scores = torch.sigmoid(logit)

        return logit, scores

    def evaluate_all(self, t, q, pc_mean, grasp_space="SO3"):

        B = q.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        all_success = []

        with torch.no_grad():
            for indcs in tqdm(split_indcs, disable=True):
                _q, _t, _pc_mean = q[indcs], t[indcs], pc_mean[indcs]
                _, success = self.forward(_q, _t, _pc_mean, grasp_space=grasp_space)
                all_success.append(success.cpu().numpy().copy())

        success = np.concatenate(all_success, axis=0)
        return success

    def __print__(self):
        print("E classifier: 1) Keeps away from singular points. 2) Pushes grasps to robot's workspace")

class D_classifier():
    def __init__(self, device, batch_size, alpha_threshold=0.2):
        '''
        Top-down classifier:
        '''
        self.device = device
        self.batch_size = batch_size
        self.SO3_holder = SO3() # this makes faster
        self.alpha_threshold = alpha_threshold

    def forward(self, t, r, grasp_space="SO3"):

        if grasp_space == 'SO3':
            r = self.SO3_holder.exp_map(r).to_quaternion()
            r = quat_normalize(r)
        elif grasp_space == 'Euler':
            r=euler2quat(r, order="XYZ")
        elif grasp_space == 'Quaternion':
            pass
        else:
            raise ValueError("Not implemented!")

        z_vector = torch.zeros_like(t)
        z_vector[:,2] = 1.0

        direction = rot_p_by_quaterion(z_vector.unsqueeze(1), r.unsqueeze(1)).squeeze(1)
        # get angle
        angle = torch.arccos((-1.0 * direction[:, 2]))

        logit = 1000*(self.alpha_threshold-angle) + (t-t).sum(dim=1)
        logit = logit.unsqueeze(-1)

        scores = torch.sigmoid(logit)

        return logit, scores

class N_classifier():
    def __init__(self, device, batch_size, dist_threshold=0.03, dist_coeff=10000):
        '''
        NLP classifier:
        '''
        self.device = device
        self.batch_size = batch_size
        self.SO3_holder = SO3() # this makes faster

        self.gcn_model = taskgrasp_refine.load_gcn_model(device=self.device)
        self.intent_model = taskgrasp_refine.load_intent_model(device=self.device)
        self.collision_evaluator = FarGraspEvaluator(threshold=dist_threshold,
                                                    coeff=dist_coeff)

        self.intent_labels = np.array(['UNK', 'handover', 'pour', 'use', 'hit', 'cut', 'dispense', 'shake', 'mix'])
        self.intent_logits = dist_coeff * torch.tensor([0, 1, 0, 0, 1, 0, 0, 1, 0]).to(device)

    def forward(self, t, r, pc, query, grasp_space="SO3"):

        #query = query.squeeze(axis=1)

        #  denormalize the translation and pc
        # t = t + pc_mean
        # pc = pc + pc_mean.unsqueeze(1)
        intent_preds = query.astype(int).flatten()
        intents = self.intent_labels[intent_preds]
        

        if grasp_space == 'SO3':
            r = self.SO3_holder.exp_map(r).to_quaternion()
            r = quat_normalize(r)
        elif grasp_space == 'Euler':
            r=euler2quat(r, order="XYZ")
        else:
            raise ValueError("Not implemented!")

        logit, scores = taskgrasp_refine.evaluate(self.gcn_model, t, r, pc, intents)

        # +logits for certain queries
        intent_pred_modifier = self.intent_logits[[intent_preds]]
        logit = logit + intent_pred_modifier

        logit = logit.unsqueeze(-1)
        scores = scores.unsqueeze(-1)


        logit_dist = self.collision_evaluator.forward(t, r, pc)
        
        logit_dist[logit_dist > 0] = logit[logit_dist > 0]
        scores = torch.sigmoid(logit_dist)

        # NOTE: both logit and scores shall be [B,1]

        return logit_dist, scores

    def get_intents(self, query):
        query = query.squeeze(-1)
        _, intent_preds = np.array(taskgrasp_refine.get_intents(query = query, model=self.intent_model))
        intent_preds = np.array(intent_preds).astype(np.int)
        return intent_preds

    def evaluate_all(self, t, q, pc, pc_mean, query, grasp_space="SO3"):

        B = q.shape[0]
        query = query.squeeze(-1)
        intents, intent_preds = np.array(taskgrasp_refine.get_intents(query = query)) # [B, 1]
        intent_preds = np.array(intent_preds)

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        all_success = []
        logits = []

        with torch.no_grad():
            for indcs in tqdm(split_indcs, disable=True):
                _q, _t, _pc, _pc_mean, _query = q[indcs], t[indcs], pc[indcs], pc_mean[indcs], intents[indcs]
                l, success = self.forward(_t, _q, _pc, intent_preds[indcs], _query, grasp_space=grasp_space)
                logits.append(l.cpu())
                all_success.append(success.cpu().numpy().copy())

        success = np.concatenate(all_success, axis=0)
        return success, logits

    def __print__(self):
        print("N (a.k.k NLP) classifier: Biases grasps towards users' intend")


class T_classifier():
    def __init__(self, device='cpu', batch_size=64, cg_mean_init=0, cg_std_init=1):
        '''
        Tactile classifier:
        '''
        self.device = device
        self.batch_size = batch_size
        self.SO3_holder = SO3() # this makes faster

        self.mean_estimate = torch.FloatTensor([0.0])
        self.mean_estimate.requires_grad_(True)
        self.std_estimate = torch.FloatTensor([1.0])
        self.std_estimate.requires_grad_(True)
        
        self.likelihood = torch.distributions.Normal(self.mean_estimate, self.std_estimate)
        self.mean_prior = torch.distributions.Normal(cg_mean_init, torch.FloatTensor([10.0]))
        self.std_prior = torch.distributions.Normal(cg_std_init, torch.FloatTensor([1.0]))
        
        self.observations = []
        self.longest_axis_vector = None

    def evaluator(self, t, rotations, cg_belief):
        unit_vector = torch.zeros([t.shape[0], 3])
        unit_vector[:, 2] = 0.11
        unit_vector = unit_vector.to(self.device)
        unit_vector = torch.einsum('ik,ijk->ij', unit_vector, rotations)

        dist = (t + unit_vector - cg_belief).pow(2).sum(1).sqrt()
        logit = 1000 * (0.015 - dist)
        
        success = torch.sigmoid(logit) # compute success
        return logit, success.reshape((success.shape[0], 1))

    def forward(self, t, r, pc, grasp_space="SO3"):
        pc = pc[0] # ignore batch, (1024, 3)

        if grasp_space == 'SO3':
            r = self.SO3_holder.exp_map(r).to_quaternion()
            r = quat_normalize(r)
            rotations = euler2matrix(quat2euler(r, order='XYZ'), order='XYZ')
        elif grasp_space == 'Euler':
            rotations = euler2matrix(r, 'XYZ')
        elif grasp_space == 'Quaternion':
            rotations = euler2matrix(quat2euler(r))
        else:
            raise ValueError("Not implemented!")

        if self.longest_axis_vector is None:
            pc_2d = pc[:, :2].cpu().numpy() # ignore z-axis
            self.pc_2d_mean = np.mean(pc_2d, axis=0)
            pca = PCA(n_components=2)
            pca.fit(pc_2d)
            comp = pca.components_[0]
            var = pca.explained_variance_[0]
            comp = comp * 0.06 # * var * 30

            unit_vector = np.zeros([1, 3])
            unit_vector[0, 1] = 0.1
            unit_vector = np.einsum('ik,ijk->ij', unit_vector, rotations.detach().cpu().numpy())[0, :2]

            angle = unit_vector.dot(comp) / (np.sqrt(unit_vector.dot(unit_vector)) * np.sqrt(comp.dot(comp)))
            self.longest_axis_vector = (-1 if angle > 0 else 1) * comp
        
        cg_belief = torch.FloatTensor([
            self.pc_2d_mean[0] + (self.mean_estimate.item() * self.longest_axis_vector[0]), 
            self.pc_2d_mean[1] + (self.mean_estimate.item() * self.longest_axis_vector[1]),
            pc.mean(0)[2]
        ]).to(self.device)

        logit, scores = self.evaluator(t, rotations, cg_belief)

        return logit, scores

    def update(self, grasp_tran, rotation_speed, z_axis_force):
        print(f"Before: CG mean estimate {self.mean_estimate.data}, std estimate: {self.std_estimate.data}")
        scaling_factor = 1
        self.observations.append(rotation_speed * scaling_factor) #* (0.5 if rotation_speed > 0.5 else 1))
        
        prior_ = self.mean_prior.log_prob(self.mean_estimate) + self.std_prior.log_prob(self.std_estimate)
        posterior = torch.mean(self.likelihood.log_prob(torch.FloatTensor(self.observations))) + prior_

        if np.isnan(posterior.data[0]) or np.isnan(prior_.data[0]):
            return

        posterior.backward()
        for param in (self.mean_estimate, self.std_estimate):
            param.data.add_(1 * param.grad.data)
            param.grad.data.zero_()
        print(f"After: CG mean estimate {self.mean_estimate.data}, std estimate: {self.std_estimate.data}")

    def update_all(self):
        with open("/home/crslab/GRASP/graspflow_models/experiments/tactile_observations.csv", "r") as f:
            reader = csv.reader(f)
            for i, line in enumerate(reader):
                rotation_speed = float(line[0])
                print(f"Updating with {rotation_speed}")
                self.update(None, rotation_speed, None)


    def evaluate_all(self, t, q, pc, pc_mean, grasp_space="SO3"):

        B = q.shape[0]

        # split into batches
        split_size = int(B/(self.batch_size+1))+1
        split_indcs = np.array_split(np.arange(B), split_size)
        all_success = []

        with torch.no_grad():
            for indcs in tqdm(split_indcs, disable=True):
                _q, _t, _pc, _pc_mean = q[indcs], t[indcs], pc[indcs], pc_mean[indcs]
                _, success = self.forward(_q, _t, _pc, grasp_space=grasp_space)
                all_success.append(success.cpu().numpy().copy())

        success = np.concatenate(all_success, axis=0)
        return success

    def __print__(self):
        print("Tactile classifier: Biases grasps cg of the objects")

if __name__ == "__main__":

    C = C_classifier(batch_size=5, device='cpu', isTable=False)

    ts =torch.FloatTensor([[[0.5803, 0.1571, 0.5528],
        [0.6453, 0.0235, 0.5180],
        [0.5560, 0.1028, 0.4921]]])
    qs = torch.FloatTensor([[[ 0.6787,  0.6463,  0.2130,  0.2761],
            [ 0.6258, -0.0433,  0.1526, -0.7637],
            [ 0.5746,  0.5956,  0.4869,  0.2794]]])

    


    
    
    # # S = S_classifier(batch_size=5, path_to_model='saved_models/evaluator/165228576343/100.pt', device='cpu')
    # # T = T_classifier(batch_size=5, device='cpu')
    # # E = E_classifier(batch_size=5, device='cpu')
    N = N_classifier(batch_size=5, device='cpu')

    # t=torch.rand([4,3])+100
    # r=torch.rand([4,3])
    # pc = torch.rand([4,1024,3])
    # pc_mean = torch.rand([4,3])
    # query = np.array(["Hand me the scissors", "I want you to pour me some water", "Can you cut some fruit",
    # "Please give me the knife on the left"])
    # # #a= N.evaluate_all(t, r, pc_mean, grasp_space="SO3")
    # np.savez("test_pc_base.npz", pc = pc, pc_mean = pc_mean, t = t, r = r, query = query)
    # b, logits= N.evaluate_all(t, r, pc, pc_mean, query, grasp_space="SO3")
    device = torch.device('cuda:0')
    dddata = np.load("test_pc_base.npz")
    pc = torch.tensor(dddata["pc"])
    pc_mean = torch.tensor(dddata['pc_mean'])
    t = torch.tensor(dddata['t'])
    r = torch.tensor(dddata['r'])
    r = N.SO3_holder.exp_map(r).to_quaternion()
    r = quat_normalize(r)
    query = dddata['query']
    dddata = np.load("test_pc.npz")
    logits_target = dddata['logits']
    
    # #print(pc)
    #b, logits= N.evaluate_all(t, r, pc, pc_mean, query, grasp_space="SO3")
    logits, scores, all= taskgrasp_refine_old.evaluate(t, r, pc, query)

    dddata = np.load("test_pc_base.npz")
    pc = torch.tensor(dddata["pc"], dtype=torch.float).to(device)
    pc_mean = torch.tensor(dddata['pc_mean']).to(device)
    t = torch.tensor(dddata['t'], dtype=torch.float).to(device)
    r = torch.tensor(dddata['r'], dtype=torch.float).to(device)
    r = N.SO3_holder.exp_map(r).to_quaternion()
    r = quat_normalize(r)
    query = dddata['query']
    dddata = np.load("test_pc.npz")
    logits_target = dddata['logits']
    logits2, scores2= taskgrasp_refine.evaluate(t, r, pc, query) # currently wrong?

    print(logits, logits2)
    
    # edges = []
    # for i in range(len(all)):
    #     edges.append(all[i][-1])
    # edges = torch.cat(edges, dim=1)



    # dddata = np.load("test_pc_base_random.npz")
    # #np.savez("test_pc_base_random.npz", pc = x.cpu(), node_x_idx = node_x_idx.cpu(), latent = latent.cpu(), edge_index = edge_index.cpu())
    # pc = torch.tensor(dddata["pc"])
    # pc_mean = torch.tensor(dddata['node_x_idx'])
    # t = torch.tensor(dddata['latent'])
    # r = torch.tensor(dddata['edge_index'])

    # b, logits= taskgrasp_refine_old.score(pc, pc_mean, t, r)
    # # print(b)

    # # test C classifier 

    # t.requires_grad_(True)
    # r.requires_grad_(True)

    # print(t)
    # print(r)
    # with torch.autograd.set_detect_anomaly(True):

    #     pc.requires_grad_(False)
    #     pc_mean.requires_grad_(False)

    #     b, scores = C.forward(t, r, pc_mean, pc, grasp_space='SO3')

    #     print(scores)
    #     print(b.shape)

    #     b.backward(torch.ones_like(b))

    #     print(t.grad)
    #     print(r.grad)



    #     # b= C.evaluate_all(t, r, pc_mean, pc, grasp_space="SO3")

