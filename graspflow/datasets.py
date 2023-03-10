#!/usr/bin/env python3
# Tasbolat Taunyazov

from numpy import random
import torch
from torch.utils.data import Dataset
import numpy as np
import yaml
import pickle
from pathlib import Path
from utils.points import regularize_pc_point_count
import networks.quaternion as quat_ops
from numpy import genfromtxt


class SemanticGraspDataset(Dataset):
    def __init__(self, path_to_grasps = 'data/semantic_data/preprocessed2', path_to_pc='data/pcs', split='train', allowed_categories = ['mug',"hammer","fork"], augment=True, full_pc=False): #, 

        # load grasp_data
        # load grasp_data
        quaternions = []
        translations = []
        labels = []
        metadata = []
        categories = []
        for cat in allowed_categories:
            quaternions.append( np.load(f'{path_to_grasps}/{cat}/quaternions_{split}.npy') )
            translations.append( np.load(f'{path_to_grasps}/{cat}/translations_{split}.npy') )
            labels.append( np.load(f'{path_to_grasps}/{cat}/labels_{split}.npy') )
            metadata.append( np.load(f'{path_to_grasps}/{cat}/metadata_{split}.npy') )
            categories = categories + [cat]*len(metadata[-1])

        self.quaternions = torch.FloatTensor(np.concatenate(quaternions))
        self.translations = torch.FloatTensor(np.concatenate(translations))
        labels = np.concatenate(labels)
        self.labels = torch.FloatTensor(labels)
        self.metadata = np.concatenate(metadata)
        self.categories = categories

        self.pos_count = torch.sum(self.labels).item()
        self.pos_indcs = np.where(labels == 1)[0]
        self.neg_indcs = np.where(labels == 0)[0]
        self.total_count = len(self.labels)
        self.size = int(self.pos_count*2) # 1 to 1 POS to NEG ratio
        print(f'total {self.total_count}')

        # load pcs
        self.pcs = {}
        for cat in allowed_categories:
            self.pcs[cat] = [] 
            for k in range(21):
                try:
                    pc = pickle.load(open(f'{path_to_pc}/{cat}/{cat}{k:03}.pkl', 'rb'))
                    self.pcs[cat].append(pc["pcs"])
                    # print("load pcl")
                except:
                    pass

        # print("Done with loading pcl")
        
        # read uniform quaternions
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')
        self.augment = augment
        self.full_pc = full_pc


    def __getitem__(self, index):
        # subsample negatives
        if index < self.pos_count-1:
            index = self.pos_indcs[index]
        else:
            index = np.random.randint(low=0, high=len(self.neg_indcs), size=1)[0]
            index = self.neg_indcs[index]

        # prepare pc
        cat = self.categories[index]
        obj_idx = int(self.metadata[index])
        pc_index = np.random.randint(low=0, high=999, size=1)[0]
        if self.full_pc:
            pc = self.pcs[cat][obj_idx][0:30]
            pc = np.concatenate(pc)
        else:
            view_size = np.random.randint(low=1, high=4, size=1)[0]
            pc_indcs = np.random.randint(low=0, high=999, size=view_size)
            
            if len(pc_indcs) == 1:
                pc = self.pcs[cat][obj_idx][pc_indcs[0]]
            else:
                __pcs = []
                for pc_index in pc_indcs:
                    __pcs = __pcs + [self.pcs[cat][obj_idx][pc_index]]
                pc = np.concatenate( __pcs )
        pc = regularize_pc_point_count(pc, 1024, False)

        # jitter point cloud here
        pc = add_jitter(pc)

        pc = torch.FloatTensor( pc )
        

        _quat = self.quaternions[index]
        _trans = self.translations[index]
        # augmentation
        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               cat

    def __len__(self):
        return self.size

class GraspDatasetBox(Dataset):
    def __init__(self, path_to_grasps = 'data/grasps/preprocessed_one_box',
                     path_to_pc='data/pcs/',
                      split='train',
                      augment=False):
        
        # load grasps
        quaternions = np.load(f'{path_to_grasps}/quaternions_{split}.npy')
        translations = np.load(f'{path_to_grasps}/translations_{split}.npy')
        isaac_labels = np.load(f'{path_to_grasps}/isaac_labels_{split}.npy')

        # load pointcloud
        pcs = pickle.load(open(f'{path_to_pc}/box/box014.pkl', 'rb'))['pcs']
        self.pc = np.concatenate(pcs[:10])
        


        self.size = int( np.sum(isaac_labels)*10 )
        self.pos_indcs = np.where(isaac_labels==1)[0]
        self.pos_size = np.sum(isaac_labels)

        self.quaternions = torch.FloatTensor(quaternions)
        self.translations = torch.FloatTensor(translations)
        self.labels = torch.FloatTensor(isaac_labels)

        # augmentation
        self.augment = augment
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')

    def __getitem__(self, index):
        if index < self.pos_size:
            index = np.random.choice(self.pos_indcs, size=1)[0]

        _quat = self.quaternions[index]
        _trans = self.translations[index]


        pc = torch.FloatTensor( regularize_pc_point_count(self.pc, 1024) )
        pc = add_jitter(pc)
        

        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               'box'

    def __len__(self):
        return self.size


class GraspDataset(Dataset):
    def __init__(self, path_to_grasps = 'data/grasps_new/preprocessed', path_to_pc='data/pcs', split='train', allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor'], augment=True, full_pc=False):

        # load grasp_data
        # load grasp_data
        quaternions = []
        translations = []
        labels = []
        metadata = []
        categories = []
        for cat in allowed_categories:
            quaternions.append( np.load(f'{path_to_grasps}/{cat}/quaternions_{split}.npy') )
            translations.append( np.load(f'{path_to_grasps}/{cat}/translations_{split}.npy') )
            labels.append( np.load(f'{path_to_grasps}/{cat}/isaac_labels_{split}.npy') )
            metadata.append( np.load(f'{path_to_grasps}/{cat}/metadata_{split}.npy') )
            categories = categories + [cat]*len(metadata[-1])

        self.quaternions = torch.FloatTensor(np.concatenate(quaternions))
        self.translations = torch.FloatTensor(np.concatenate(translations))
        labels = np.concatenate(labels)
        self.labels = torch.FloatTensor(labels)
        self.metadata = np.concatenate(metadata)
        self.categories = categories

        self.pos_count = torch.sum(self.labels).item()
        self.pos_indcs = np.where(labels == 1)[0]
        self.neg_indcs = np.where(labels == 0)[0]
        self.total_count = len(self.labels)
        self.size = int(self.pos_count*2) # 1 to 1 POS to NEG ratio
        print(f'total {self.total_count}')

        # load pcs
        self.pcs = {}
        for cat in allowed_categories:
            self.pcs[cat] = [] 
            for k in range(21):
                try:
                    pc = pickle.load(open(f'{path_to_pc}/{cat}/{cat}{k:03}.pkl', 'rb'))
                    self.pcs[cat].append(pc)
                except:
                    pass

        # read uniform quaternions
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')
        self.augment = augment
        self.full_pc = full_pc

    def __getitem__(self, index):
        # subsample negatives
        if index < self.pos_count-1:
            index = self.pos_indcs[index]
        else:
            index = np.random.randint(low=0, high=len(self.neg_indcs), size=1)[0]
            index = self.neg_indcs[index]

        # prepare pc
        cat = self.categories[index]
        obj_idx = int(self.metadata[index])
        pc_index = np.random.randint(low=0, high=999, size=1)[0]
        if self.full_pc:
            pc = self.pcs[cat][obj_idx][0:30]
            pc = np.concatenate(pc)
        else:
            pc = self.pcs[cat][obj_idx][pc_index]

        pc = regularize_pc_point_count(pc, 1024, False)

        # jitter point cloud here
        pc = add_jitter(pc)

        pc = torch.FloatTensor( pc )

        _quat = self.quaternions[index]
        _trans = self.translations[index]
        # augmentation
        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               cat

    def __len__(self):
        return self.size


class GraspDatasetWithTight(Dataset):
    def __init__(self, path_to_grasps = 'data/grasps/preprocessed',
                       path_to_grasps_tight = 'data/grasps_tight/preprocessed',
                       path_to_pc='data/pcs',
                       split='train', 
                       allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor'], 
                       augment=True, 
                       full_pc=False,
                       mode=2):

        assert mode in [0,1,2], print(f'Please choose mode from [0,1,2], given {mode}')

        if mode == 0: # only data 1
            path1 = path_to_grasps
        elif mode == 1: # only data tight
            path1 = path_to_grasps_tight
        else: # both
            path1 = path_to_grasps
            path2 = path_to_grasps_tight

        # load grasp_data
        self.quaternions, self.translations, self.labels, self.metadata, self.categories = load_data(path1, split, allowed_categories=allowed_categories)

        if mode == 2:
            _quaternions, _translations, _labels, _metadata, _categories = load_data(path2, split, allowed_categories=allowed_categories)
            self.quaternions = torch.cat([self.quaternions, _quaternions], 0)
            self.translations = torch.cat([self.translations, _translations], 0)
            self.labels = torch.cat([self.labels, _labels], 0)
            self.metadata = np.concatenate([self.metadata, _metadata], 0)
            self.categories = self.categories + _categories

            del _quaternions, _translations, _labels, _metadata, _categories

        
        self.pos_indcs = np.where(self.labels.numpy() == 1)[0]
        self.pos_count = len(self.pos_indcs)
        self.neg_indcs = np.where(self.labels.numpy() == 0)[0]
        self.total_count = len(self.labels)
        self.size = int(self.pos_count*2) # 1 to 1 POS to NEG ratio
        
        print(f'Total: {self.total_count}')
        print(f'Positives: {self.pos_count}, negatives: {self.size - self.pos_count}')
        print(f'Negatives will be sampled from: {len(self.neg_indcs)}')

        # load pcs
        self.pcs = load_pcs(path_to_pc, allowed_categories)

        # read uniform quaternions
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')
        self.augment = augment
        self.full_pc = full_pc

    def __getitem__(self, index):
        # subsample negatives
        if index < self.pos_count-1:
            index = self.pos_indcs[index]
        else:
            index = np.random.randint(low=0, high=len(self.neg_indcs), size=1)[0]
            index = self.neg_indcs[index]

        # prepare pc
        cat = self.categories[index]
        obj_idx = int(self.metadata[index])
        if self.full_pc:
            pc = self.pcs[cat][obj_idx][0:30]
            pc = np.concatenate(pc)
        else:
            view_size = np.random.randint(low=1, high=4, size=1)[0]
            pc_indcs = np.random.randint(low=0, high=999, size=view_size)
            
            if len(pc_indcs) == 1:
                pc = self.pcs[cat][obj_idx][pc_indcs[0]]
            else:
                __pcs = []
                for pc_index in pc_indcs:
                    __pcs = __pcs + [self.pcs[cat][obj_idx][pc_index]]
                pc = np.concatenate( __pcs )

        pc = regularize_pc_point_count(pc, 1024, False)

        # jitter point cloud here
        pc = add_jitter(pc)

        pc = torch.FloatTensor( pc )

        _quat = self.quaternions[index]
        _trans = self.translations[index]
        # augmentation
        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               cat, \
               obj_idx

    def __len__(self):
        return self.size


class GraspDatasetWithTightSoft(Dataset):
    def __init__(self, path_to_grasps = 'data/grasps/preprocessed',
                       path_to_grasps_tight = 'data/grasps_tight/preprocessed',
                       path_to_pc='data/pcs',
                       split='train', 
                       allowed_categories = ['mug', 'box', 'bowl', 'bottle', 'cylinder', 'spatula', 'hammer', 'pan', 'fork', 'scissor'], 
                       augment=True, 
                       full_pc=False,
                       tight=True):

        # load grasp_data
        self.quaternions, self.translations, self.labels, self.metadata, self.categories = load_data(path_to_grasps, split, allowed_categories=allowed_categories)

        self.pos_indcs = np.where(self.labels.numpy() == 1)[0]
        self.neg_indcs = np.where(self.labels.numpy() == 0)[0]
        self.tight_start = len(self.labels)

        if tight:
            _quaternions, _translations, _labels, _metadata, _categories = load_data(path_to_grasps_tight, split, allowed_categories=allowed_categories)
            self.quaternions = torch.cat([self.quaternions, _quaternions], 0)
            self.translations = torch.cat([self.translations, _translations], 0)
            self.labels = torch.cat([self.labels, _labels], 0)
            self.metadata = np.concatenate([self.metadata, _metadata], 0)
            self.categories = self.categories + _categories

            total_tight_size = len(_labels)

            del _quaternions, _translations, _labels, _metadata, _categories

        self.size = int(total_tight_size*100/70) # 70 to 30 POS to NEG ratio
        self.tight_end = self.tight_start + total_tight_size
        # print(f'Total: {self.total_count}')
        # print(f'Positives: {self.pos_count}, negatives: {self.size - self.pos_count}')
        # print(f'Negatives will be sampled from: {len(self.neg_indcs)}')

        # load pcs
        self.pcs = load_pcs(path_to_pc, allowed_categories)

        # read uniform quaternions
        self.uniform_quaternions = genfromtxt('configs/data3_36864.qua', delimiter='\t')
        self.augment = augment
        self.full_pc = full_pc

    def __getitem__(self, index):
        # subsample 70 to 30
        index = index % 100
        if index < 30:
            index = self.neg_indcs[index]
        else:
            index = np.random.randint(low=self.tight_start, high=self.tight_end, size=1)[0]

        # prepare pc
        cat = self.categories[index]
        obj_idx = int(self.metadata[index])
        if self.full_pc:
            pc = self.pcs[cat][obj_idx][0:30]
            pc = np.concatenate(pc)
        else:
            view_size = np.random.randint(low=1, high=4, size=1)[0]
            
            pc_indcs = np.random.randint(low=0, high=999, size=view_size)
            
            if len(pc_indcs) == 1:
                pc = self.pcs[cat][obj_idx][pc_indcs[0]]
            else:
                __pcs = []
                for pc_index in pc_indcs:
                    __pcs = __pcs + [self.pcs[cat][obj_idx][pc_index]]
                pc = np.concatenate( __pcs )

        pc = regularize_pc_point_count(pc, 1024, False)
        pc = add_jitter(pc)

        pc = torch.FloatTensor( pc )

        _quat = self.quaternions[index]
        _trans = self.translations[index]
        # augmentation
        if self.augment:
            pc, _quat, _trans = augment_grasp(pc, _quat, _trans, self.uniform_quaternions)

        return _quat,\
               _trans,\
               pc,\
               self.labels[index],\
               cat

    def __len__(self):
        return self.size

def add_jitter(pc, sigma=0.0015, clip=0.007):
    # jitter point cloud here
    # https://github.com/charlesq34/pointnet/blob/master/provider.py#L74
    pc = pc + np.clip(sigma * np.random.randn(*pc.shape), -1 * clip, clip).astype(np.float32)
    return pc

def load_data(path_to_grasps, split, allowed_categories):
    quaternions = []
    translations = []
    labels = []
    metadata = []
    categories = []
    for cat in allowed_categories:
        quaternions.append( np.load(f'{path_to_grasps}/{cat}/quaternions_{split}.npy') )
        translations.append( np.load(f'{path_to_grasps}/{cat}/translations_{split}.npy') )
        labels.append( np.load(f'{path_to_grasps}/{cat}/isaac_labels_{split}.npy') )
        metadata.append( np.load(f'{path_to_grasps}/{cat}/metadata_{split}.npy') )
        categories = categories + [cat]*len(metadata[-1])

    quaternions = torch.FloatTensor(np.concatenate(quaternions))
    translations = torch.FloatTensor(np.concatenate(translations))
    labels = torch.FloatTensor(np.concatenate(labels))
    metadata = np.concatenate(metadata).astype(np.int)
    categories = categories

    return quaternions, translations, labels, metadata, categories

def load_pcs(path_to_pc, allowed_categories):
    pcs = {}
    for cat in allowed_categories:
        pcs[cat] = [] 
        for k in range(21):
            try:
                data = pickle.load(open(f'{path_to_pc}/{cat}/{cat}{k:03}.pkl', 'rb'))
                pcs[cat].append(data['pcs'])
            except:
                pcs[cat].append([])
    return pcs

def augment_grasp(pc, quaternion, translation, uniform_quaternions):
    '''
    pc: [n, 3]
    quaternion: [4]
    translation: [3]
    '''
    # sample random unit quaternion
    # rand_quat = torch.FloatTensor([[0.707,0.707,0,0.0]])
    rand_ind = np.random.randint(low=0, high=len(uniform_quaternions), size=1)[0]
    rand_quat = uniform_quaternions[rand_ind]
    rand_quat = torch.FloatTensor(rand_quat).unsqueeze(0)

    # rotate pc
    pc = pc.unsqueeze(0)
    rand_quat1 = rand_quat.unsqueeze(1).repeat([1,pc.shape[1], 1])
    #print(rand_quat1.shape)
    pc = quat_ops.rot_p_by_quaterion(pc, rand_quat1).squeeze()

    # rotate translation
    translation = translation.unsqueeze(0).unsqueeze(0)
    rand_quat2 = rand_quat.unsqueeze(1).repeat([1,1, 1])
    translation = quat_ops.rot_p_by_quaterion(translation, rand_quat2).squeeze()

    # rotate quaternion
    quaternion = quaternion.unsqueeze(0)
    quaternion = quat_ops.quaternion_mult(rand_quat, quaternion).squeeze()
    return pc, quaternion, translation
