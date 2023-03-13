#!/usr/bin/env python3
# Objective: preprocess for box014 only
# Tasbolat Taunyazov

import numpy as np
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path
from openpyxl import load_workbook
import pandas as pd

grasp_data_dir = 'data/semantic_data' #'data/grasps_new'
save_dir = 'data/semantic_data/preprocessed2'

TRANS_DISTANCE_MAX = 0.30 # in meters -> use this to filter out far grasps

test_info = {
    'mug': [2, 8, 14],
    'fork': [1, 11],
    'hammer': [2, 15],
}


# def get_filenames(cat='', idx=0):
#     return glob.glob(f"{grasp_data_dir}/{cat}/{cat}{idx:03}_isaac*/*.npz")


        

# def read_data(cat, i, collect_info=False):

#     # list all the files
#     data = {}
#     data['quaternions'] = []
#     data['translations'] = []
#     data['isaac_labels'] = []
#     data['metadata']  = []

#     fnames = get_filenames(cat=cat, idx=i)
#     if len(fnames) == 0:
#         # raise ValueError('No files in the file')
#         return None, None
    
#     for fname in fnames:
#         sample_data = np.load(fname)

#         grasp_trans_distance = np.linalg.norm(sample_data['translations'], axis=1)
#         chosen_mask = (grasp_trans_distance <= TRANS_DISTANCE_MAX)

#         if not('isaac_labels' in sample_data):
#             continue

#         data['isaac_labels'].append( sample_data['isaac_labels'][chosen_mask] )
#         data['quaternions'].append( sample_data['quaternions'][chosen_mask] )
#         data['translations'].append( sample_data['translations'][chosen_mask] )

#     data['quaternions'] = np.concatenate(data['quaternions'])
#     data['translations'] = np.concatenate(data['translations'])
#     data['isaac_labels'] = np.concatenate(data['isaac_labels'])
#     data['metadata'] = np.ones(len(data['translations']))*i

#     pos_count = np.sum(data['isaac_labels'])
#     neg_count = len(data['isaac_labels']) - pos_count
        
#     print(f'{cat}{i:03}: {pos_count} {neg_count}')
#     print(f'{cat}{i:03}: {pos_count} {neg_count}')

#     if pos_count == 0:
#         return None, None

#     if collect_info:
#         info = {
#         'cat': cat,
#         'idx': [],
#         'Pos': [],
#         'Neg': [],
#         'Total': [],
#         'Pos_share_percent': [],
#         'Ratio':[]
#         }
        
#         info['idx'].append(i)
#         info['Pos'].append(pos_count)
#         info['Neg'].append(neg_count)
#         total = pos_count + neg_count
#         info['Total'].append(total)
#         _percentage = pos_count/total*100 if total!=0 else np.inf
#         info['Pos_share_percent'].append(_percentage)       
#         _ratio = neg_count/pos_count if pos_count !=0 else np.inf   
#         info['Ratio'].append(_ratio)
#         df_info = pd.DataFrame(info)
#         return data, df_info    

#     return data, None

# def preprocess(cat, is_train=True):
#     quaternions = []
#     translations = []
#     isaac_labels = []
#     metadata = []
#     for i in range(0,21,1):

#         if i in test_info[cat]:
#             if is_train:
#                 continue
#         else:
#             if not is_train:
#                 continue

#         data, _ = read_data(cat=cat, i=i)

#         if data is None:
#             continue


#         quaternions.append( data['quaternions'] )
#         translations.append( data['translations'] )
#         isaac_labels.append( data['isaac_labels'] )
#         metadata.append( data['metadata'] )
#         del data

#     quaternions = np.concatenate(quaternions)
#     translations = np.concatenate(translations)
#     isaac_labels = np.concatenate(isaac_labels)
#     metadata = np.concatenate(metadata)

#     # reorder from pos to neg
#     print(f'Final count for {cat} is {len(isaac_labels)}:')
#     print(f'Quaternions shape is {quaternions.shape}')
#     print(f'Translations shape is {translations.shape}')
#     print(f'isaac_labels shape is {isaac_labels.shape}')
#     print()
#     Path(f'{save_dir}/{cat}').mkdir(parents=True, exist_ok=True)
#     if is_train:
#         np.save(f'{save_dir}/{cat}/quaternions_train', quaternions)
#         np.save(f'{save_dir}/{cat}/translations_train', translations)
#         np.save(f'{save_dir}/{cat}/isaac_labels_train', isaac_labels)
#         np.save(f'{save_dir}/{cat}/metadata_train', metadata)
#         print('Done with train set preprocessing')
#     else:
#         np.save(f'{save_dir}/{cat}/quaternions_test', quaternions)
#         np.save(f'{save_dir}/{cat}/translations_test', translations)
#         np.save(f'{save_dir}/{cat}/isaac_labels_test', isaac_labels)
#         np.save(f'{save_dir}/{cat}/metadata_test', metadata)
#         print('Done with test set preprocessing')

#     del quaternions, translations, isaac_labels

def read_data(cat, idx, sample_size=2500):
    f = f'{grasp_data_dir}/{cat}/{cat}{idx:003}/all_stable_grasps_labelled.npz'
    data = np.load(f)
    N = len(data['translations'])
    metadata = np.ones(N)*idx

    pos_index = np.where(data['handover'] == 1)[0]
    neg_index = np.where(data['handover'] == 0)[0]
    
    if (not (len(pos_index) == 0)) and (not (len(neg_index) == 0)):
        chosen_pos_idx = np.random.choice(pos_index, size=sample_size)
        chosen_neg_idx = np.random.choice(neg_index, size=sample_size)
        chosen_idx = np.concatenate([chosen_pos_idx, chosen_neg_idx])
    elif len(pos_index) == 0:
        chosen_neg_idx = np.random.choice(neg_index, size=sample_size)
        chosen_idx = chosen_neg_idx
    elif len(neg_index) == 0:
        chosen_pos_idx = np.random.choice(pos_index, size=sample_size)
        chosen_idx = chosen_pos_idx

    return data['quaternions'][chosen_idx], data['translations'][chosen_idx], data['handover'][chosen_idx], metadata[chosen_idx]

def preprocess(cat, is_train=True):
    quaternions = []
    translations = []
    labels = []
    metadata = []
    for i in range(0,20,1):

        if i in test_info[cat]:
            if is_train:
                continue
        else:
            if not is_train:
                continue

        q, t, l, m = read_data(cat=cat, idx=i)

        quaternions.append(q)
        translations.append(t)
        labels.append(l)
        metadata.append(m)

    quaternions = np.concatenate(quaternions)
    translations = np.concatenate(translations)
    labels = np.concatenate(labels)
    metadata = np.concatenate(metadata)

    # reorder from pos to neg
    print(f'Final count for {cat} is {len(labels)}:')
    print(f'Quaternions shape is {quaternions.shape}')
    print(f'Translations shape is {translations.shape}')
    print(f'labels shape is {labels.shape}')
    Path(f'{save_dir}/{cat}').mkdir(parents=True, exist_ok=True)
    if is_train:
        np.save(f'{save_dir}/{cat}/quaternions_train', quaternions)
        np.save(f'{save_dir}/{cat}/translations_train', translations)
        np.save(f'{save_dir}/{cat}/labels_train', labels)
        np.save(f'{save_dir}/{cat}/metadata_train', metadata)
        print('Done with train set preprocessing')
    else:
        np.save(f'{save_dir}/{cat}/quaternions_test', quaternions)
        np.save(f'{save_dir}/{cat}/translations_test', translations)
        np.save(f'{save_dir}/{cat}/labels_test', labels)
        np.save(f'{save_dir}/{cat}/metadata_test', metadata)
        print('Done with test set preprocessing')


if __name__ == '__main__':
    cats = ['mug', 'hammer', 'fork']


    ######### FOR DATA PREPROCESS #################
    for cat in cats:
        preprocess(cat, is_train=False)
    for cat in cats:
        preprocess(cat, is_train=True)

    ##### FOR DATA COUNTING ##############
    # for cat in cats:
    #     count_data(cat)