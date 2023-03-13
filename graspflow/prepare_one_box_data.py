#!/usr/bin/env python3
# Objective: preprocess for box014 only
# Tasbolat Taunyazov

import numpy as np
from pathlib import Path
import glob
from sklearn.model_selection import train_test_split
from pathlib import Path

grasp_data_dir = 'data/grasps'
save_dir = 'data/grasps/preprocessed_one_box'

def get_filenames(cat='', idx=0):
    return glob.glob(f"{grasp_data_dir}/{cat}/{cat}{idx:03}_isaac/*.npz")

def read_data(cat, i):

    # list all the files
    data = {}
    data['quaternions'] = []
    data['translations'] = []
    data['isaac_labels'] = []

    fnames = get_filenames(cat=cat, idx=i)
    if len(fnames) == 0:
        raise ValueError('No files in the file')

    for fname in fnames:
        sample_data = np.load(fname)

        if not('isaac_labels' in sample_data):
            data['isaac_labels'].append(np.zeros(len(sample_data['quaternions'])))
        else:
            data['isaac_labels'].append(sample_data['isaac_labels'])

        data['quaternions'].append(sample_data['quaternions'])
        data['translations'].append(sample_data['translations'])

    data['quaternions'] = np.concatenate(data['quaternions'])
    data['translations'] = np.concatenate(data['translations'])
    data['isaac_labels'] = np.concatenate(data['isaac_labels'])

    pos_count = np.sum(data['isaac_labels'])
    neg_count = len(data['isaac_labels']) - pos_count
        
    print(f'{cat}{i:03}: {pos_count} {neg_count}')

    return data

if __name__ == '__main__':
    cat = 'box'
    idx = 14
    
    # prepare data
    data = read_data(cat=cat, i=idx)

    # get positive indices
    pos_ind = np.where(data['isaac_labels'] == 1)[0]

    train_ind, test_ind = train_test_split(np.arange(len(data['isaac_labels'])), stratify=data['isaac_labels'])

    # save
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    # train
    np.save(f'{save_dir}/quaternions_train', data['quaternions'][train_ind])
    np.save(f'{save_dir}/translations_train', data['translations'][train_ind])
    np.save(f'{save_dir}/isaac_labels_train', data['isaac_labels'][train_ind])

    np.save(f'{save_dir}/quaternions_test', data['quaternions'][test_ind])
    np.save(f'{save_dir}/translations_test', data['translations'][test_ind])
    np.save(f'{save_dir}/isaac_labels_test', data['isaac_labels'][test_ind])

    print(f"Train size: {len(data['isaac_labels'][train_ind])}")
    print(f"Test size: {len(data['isaac_labels'][test_ind])}")
    print('Done')

    del data, train_ind, test_ind