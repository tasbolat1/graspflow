import numpy as np
from pathlib import Path
import pandas as pd

data_dir = '../experiments/generated_grasps_experiment4'

sampler = 'graspnet'
all_npzs = Path(data_dir).glob(f'*_{sampler}.npz')

test_info = {
    'mug': [2, 8, 14],
    'bottle': [3, 12, 19],
    'box':[14, 17],
    'bowl':[1, 16],
    'cylinder': [2, 11],
    'pan': [3,6],
    'scissor': [4,7],
    'fork': [1, 11],
    'hammer': [15],
    'spatula': [1, 14]
}


classifiers=['S', 'Sminus', 'SE', 'N']
grasp_spaces=['N', 'Theta', 'Euler', 'SO3']
methods=['graspnet', 'metropolis', 'GraspFlow', 'N']


all_prefixes_ordered = []
all_prefixes_ordered.append(f'{sampler}_N_N_N')
if sampler == 'graspnet':
    all_prefixes_ordered.append(f'{sampler}_Sminus_graspnet_Euler')
all_prefixes_ordered.append(f'{sampler}_S_graspnet_Euler')
all_prefixes_ordered.append(f'{sampler}_S_metropolis_Euler')
all_prefixes_ordered.append(f'{sampler}_S_GraspFlow_Euler')
all_prefixes_ordered.append(f'{sampler}_S_GraspFlow_SO3')
all_prefixes_ordered.append(f'{sampler}_S_GraspFlow_Theta')
all_prefixes_ordered.append(f'{sampler}_SE_GraspFlow_Theta')



def get_label(_cat, _idx, _prefix, label_requested=1):
    fp = open(f'{data_dir}/Experiment 4 log {_prefix}_ik_res.txt')
    all_data_txt = fp.read().split('\n')
    fp.close()

    res = {}
    for cat in test_info.keys():
        if not (cat in res):
            res[cat] = {}
        for idx in test_info[cat]:
            res[cat][idx] = 0


    for line in all_data_txt:
        if len(line) > 5:
            line_features = line.split('-')
            cat = line_features[0]
            idx = int(line_features[1])
            labels = line_features[3][1:-1]
            for l in labels.split(' '):
                if int(l) == label_requested:
                    res[cat][idx] += 1


    return res[_cat][_idx]







df_data_scores = {}
df_data_robot_scores = {}
df_data_time = {}
df_success_rate = {}

for prefix in all_prefixes_ordered:
    df_data_scores[prefix] = []
    df_data_robot_scores[prefix] = []
    df_data_time[prefix] = []
    df_success_rate[prefix] = []


for cat in test_info.keys():
    for idx in test_info[cat]:
        npz_file = f'{data_dir}/{cat}{idx:003}_{sampler}.npz'
        data = np.load(npz_file)

        df_data_scores[all_prefixes_ordered[0]].append( np.mean(data[f'{all_prefixes_ordered[2]}_init_scores']) )
        df_data_robot_scores[all_prefixes_ordered[0]].append( np.mean(data[f'{all_prefixes_ordered[2]}_init_robot_scores']) )

        df_data_time[all_prefixes_ordered[0]].append( np.mean( data[f'{all_prefixes_ordered[0]}_time'] ) )
        df_success_rate[all_prefixes_ordered[0]].append( get_label(cat, idx, all_prefixes_ordered[0]) )

        for prefix in all_prefixes_ordered[1:]:
            df_data_scores[prefix].append( np.mean(data[f'{prefix}_scores']) )
            if 'Sminus' in prefix:
                df_data_robot_scores[prefix].append(0)
            else:
                df_data_robot_scores[prefix].append( np.mean(data[f'{prefix}_robot_scores']) )

            df_data_time[prefix].append( np.mean( data[f'{prefix}_time'] ) )
            df_success_rate[prefix].append( get_label(cat, idx, prefix) )
            
    
df_scores = pd.DataFrame(df_data_scores)
df_robot_scores = pd.DataFrame(df_data_robot_scores)
df_time = pd.DataFrame(df_data_time)
df_success = pd.DataFrame(df_success_rate)


df_scores.to_csv(f'{data_dir}/{sampler}_scores.csv')
df_robot_scores.to_csv(f'{data_dir}/{sampler}_robot_scores.csv')
df_time.mean().to_csv(f'{data_dir}/{sampler}_time.csv')


df_success.to_csv(f'{data_dir}/{sampler}_success.csv')