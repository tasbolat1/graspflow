import numpy as np
from pathlib import Path
from utils.auxilary import InfoHolder, info_holder_stack
import copy
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_info_holders_combined(data_dir, prefix, cat, idx):
    args_and_info_file_dirs = f'{data_dir}/refinement_infoholders/{cat}'
    paths_to_args_and_files = list(Path(args_and_info_file_dirs).glob(f"{cat}{idx:003}_{prefix}_*_info.npz"))

    num_uniq_pcs = len(paths_to_args_and_files)

    # load info holders
    all_info = []
    
    for i in range(num_uniq_pcs):
        info = InfoHolder()
        info.load(f'{args_and_info_file_dirs}/{cat}{idx:003}_{prefix}_{i}_info.npz')
        all_info.append(info)

    print(all_info)

    info = info_holder_stack(all_info)


    with open(f'{args_and_info_file_dirs}/{cat}{idx:003}_{prefix}_args.pkl', 'rb') as fp:
        r_args = pickle.load(fp)
    return info, r_args
    
    # parse info holder combine them

def load_args(data_dir, prefix, cat, idx):
    args_and_info_file_dirs = f'{data_dir}/refinement_infoholders/{cat}'
    with open(f'{args_and_info_file_dirs}/{cat}{idx:003}_{prefix}_args.pkl', 'rb') as fp:
        r_args = pickle.load(fp)
    return r_args


def str2array(x, delimter=' ', dtype=float):
    '''
    converts string of [23 21 0 12] to numpy array
    '''
    xs = x.strip()[1:-1].split(delimter)
    new_x = []
    for _x in xs:
        new_x.append(dtype(_x))
    return np.array(new_x)

def load_isaac_results(experiment, prefix, full_exeriment=True, experiments_dir = '../experiments'):
    '''
    Loads isaac data regardless of what experiment (full or gripper only).
    '''
    
    cats = []
    indcs = []
    env_nums = []
    labels = []
    mi_scores = []

    if full_exeriment:
        fname = f'{experiments_dir}/generated_grasps_experiment{experiment}/Full Experiment {experiment} log {prefix}.txt'
        
    else:
        fname = f'{experiments_dir}/generated_grasps_experiment{experiment}/Experiment {experiment} log {prefix}.txt'

        
    fp = open(fname)
    all_data_txt = fp.read().split('\n')
    fp.close()
    

    for line in all_data_txt:
        
        if len(line) < 4:
            continue
        
        if full_exeriment:
            cat, idx, env_num, label, mi_score = line.split('=')
        else:
            cat, idx, env_num, label = line.split('=')
        cats.append(cat)
        indcs.append(int(idx))
        env_nums.append(int(env_num))
        labels.append(str2array(label, dtype=int))
        if full_exeriment:
            mi_scores.append(str2array(mi_score[:-1], dtype=float))
        else:
            mi_scores.append(np.zeros_like(labels[-1]))
                
    return cats, indcs, env_nums, labels, mi_scores




def construct_experiment_df(experiment, sampler, prefix, experiments_dir, full_exeriment=True):
    
    prefix_sampler = f'{sampler}_N_N_N'

    # load isaac data
    cats, indcs, env_nums, labels, mi_scores = load_isaac_results(experiment, prefix=prefix, full_exeriment=full_exeriment)
    _, _, _, sampler_labels, sampler_mi_scores = load_isaac_results(experiment, prefix=prefix_sampler, full_exeriment=full_exeriment)

    # load npz data and sort
    init_scores = []
    final_scores = []
    exec_times = []
    init_robot_scores = []
    final_robot_scores = []

    original_scores = [] 
    sample_times = []

    eta_t = []
    eta_e = []
    noise_t = []
    noise_e = []

    eta_theta_s = []
    eta_theta_e = []
    noise_theta = []

    grasp_number = []
    
    for i in range(len(cats)):
        data = np.load(f'{experiments_dir}/generated_grasps_experiment{experiment}/{cats[i]}{indcs[i]:003}_{sampler}.npz')
        init_score = data[f'{prefix}_init_scores'][env_nums[i]]
        sorted_indcs = np.argsort(init_score)
        labels[i] = labels[i][sorted_indcs]
        mi_scores[i] = mi_scores[i][sorted_indcs]
        init_scores.append(init_score[sorted_indcs])
        # print(i,data[f'{prefix}_time'][env_nums[i]])
        exec_times.append(data[f'{prefix}_time'][env_nums[i]])
        final_scores.append( data[f'{prefix}_scores'][env_nums[i]][sorted_indcs])
        init_robot_scores.append( data[f'{prefix}_init_robot_scores'][env_nums[i]][sorted_indcs] )
        final_robot_scores.append( data[f'{prefix}_robot_scores'][env_nums[i]][sorted_indcs] )
        sample_times.append(data[f'{prefix_sampler}_time'][env_nums[i]])
        original_scores.append( data[f'{prefix_sampler}_original_scores'][env_nums[i]][sorted_indcs])
        sampler_labels[i] = sampler_labels[i][sorted_indcs]
        sampler_mi_scores[i] = sampler_mi_scores[i][sorted_indcs]

            # load args
        args = load_args(data_dir=f'{experiments_dir}/generated_grasps_experiment{experiment}', cat=cats[i], idx=indcs[i], prefix=prefix)
        eta_t.append(args.eta_t)
        eta_e.append(args.eta_e)
        noise_t.append(args.noise_t)
        noise_e.append(args.noise_e)

        eta_theta_s.append(args.eta_theta_s)
        eta_theta_e.append(args.eta_theta_e)
        noise_theta.append(args.noise_theta)
        grasp_number.append(np.arange(1,len(labels[0])+1,1))
        

    # create big dataframe to work with ease
    num_grasps = len(labels[0])
    num_envs = len(np.unique(env_nums))
    
    if sampler == 'gpd':
        sample_times = np.repeat(sample_times, num_grasps)
    elif sampler == 'graspnet':
        sample_times = np.array(sample_times).flatten()

    df_data = {
        'cat': np.repeat(cats, num_grasps),
        'idx': np.repeat(indcs, num_grasps),
        # 'labels': np.repeat(env_nums, env_nums),
        'env_num': np.repeat(env_nums, num_grasps),
        'final_label': np.concatenate(labels),
        'mi_score': np.concatenate(mi_scores),
        'init_scores': np.concatenate(init_scores),
        'final_scores': np.concatenate(final_scores),
        'init_robot_scores': np.concatenate(init_robot_scores),
        'final_robot_scores': np.concatenate(final_robot_scores),
        'exec_times': np.repeat(exec_times, num_grasps),
        'original_scores': np.concatenate(original_scores),
        'sample_times': sample_times,
        'init_label': np.concatenate(sampler_labels),
        'sampler_mi_scores': np.concatenate(sampler_mi_scores),
        'eta_t': np.repeat(eta_t, num_grasps),
        'eta_e': np.repeat(eta_e, num_grasps),
        'noise_t': np.repeat(noise_t, num_grasps),
        'noise_e': np.repeat(noise_e, num_grasps),
        'eta_theta_s': np.repeat(eta_theta_s, num_grasps),
        'eta_theta_e': np.repeat(eta_theta_e, num_grasps),
        'noise_theta': np.repeat(noise_theta, num_grasps),
        'grasp_id': np.concatenate(grasp_number)
    }
    
    print(df_data['exec_times'].shape, df_data['sample_times'].shape)
    
    df = pd.DataFrame(df_data)

    df['final_success'] = df.apply(lambda row: 1 if row.final_label==1 else 0, axis=1)
    df['init_success'] = df.apply(lambda row: 1 if row.init_label==1 else 0, axis=1)    

    return df, num_grasps, num_envs
    



def summary(prefix, df, save_dir, full_exeriment=True):
    num_envs = len(df.env_num.unique())
    num_grasps = len(df.grasp_id.unique())
    df_summary = df[['cat', 'idx', 'init_scores', 'final_scores', 'init_success', 'final_success']]
    df_summary = df_summary.groupby(['cat', 'idx']).agg({'init_scores': [np.mean],
                                        'final_scores': [np.mean],
                                        'init_success': [np.sum],
                                        'final_success': [np.sum]}).reset_index()
    df_summary = df_summary.assign(total_grasps = num_envs*num_grasps)

    if full_exeriment:
        writer = pd.ExcelWriter(f'{save_dir}/{prefix}_full_results.xlsx', engine = 'openpyxl')
    else:
        writer = pd.ExcelWriter(f'{save_dir}/{prefix}_results.xlsx', engine = 'openpyxl')
    df.to_excel(writer, sheet_name = 'All')
    df_summary.to_excel(writer, sheet_name = 'Summary')
    writer.close()

    df_summary.columns = [a[0] for a in df_summary.columns.to_flat_index()]

    fig, ax = plt.subplots(ncols=1, nrows=2, figsize=(16,10))
    df_temp = df_summary.melt(id_vars=['cat', 'idx'], value_vars=['init_scores', 'final_scores']).rename(columns=str.title)
    sns.barplot(ax = ax[0], data=df_temp, x="Cat", y="Value", hue="Variable")

    df_temp = df_summary.melt(id_vars=['cat', 'idx'], value_vars=['init_success', 'final_success']).rename(columns=str.title)
    sns.barplot(ax = ax[1], data=df_temp, x="Cat", y="Value", hue="Variable")
    if full_exeriment:
        plt.savefig(f'{save_dir}/{prefix}_full_scores_successes.pdf')
    else:
        plt.savefig(f'{save_dir}/{prefix}_scores_successes.pdf')
    plt.show()


def summary_labels(prefix, df, save_dir, full_exeriment=True):
    num_envs = len(df.env_num.unique())
    num_grasps = len(df.grasp_id.unique())
    df_labels = df[['cat', 'idx','init_label', 'final_label']]

    fig, ax = plt.subplots(nrows=5, ncols=1, figsize=(16,8))

    label_map = {
        1:'success',
        2:'item collision',
        3:'unreachable_or_table_collision',
        4:'failure',
        5:'NaN'
    }

    for i in range(1,6):
        df_temp = df_labels[['cat', 'idx']]
        df_temp = df_temp.assign(init_label = df_labels.apply(lambda row: 1 if row.init_label==i else 0, axis=1))
        df_temp = df_temp.assign(final_label = df_labels.apply(lambda row: 1 if row.final_label==i else 0, axis=1))

        df_temp = df_temp.groupby(['cat']).agg({'init_label': [np.sum], 
                                        'final_label': [np.sum]}).reset_index()
        df_temp.columns = [a[0] for a in df_temp.columns.to_flat_index()]

        df_temp = df_temp.melt(id_vars=['cat'], value_vars=['init_label', 'final_label']).rename(columns={'variable':label_map[i]})

        sns.barplot(ax = ax[i-1], data=df_temp, x="cat", y="value", hue=label_map[i])
        ax[i-1].set_ylim([0,num_envs*num_grasps*3])
        
    
    if full_exeriment:
        plt.savefig(f'{save_dir}/{prefix}_full_labels.pdf')
    else:
        plt.savefig(f'{save_dir}/{prefix}_labels.pdf')


def summary_cumulative_plot(prefix, df, save_dir, num_unique_objects, full_experiment=True):
    num_envs = len(df.env_num.unique())
    num_grasps = len(df.grasp_id.unique())
    df_temp = df[['cat', 'idx', 'grasp_id', 'env_num', 'final_success']]

    dfs_plot = []
    for cat in df.cat.unique():
        all_cumulative_labels = []
        coeff = 0
        df_temp2 = df_temp[df_temp.cat == cat]
        for idx in df_temp2.idx.unique():
            df_temp3 = df_temp2[df_temp2.idx == idx]
            for env_num in df_temp2.env_num.unique():
                df_temp4 = df_temp3[df_temp3.env_num == env_num]
                
                label_indicator = np.zeros(num_grasps+1)
                if df_temp4.final_success[df_temp4.final_success.ne(0).idxmax()] != 0:
                    label_indicator[df_temp4.grasp_id[df_temp4.final_success.ne(0).idxmax()]] = 1
                all_cumulative_labels.append(label_indicator)
                coeff+=1
        all_cumulative_labels = np.stack(all_cumulative_labels)
        all_cumulative_labels = np.cumsum(all_cumulative_labels, axis=1)*coeff
        
        data_plot = {'Grasps iterations':[], 'Success rate':[]}
        for _cumulative_labels in all_cumulative_labels:
            data_plot['Grasps iterations'].append(np.arange(0,11,1))
            data_plot['Success rate'].append(_cumulative_labels)
        
        data_plot['Grasps iterations'] = np.concatenate(data_plot['Grasps iterations'])
        data_plot['Success rate'] = np.concatenate(data_plot['Success rate'])
        df_plot = pd.DataFrame(data_plot)
        dfs_plot.append(df_plot)

    fig, ax = plt.subplots(nrows=len(dfs_plot), ncols=1, figsize=(16, 2*len(dfs_plot)))

    for i, (df_plot, cat) in enumerate(zip(dfs_plot, df.cat.unique())):
        sns.lineplot(ax = ax[i], data=df_plot, x='Grasps iterations', y='Success rate')
        ax[i].text(num_grasps-2, 1, cat, fontsize=18)
        ax[i].set_xlim([0, num_grasps])
        ax[i].set_ylim([0, df_plot['Success rate'].max()+0.5])

    fig.savefig(f'{save_dir}/{prefix}_grasp_successes_by_order_break_down.pdf')
    plt.show()

    df_plot = pd.concat(dfs_plot)
    df_plot.loc[df_plot['Success rate'] > 1, ['Success rate']] = 1
    df_plot['Success rate'] = df_plot['Success rate']*num_unique_objects


    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(16, 6))
    sns.lineplot(ax = ax, data=df_plot, x='Grasps iterations', y='Success rate')
    ax.set_xlim([0, num_grasps])
    ax.set_ylim([0, df_plot['Success rate'].max()+0.5])
    if full_experiment:
        fig.savefig(f'{save_dir}/{prefix}_full_grasp_successes_by_order.pdf')
    else:
        fig.savefig(f'{save_dir}/{prefix}_grasp_successes_by_order.pdf')
    plt.show()