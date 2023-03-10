import numpy as np
import torch


def quat_norm_diff(q_a, q_b):
    assert(q_a.shape == q_b.shape)
    assert(q_a.shape[-1] == 4)
    if q_a.dim() < 2:
        q_a = q_a.unsqueeze(0)
        q_b = q_b.unsqueeze(0)
    return torch.min((q_a-q_b).norm(dim=1), (q_a+q_b).norm(dim=1)).squeeze()

def quat_chordal_squared_loss(q, q_target, reduce=True):
    assert(q.shape == q_target.shape)
    d = quat_norm_diff(q, q_target)
    losses =  2*d*d*(4. - d*d) 
    loss = losses.mean() if reduce else losses
    return loss    


MSE = torch.nn.MSELoss()

def load_data(fname='temp_data/data_to_test.npz'):
    data = np.load(fname)
    qs = data['q']
    ts = data['t']
    thetas = data['theta']
    torques = data['torque']
    return torch.FloatTensor(qs), torch.FloatTensor(ts), torch.FloatTensor(thetas), torch.FloatTensor(torques)


if __name__ == "__main__":

    # load the data
    qs, ts, thetas, torques = load_data(fname='temp_data/data_to_test_good.npz')

    # check simulation controllers
    sim_theta = []
    sim_trans = []
    sim_quat = []
    total_theta_mse = []
    total_torque_mse = []
    total_trans_mse = []
    total_quats_mse = []
    for i in range(5):
        #print (f'Trial {i+1}:')
        data = np.load(f'temp_data/test_results_new_controller_2/data_to_test_good_{i}.npz')
        print(list(data.keys()))
        _thetas = torch.Tensor(data['thetas']).squeeze(-1)
        _torques = torch.Tensor(data['forces']).squeeze(-1)[:,:,:7]
        _trans = torch.Tensor(data['trans']).squeeze(-1)
        _quats = torch.Tensor(data['quats']).squeeze(-1)



        for j in range(5):
            theta_mse = MSE(_thetas[j], thetas)
            total_theta_mse.append(theta_mse)

            torque_mse = MSE(_torques[j], torques)
            total_torque_mse.append(torque_mse)

            trans_mse = MSE(_trans[j], ts)
            total_trans_mse.append(trans_mse)

            quat_mse = quat_chordal_squared_loss(_quats[j], qs)
            total_quats_mse.append(quat_mse)


    
    total_theta_mse = torch.Tensor(total_theta_mse).reshape(5,5)
    
    total_torque_mse = torch.Tensor(total_torque_mse).reshape(5,5)
    total_trans_mse = torch.Tensor(total_trans_mse).reshape(5,5)
    total_quats_mse = torch.Tensor(total_quats_mse).reshape(5,5)

    print('Theta:')
    print(torch.mean(total_theta_mse, dim=0), torch.std(total_theta_mse, dim=0))
    print('Torque:')
    print(torch.mean(total_torque_mse, dim=0), torch.std(total_torque_mse, dim=0))
    print('Trans:')
    print(torch.mean(total_trans_mse, dim=0), torch.std(total_trans_mse, dim=0))
    print('Quats:')
    print(torch.mean(total_quats_mse, dim=0), torch.std(total_quats_mse, dim=0))
    





