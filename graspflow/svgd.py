'''
SVGD:
Original implementation:

'''

import torch
import torch.autograd as autograd


class SVGD:
  def __init__(self, P, K, optimizer):
    self.P = P
    self.K = K
    self.optim = optimizer

  def phi(self, pc, trans, eulers):
    # TODO:
    '''
    It seems that gpu issue with realizing autograd on log_prob?
    '''
    trans = trans.detach().requires_grad_(True)
    eulers = eulers.detach().requires_grad_(True)

    log_prob = self.P.log_prob_with_eulers(pc=pc, trans=trans, eulers=eulers)

    # TODO: outdated -> replace with torch parameter!
    score_func_trans = autograd.grad(log_prob.sum(), trans)[0]
    score_func_eulers = autograd.grad(log_prob.sum(), eulers)[0]

    K_XX = self.K(X, X.detach())
    grad_K = -autograd.grad(K_XX.sum(), X)[0]

    phi = (K_XX.detach().matmul(score_func) + grad_K) / X.size(0)

    return phi

  def step(self, X):
    self.optim.zero_grad()
    X.grad = -self.phi(X)

    # TODO: balance optimizer over here?
    self.optim.step()




class RBF(torch.nn.Module):
  def __init__(self, sigma=None):
    super(RBF, self).__init__()

    self.sigma = sigma

  def forward(self, X, Y):
    XX = X.matmul(X.t())
    XY = X.matmul(Y.t())
    YY = Y.matmul(Y.t())

    dnorm2 = -2 * XY + XX.diag().unsqueeze(1) + YY.diag().unsqueeze(0)

    # Apply the median heuristic (PyTorch does not give true median)
    if self.sigma is None:
      np_dnorm2 = dnorm2.detach().cpu().numpy()
      h = np.median(np_dnorm2) / (2 * np.log(X.size(0) + 1))
      sigma = np.sqrt(h).item()
    else:
      sigma = self.sigma

    gamma = 1.0 / (1e-8 + 2 * sigma ** 2)
    K_XY = (-gamma * dnorm2).exp()

    return K_XY
  
     

