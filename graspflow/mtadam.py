'''
Unofficial implementation of MTAadam optimizer.

Implemented by Tasbolat Taunyazov.

Adopted from:
https://github.com/ItzikMalkiel/MTAdam/blob/master/mtadam.py

Paper citation:
https://arxiv.org/pdf/2006.14683.pdf
'''
 

import math
import torch
from torch.optim.optimizer import Optimizer
import time
import numpy as np

class MTAdam(Optimizer):
    r"""Implements MTAdam algorithm.
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-3)
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-8)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        amsgrad (boolean, optional): whether to use the AMSGrad variant of this
            algorithm from the paper `On the Convergence of Adam and Beyond`_
            (default: False)
        maximize (bool, optional): maximize the params based on the objective, instead of
            minimizing (default: False)
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9), eps=1e-8,
                 weight_decay=0, amsgrad=False, maximize=False):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, maximize=maximize)
        super(MTAdam, self).__init__(params, defaults)


    def __setstate__(self, state):
        super(MTAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)
            group.setdefault('maximize', False)

    @torch.no_grad()
    def step(self, loss_array, ranks, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        states = self.update_weights(loss_array, ranks)

        return loss, states

    def update_weights(self, loss_array, ranks):

      for loss_index, loss in enumerate(loss_array):
        loss.backward(torch.ones_like(loss).to(loss.device), retain_graph=True)
        for group in self.param_groups:
          for p in group['params']:

            if p.grad is None:
              print("breaking")
              break

            p.grad = torch.nan_to_num(p.grad, nan=0) # thesus error: adhoc!

            if p.grad.is_sparse:
              raise RuntimeError('MTAdam does not support sparse gradients')

            amsgrad = group['amsgrad']
            maximize = 1 if not group['maximize'] else -1 
            p.grad.mul_(maximize)

            state = self.state[p]

            # State initialization
            if len(state) == 0:
              state['step'] = 1
              for j, _ in enumerate(loss_array):
                # Exponential moving average of gradient values
                state['exp_avg'+str(j)] = torch.zeros_like(p.data)
                # Exponential moving average of squared gradient values
                state['exp_avg_sq'+str(j)] = torch.zeros_like(p.data)
                if amsgrad:
                  # Maintains max of all exp. moving avg. of sq. grad. values
                  state['max_exp_avg_sq'+str(j)] = torch.zeros_like(p.data)

                if j == 0: p.norms = [torch.ones(1).cuda()]
                else: p.norms.append(torch.ones(1).cuda())

            beta1, beta2, beta3 = group['betas']

            # normalize the norm of current loss gradients to be the same as the anchor
            if state['step'] == 1:
              p.norms[loss_index] = torch.norm(p.grad)
            else:
              p.norms[loss_index] = (p.norms[loss_index]*beta3) + ((1-beta3)*torch.norm(p.grad))
            if p.norms[loss_index] > 1e-10:
              for anchor_index in range(len(loss_array)):
                if p.norms[anchor_index] > 1e-10:
                  p.grad = ranks[loss_index] * p.norms[anchor_index] * p.grad / p.norms[loss_index]
                  break

            exp_avg, exp_avg_sq = state['exp_avg'+str(loss_index)], state['exp_avg_sq'+str(loss_index)]
            if amsgrad:
              max_exp_avg_sq = state['max_exp_avg_sq'+str(loss_index)]

            bias_correction1 = 1 - beta1 ** state['step']
            bias_correction2 = 1 - beta2 ** state['step']
            if loss_index == len(loss_array) - 1:
              state['step'] += 1

            if group['weight_decay'] != 0:
              p.grad = p.grad.add(p, alpha=group['weight_decay'])

            exp_avg.mul_(beta1).add_(p.grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(p.grad, p.grad, value=1 - beta2)
            if amsgrad:
              # Maintains the maximum of all 2nd moment running avg. till now
              torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
              # Use the max. for normalizing running avg. of gradient
              denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
            else:
              denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

            step_size = group['lr'] / bias_correction1

            if loss_index == 0 or not hasattr(p, 'exp_avg'):
              p.exp_avg = [exp_avg]
              p.denom = [denom]
              p.step_size = [step_size]
            else:
              p.exp_avg.append(exp_avg)
              p.denom.append(denom)
              p.step_size.append(step_size)
            if p.grad is not None:
              p.grad.detach_()
              p.grad.zero_()

      states = {}
      for i, group in enumerate(self.param_groups):
        states[i] = {}
        for j, p in enumerate(group['params']):
          states[i][j] = []
          temp = 0
          max_denom = p.denom[0]
          for index in range(1, len(p.exp_avg)):
              max_denom = torch.max(max_denom, p.denom[index])

          for index in range(len(p.exp_avg)):
            delta_change = p.exp_avg[index]/max_denom
            states[i][j].append(delta_change.detach().cpu().numpy())
            update_step = -p.step_size[index]*(delta_change)
            temp += update_step
          p.add_(temp)

          states[i][j] = np.array(states[i][j])

      return states 
      