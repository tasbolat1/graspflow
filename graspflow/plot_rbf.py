import torch
import gpytorch
import numpy as np
import matplotlib.pyplot as plt

lengthscale = 1.0
K_rots = gpytorch.kernels.RBFKernel()
K_rots.lengthscale = torch.FloatTensor([[lengthscale]])

K_trans = gpytorch.kernels.RBFKernel()
K_trans.lengthscale = torch.FloatTensor([[np.sqrt(0.02)]])


def plot_rbf(ax, x, lengthscale):

    # define RBF kernel
    K = gpytorch.kernels.RBFKernel()
    K.lengthscale = torch.FloatTensor([[lengthscale]])

    # compute distance and kernel
    x_norm = np.linalg.norm(x[0] - x, axis=1)
    K_x = K(torch.FloatTensor(x))[0].detach().numpy()

    # plot them
    ax.plot(x_norm, K_x, label=f'l={lengthscale}')
    ax.set_ylim([0,1.05])
    ax.set_xlim([0,np.max(x_norm)])

N = 1000
start=0.0
stop=0.3
X=np.linspace(start=start, stop=stop, num=N)
Y=np.linspace(start=start, stop=stop, num=N)
Z=np.linspace(start=start, stop=stop, num=N)
trans = np.stack([X,Y,Z]).T


start=-np.pi/2
stop=np.pi/2
X=np.linspace(start=start, stop=stop, num=N)
Y=np.linspace(start=start, stop=stop, num=N)
Z=np.linspace(start=start, stop=stop, num=N)
rots = np.stack([X,Y,Z]).T

trans_ls = [0.0001, 0.001, 0.01, 0.025, 0.05, 0.1, 1.0]
rots_ls = [0.0001, 0.001, 0.01, 0.1, 1.0]

fig, ax = plt.subplots(nrows=2)
for lengthscale in trans_ls:
    plot_rbf(ax[0], trans, lengthscale)
ax[0].legend()
ax[0].set_xlabel('distance (2-norm)')
ax[0].set_ylabel("K(x,x')")
ax[0].set_title('Translation')

for lengthscale in rots_ls:
    plot_rbf(ax[1], rots, lengthscale)
ax[1].legend()
ax[1].set_xlabel('distance (2-norm)')
ax[1].set_title('Rotations')
ax[1].set_ylabel("K(x,x')")

plt.show()