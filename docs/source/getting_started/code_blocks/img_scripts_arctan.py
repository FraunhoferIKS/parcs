from pyparcs.api.mapping_functions import *
from matplotlib import pyplot as plt
import matplotlib
import numpy as np
import cycler

n = 5
color = plt.cm.viridis(np.linspace(0, 1,n))
matplotlib.rcParams['axes.prop_cycle'] = cycler.cycler('color', color)

fig, axs = plt.subplots(2, 2, sharey=True, sharex=True)

for alpha in [1, 2, 3, 4]:
    x = np.linspace(-2, 2, 100)
    axs[0, 0].plot(x, edge_arctan(x, alpha=alpha), label=str(alpha))
    axs[0, 0].set_title('Alpha')
    axs[0, 0].legend()
for beta in [-2, -1, 0, 1, 2]:
    x = np.linspace(-2, 2, 100)
    axs[0, 1].plot(x, edge_arctan(x, beta=beta), label=str(beta))
    axs[0, 1].set_title('Beta')
    axs[0, 1].legend()
for gamma in [0, 1]:
    x = np.linspace(-2, 2, 100)
    axs[1, 0].plot(x, edge_arctan(x, gamma=gamma), label=str(gamma))
    axs[1, 0].set_title('Gamma')
    axs[1, 0].legend()

fig.delaxes(axs[1][1])
fig.suptitle('edge Arctan parameter variations')
plt.savefig('../img/edge_arctan.png')
