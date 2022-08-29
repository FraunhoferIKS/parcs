import numpy as np
from scipy.special import expit


def edge_empty(**kwargs):
    assert kwargs
    print('edge function not implemented')
    raise ValueError


def edge_sigmoid(array=None,
                 alpha=1.0, beta=0.0, gamma=1, tau=1):
    expon = (-1)**gamma * 2 * alpha * ((array - beta)**tau)

    return expit(expon)


def edge_gaussian_rbf(array=None,
                      alpha=1, beta=0, gamma=0, tau=2):
    expon = -alpha * ((array - beta)**tau)
    return gamma + ((-1)**gamma) * np.exp(expon)


def edge_binary_identity(array=None):
    return array


if __name__ == '__main__':
    arr = np.array(range(100))

    # v = edge_sigmoid_new(array=arr)
    v = edge_gaussian_rbf(array=arr)

    from matplotlib import pyplot as plt
    plt.scatter(arr, v)
    plt.show()
