import numpy as np
from scipy.special import expit

def edge_empty(**kwargs):
    assert kwargs
    print('edge function not implemented')
    raise ValueError


def edge_sigmoid(array=None,
                 alpha=1.0, beta=0.0, gamma=1, tau=1):
    r"""
    here is the formula

    .. math::
        z^*_i = \sigma\Big(
            \alpha
        \Big)

    Parameters
    ----------
    array
    alpha
    beta
    gamma
    tau

    Returns
    -------

    """
    expon = (-1)**gamma * 2 * alpha * ((array - beta)**tau)

    return expit(expon)


def edge_gaussian_rbf(array=None,
                      alpha=1, beta=0, gamma=0, tau=2):
    r"""
    .. math::
        \begin{align}
            z^*_i = \gamma + (-1)^\gamma .
            \exp\big(-\alpha \|z_i - \beta\|^\tau\big),
        \end{align}

    Parameters
    ----------
    array
    alpha
    beta
    gamma
    tau

    Returns
    -------

    """
    expon = -alpha * ((array - beta)**tau)
    return gamma + ((-1)**gamma) * np.exp(expon)


def edge_identity(array=None):
    r"""

    .. math::
        z^*_i = z_i

    Parameters
    ----------
    array

    Returns
    -------

    """
    return array


EDGE_FUNCTIONS = {
    'identity': edge_identity,
    'sigmoid': edge_sigmoid,
    'gaussian_rbf': edge_gaussian_rbf
}

FUNCTION_PARAMS = {
    'identity': [],
    'sigmoid': ['alpha', 'beta', 'gamma', 'tau'],
    'gaussian_rbf': ['alpha', 'beta', 'gamma', 'tau']
}


if __name__ == '__main__':
    arr = np.array(range(100))

    # v = edge_sigmoid_new(array=arr)
    v = edge_gaussian_rbf(array=arr)

    from matplotlib import pyplot as plt
    plt.scatter(arr, v)
    plt.show()
