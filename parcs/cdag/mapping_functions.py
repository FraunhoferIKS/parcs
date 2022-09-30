import numpy as np
from scipy.special import expit

def edge_empty(**kwargs):
    assert kwargs
    print('edge function not implemented')
    raise ValueError


def edge_sigmoid(array=None,
                 alpha=1.0, beta=0.0, gamma=0, tau=1):
    r"""
    This edge function transforms input variable according to the following equation:

    .. math::
        x^{*} = \sigma\Big(
            (-1)^{\gamma+1} \alpha (x - \beta)^{\tau}
        \Big), \quad \sigma(a) = \frac{1}{1+e^{-a}}

    .. warning::
        put images

    Parameters
    ----------
    array : array-like
        input array
    alpha : float, default=1.0
        scale parameter. Reasonable range in `[0.5, 3]`
    beta : float, default=0.0
        offset parameter. Reasonable range in `[-0.8, 0.8]`
    gamma : {0, 1}, default=0
        mirroring parameter.
    tau : odd integer, default=1
        power parameter

    Returns
    -------
    transformed_array : array-like

    """
    expon = (-1)**gamma * alpha * ((array - beta)**tau)

    return expit(expon)


def edge_gaussian_rbf(array=None,
                      alpha=1, beta=0, gamma=0, tau=2):
    r"""
    .. math::
        \begin{align}
            z^*_i = \gamma + (-1)^\gamma .
            \exp\big(-\alpha \|z_i - \beta\|^\tau\big),
        \end{align}

    .. warning::
        put images

    Parameters
    ----------
    array : array-like
        input array
    alpha : float, default=1.0
        scale parameter. Reasonable range in `[0.5, 3]`
    beta : float, default=0.0
        offset parameter. Reasonable range in `[-0.8, 0.8]`
    gamma : {0, 1}, default=0
        mirroring parameter.
    tau : even integer, default=2
        power parameter

    Returns
    -------
    transformed_array : array-like

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
    arr = np.random.normal(0, 1, size=300)

    # v = edge_sigmoid_new(array=arr)
    v = edge_gaussian_rbf(array=arr, alpha=1, beta=-1, gamma=1)

    from matplotlib import pyplot as plt
    plt.scatter(arr, v)
    plt.show()
