import numpy as np
from scipy.special import expit
from parcs.exceptions import DataError, parcs_assert
from typeguard import typechecked

ALPHA_MIN, ALPHA_MAX = 0.1, 30
BETA_MIN, BETA_MAX = -3, 3
GAMMA_RANGE = [0, 1]

def edge_empty(**kwargs):
    assert kwargs
    print('edge function not implemented')
    raise ValueError

@typechecked
def edge_sigmoid(array: np.ndarray = None,
                 alpha: float = 2.0, beta: float = 0.0, gamma: np.int_ = 0, tau: int = 1) -> np.ndarray:
    r"""
    This edge function transforms input variable according to the following equation:

    .. math::
        x^{*} = \sigma\Big(
            (-1)^{\gamma} \alpha (x - \beta)^{\tau}
        \Big), \quad \sigma(a) = \frac{1}{1+e^{-a}}

    Below figures depict the effect of different parameters.

    .. image:: ../first_steps/img/edge_sigmoid.png

    Parameters
    ----------
    array : array-like
        input array
    alpha : float, default=2.0
        scale parameter. Reasonable range in `[1, 3]`
    beta : float, default=0.0
        offset parameter. Reasonable range in `[-2, 2]`
    gamma : {0, 1}, default=0
        mirroring parameter.
    tau : odd integer, default=1
        power parameter

    Returns
    -------
    transformed_array : array-like

    """
    parcs_assert(ALPHA_MIN <= alpha <= ALPHA_MAX, DataError, f'alpha should be a float in [{ALPHA_MIN}, {ALPHA_MAX}], got {alpha} instead')
    parcs_assert(BETA_MIN <= beta <= BETA_MAX, DataError, f'beta should be a float in [{BETA_MIN}, {BETA_MAX}], got {beta} instead')
    parcs_assert(gamma in GAMMA_RANGE, DataError, f'gamma should be an integer in {GAMMA_RANGE}, got {gamma} instead')
    parcs_assert(tau % 2 == 1, DataError, f'tau should be an odd integer, got {tau} instead')

    expon = (-1)**gamma * alpha * ((array - beta)**tau)

    return expit(expon)

@typechecked
def edge_gaussian_rbf(array: np.ndarray = None,
                      alpha: float = 1.0, beta: float = 0, gamma: np.int_ = 0, tau: int = 2) -> np.ndarray:
    r"""
    .. math::
        \begin{align}
            z^*_i = \gamma + (-1)^\gamma .
            \exp\big(-\alpha \|z_i - \beta\|^\tau\big),
        \end{align}

    Below figures depict the effect of different parameters.

    .. image:: ../first_steps/img/edge_gaussian_rbf.png

    Parameters
    ----------
    array : array-like
        input array
    alpha : float, default=1.0
        scale parameter. Reasonable range in `[1, 3]`
    beta : float, default=0.0
        offset parameter. Reasonable range in `[-2, 2]`
    gamma : {0, 1}, default=0
        mirroring parameter.
    tau : even integer, default=2
        power parameter

    Returns
    -------
    transformed_array : array-like

    """
    parcs_assert(ALPHA_MIN <= alpha <= ALPHA_MAX, DataError, f'alpha should be a float in [{ALPHA_MIN}, {ALPHA_MAX}], got {alpha} instead')
    parcs_assert(BETA_MIN <= beta <= BETA_MAX, DataError, f'beta should be a float in [{BETA_MIN}, {BETA_MAX}], got {beta} instead')
    parcs_assert(gamma in GAMMA_RANGE, DataError, f'gamma should be an integer in {GAMMA_RANGE}, got {gamma} instead')
    parcs_assert(tau % 2 == 0, DataError, f'tau should be an even integer, got {tau} instead')

    expon = -alpha * ((array - beta)**tau)

    return gamma + ((-1)**gamma) * np.exp(expon)

@typechecked
def edge_arctan(array: np.ndarray = None,
                alpha: float = 2, beta: float = 0, gamma: np.int_ = 0) -> np.ndarray:
    r"""
    .. math::
        \begin{align}
            z^*_i = (-1)^{\gamma} .
            \arctan{(\alpha(z_i-\beta))},
        \end{align}

    Below figures depict the effect of different parameters.

    .. image:: ../first_steps/img/edge_arctan.png

    Parameters
    ----------
    array : array-like
        input array
    alpha : float, default=2.0
        scale parameter. Reasonable range in `[1, 3]`
    beta : float, default=0.0
        offset parameter. Reasonable range in `[-2, 2]`
    gamma : {0, 1}, default=0
        mirroring parameter.

    Returns
    -------
    transformed_array : array-like

    """
    parcs_assert(ALPHA_MIN <= alpha <= ALPHA_MAX, DataError, f'alpha should be a float in [{ALPHA_MIN}, {ALPHA_MAX}], got {alpha} instead')
    parcs_assert(BETA_MIN <= beta <= BETA_MAX, DataError, f'beta should be a float in [{BETA_MIN}, {BETA_MAX}], got {beta} instead')
    parcs_assert(gamma in GAMMA_RANGE, DataError, f'gamma should be an integer in {GAMMA_RANGE}, got {gamma} instead')
    
    return (-1)**gamma * np.arctan(alpha*(array-beta))

@typechecked
def edge_identity(array: np.ndarray = None) -> np.ndarray:
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
    'gaussian_rbf': edge_gaussian_rbf,
    'arctan': edge_arctan
}

FUNCTION_PARAMS = {
    'identity': [],
    'sigmoid': ['alpha', 'beta', 'gamma', 'tau'],
    'gaussian_rbf': ['alpha', 'beta', 'gamma', 'tau'],
    'arctan': ['alpha', 'beta', 'gamma']
}


if __name__ == '__main__':
    arr = np.random.normal(0, 1, size=300)

    # v = edge_sigmoid_new(array=arr)
    v = edge_gaussian_rbf(array=arr, alpha=1, beta=-1, gamma=1)

    from matplotlib import pyplot as plt
    plt.scatter(arr, v)
    plt.show()
