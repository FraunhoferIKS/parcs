from scipy import stats as dists
from scipy.special import expit
import numpy as np
from numpy import array


class EdgeFunctionsData:
    # tests identity
    # inputs: array
    identity_data = [
        (array([1, 2, 3])),
        (array([])),
        (array([-1, 0, 2.9]))
    ]

    # tests sigmoid
    # inputs: input array, sigmoid params, expected output
    sigmoid_data = [
        (inp := array([-3, -2, 0, 1, 2, 3]),
         {'alpha': 1, 'beta': 0, 'gamma': 0, 'tau': 1},
         expit(inp)),
        (inp := array([-3, -2, 0, 1, 2, 3]),
         {'alpha': 2, 'beta': 1, 'gamma': 0, 'tau': 1},
         expit(2*inp-2))
    ]

    # tests gaussian rbf
    # inputs: input array, sigmoid params, expected output
    gaussian_rbf_data = [
        (inp := array([-3, -2, 0, 1, 2, 3]),
         {'alpha': 1, 'beta': 0, 'gamma': 0, 'tau': 2},
         np.exp(-inp**2)),
        (inp := array([-3, -2, 0, 1, 2, 3]),
         {'alpha': 2, 'beta': 1, 'gamma': 1, 'tau': 4},
         1 - np.exp(-2 * (inp - 1)**4))
    ]

    # tests arctan
    # inputs: input array, sigmoid params, expected output
    arctan_data = [
        (inp := array([-3, -2, 0, 1, 2, 3]),
         {'alpha': 1, 'beta': 0, 'gamma': 0},
         np.arctan(inp)),
        (inp := array([-3, -2, 0, 1, 2, 3]),
         {'alpha': 2, 'beta': 1, 'gamma': 1},
         -np.arctan(2*(inp - 1)))
    ]
