#  Copyright (c) 2022. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
#  acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses/>.
#
#  https://www.gnu.de/documents/gpl-2.0.de.html
#
#  Contact: alireza.zamanian@iks.fraunhofer.de

import numpy as np
from scipy.special import expit
from pyparcs.core.exceptions import EdgeFunctionError, parcs_assert


def edge_empty(**kwargs):
    assert kwargs
    print('edge function not implemented')
    raise ValueError


def edge_identity(array: np.ndarray) -> np.ndarray:
    """Identity edge function"""
    return array


def edge_sigmoid(array: np.ndarray,
                 alpha: float = 2.0, beta: float = 0.0,
                 gamma: float = 0, tau: float = 1) -> np.ndarray:
    """Sigmoid edge function"""
    parcs_assert(gamma in [0, 1] and tau % 2 == 1,
                 EdgeFunctionError,
                 f'error in gamma={gamma} or tau={tau} values')

    expon = (-1) ** gamma * alpha * ((array - beta) ** tau)
    return expit(expon)


def edge_gaussian_rbf(array: np.ndarray,
                      alpha: float = 1.0, beta: float = 0,
                      gamma: float = 0, tau: float = 2) -> np.ndarray:
    """Gaussian RBF edge function"""
    parcs_assert(gamma in [0, 1] and tau % 2 == 0,
                 EdgeFunctionError,
                 f'error in gamma={gamma} or tau={tau} values')

    expon = -alpha * ((array - beta) ** tau)
    return gamma + ((-1) ** gamma) * np.exp(expon)


def edge_arctan(array: np.ndarray,
                alpha: float = 2, beta: float = 0, gamma: float = 0) -> np.ndarray:
    """Arctan edge function"""
    parcs_assert(gamma in [0, 1],
                 EdgeFunctionError,
                 f'error in gamma={gamma} value')

    return (-1) ** gamma * np.arctan(alpha * (array - beta))


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
