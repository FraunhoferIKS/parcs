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

import warnings
import numpy as np
import pandas as pd
from scipy.special import expit
from itertools import combinations_with_replacement as comb_w_repl
from pyparcs.exceptions import parcs_assert


def topological_sort(adj_matrix: pd.DataFrame = None):
    """
    performs topological sorting of a given adjacency matrix. It is an implementation of Kahn's algorithm.

    Parameters
    ----------
    adj_matrix : pd.DataFrame
        the adjacency matrix to be sorted

    Returns
    -------
    sorted_nodes : list(str)
        sorted list of all nodes


    Raises
    ------
    ValueError
        If index and column names doesn't match
    ValueError
        If the adjacency matrix is not acyclic

    """
    parcs_assert(
        set(adj_matrix.columns) == set(adj_matrix.index),
        ValueError,
        "index and column names in the adjacency matrix doesn't match"
    )

    adjm = adj_matrix.copy(deep=True).values
    ordered_list = []
    covered_nodes = 0
    while covered_nodes < adjm.shape[0]:
        # sum r/x -> r edges
        sum_c = adjm.sum(axis=0)
        # find nodes with no parents
        parent_inds = list(np.where(sum_c == 0)[0])
        parcs_assert(len(parent_inds) != 0, ValueError, "Adjacency matrix is not acyclic")

        covered_nodes += len(parent_inds)
        # add to the list
        ordered_list += parent_inds
        # remove parent edges
        adjm[parent_inds, :] = 0
        # eliminate from columns by assigning values
        adjm[:, parent_inds] = 10
    return [adj_matrix.columns.tolist()[idx] for idx in ordered_list]


def is_adj_matrix_acyclic(adj_matrix):
    try:
        topological_sort(adj_matrix)
        return True
    except AssertionError:
        return False


def get_interactions(data):
    """ **Creates interaction terms**

    Returns the columns of product of interaction terms. The interaction terms are of length 2.
    The order of interaction terms follow the order
    of ``itertools.combination_with_replacement`` module. Example: for ``[X,Y,Z]`` the method returns:
    ``[XX, XY, XZ, YY, YZ, ZZ]``.

    Parameters
    ----------
    data : array-like
        with `n x m` shape, `m` being the number of features

    Returns
    -------
    data : array-like
        interaction terms

    Examples
    --------
    >>> from pyparcs.cdag.utils import get_interactions
    >>> data_ = np.array([[1, 2, 3], [10, 12, 20]])
    >>> get_interactions(data_)
    array([[  1,   2,   3,   4,   6,   9],
           [100, 120, 200, 144, 240, 400]])
    >>> data_ = np.array([[1], [2]])
    >>> get_interactions(data_)
    array([[1],
           [4]])
    """
    out = np.array([
        [np.prod(i) for i in comb_w_repl(row, 2)]
        for row in data
    ])
    return out


def get_interactions_length(len_):
    """ **Returns length of interaction terms**

    This function returns the length of the output of :func:`~cdag.parcs.utils.get_interactions`.

    Parameters
    ----------
    len_ : int
        `shape[1]` of the raw data (number of columns)

    Returns
    -------
    length : int
        length of the interaction data

    Examples
    --------
    >>> from pyparcs.cdag.utils import get_interactions, get_interactions_length
    >>> import numpy
    >>> data = numpy.random.normal(size=(9, 3))
    >>> interactions = get_interactions(data)
    >>> interaction_len = get_interactions_length(data.shape[1])
    >>> interactions.shape[1] == interaction_len
    True
    """
    dummy_data = np.ones(shape=(len_,))
    return len([
        np.prod(i)
        for i in comb_w_repl(dummy_data, 2)
    ])


def get_interactions_dict(parents):
    """ **gives parents pairings for each interaction term**

    This function is used to trace which parents are making an interaction term in some index.

    Parameters
    ----------
    parents : list of str
        list of parents

    Returns
    -------
    pairings : list of tuple
        list of all parent pairings, where the index corresponds to the interaction data

    Examples
    --------
    >>> from pyparcs.cdag.utils import get_interactions, get_interactions_dict
    >>> parents_ = ['a', 'b', 'c']
    >>> get_interactions_dict(parents_)
    [('a', 'a'), ('a', 'b'), ('a', 'c'), ('b', 'b'), ('b', 'c'), ('c', 'c')]

    """
    return [sorted(i) for i in comb_w_repl(parents, 2)]


def dot_prod(data, coef):
    """
    dot product of data and bias/linear/interactions coefs

    >>> import numpy
    >>> data = numpy.array([[1], [2]])
    >>> coef = {'bias': 0, 'linear': [1], 'interactions': numpy.array([1])}
    >>> dot_prod(data, coef)
    array([2, 6])
    """
    if data.shape[0] == 0:
        data_augmented = [np.array([]) for _ in range(3)]
    else:
        data_augmented = [data, get_interactions(data)]

    return coef['bias'] + np.array([
        np.dot(d, coef[i])
        for (i, d) in zip(['linear', 'interactions'], data_augmented)
    ]).sum(axis=0)


class SigmoidCorrection:
    r"""
    This object, transforms the values on :math:`\mathbb{R}` support to a `[L, U]` range
    using the sigmoid function. The Equation is:

    .. math::
        \begin{align}
            x^{'} = ( U - L ) \sigma(x - x_0) + L, \quad \sigma(a) = \frac{1}{1+e^{-a}}
        \end{align}

    where :math:`U` and :math:`L` are user-defined upper and lower bounds for transformed variable, and :math:`X_0`
    is the `offset` which is defined according to user needs, defined by ``target_mean`` and ``to_center`` parameters.
    see the parameter descriptions below for more details.

    Parameters
    ----------
    lower, upper : float
        lower and upper bounds for transformed variable
    target_mean : float, default=None
        If a float value (not ``None``), then the mean of transformed value is fixed. This value must be
        in the `[L, U]` range.

    Raises
    ------
    AssertionError
        if the `target_mean` doesn't lie in the lower, upper range or `lower >= upper`

    Examples
    --------
    This class is used internally by PARCS if `correction` parameter is chosen for a node. However, to understand
    the functionality better, we make an example using the class:

    >>> from pyparcs.cdag.utils import SigmoidCorrection
    >>> import numpy
    >>> x = numpy.linspace(-10, 10, 200)
    >>> sc = SigmoidCorrection(lower=-3, upper=2)
    >>> x_t = sc.transform(x)
    >>> print(numpy.round(x_t.min(), 3), numpy.round(x_t.max(), 3), numpy.round(x_t.mean(), 3))
    -3.0 2.0 -0.5
    >>> sc_2 = SigmoidCorrection(lower=0, upper=1, target_mean=0.8)
    >>> x_t = sc_2.transform(x)
    >>> print(numpy.round(x_t.min(), 3), numpy.round(x_t.max(), 3), numpy.round(x_t.mean(), 3))
    0.019 1.0 0.8

    .. note::
        If ``target_mean`` is given, sigmoid correction searches for an offset term to add to the input values,
        such that the required mean is obtained. The process is a manual search near the support of data points.
    """

    def __init__(self, lower=0, upper=1, target_mean=None):
        assert upper > lower
        if target_mean is not None:
            assert lower < target_mean < upper
        self.config = {
            'lower': lower,
            'upper': upper,
            'offset': 0
        }
        self.is_initialized = False
        self.target_mean = target_mean

    def get_params(self):
        assert self.is_initialized
        return self.config

    def transform(self, array):
        """
        transform the input variable according to parameters set upon instantiation.

        Parameters
        ----------
        array : array-like
            input array

        Returns
        -------
        transformed_array : array-like
            transformed array by the sigmoid correction
        """
        if not self.is_initialized:
            # transform by sigmoid
            U = (self.config['upper'] - self.config['lower'])
            L = self.config['lower']
            # fixing the mean via another offset
            if self.target_mean is not None:
                # min - I =  6 -> I = min - 6
                # max - I = -6 -> I = max + 6
                error = np.inf
                theta = 0
                for i in np.linspace(np.min(array) - 6, np.max(array) + 6, 1000):
                    h = U * expit(array - i) + L
                    new_error = abs(h.mean() - self.target_mean)
                    if new_error <= error:
                        theta = i
                        error = new_error
                    else:
                        break
                self.config['offset'] = theta
            self.is_initialized = True
        return (self.config['upper'] - self.config['lower']) * \
            expit(array - self.config['offset']) + \
            self.config['lower']


class EdgeCorrection:
    r"""
    This object normalizes the input variables using the mean and standard deviation **of the first data batch** that
    it receives.

    .. math::
        \begin{align}
            x^{'} = \frac{x-\mu_{b_1}}{\sigma_{b_1}}
        \end{align}


    Examples
    --------
    This class is used internally by PARCS if `correction` parameter is chosen for an edge. However, to understand
    the functionality better, we make an example using the class:

    >>> from pyparcs.cdag.utils import EdgeCorrection
    >>> import numpy
    >>> x = numpy.random.normal(2, 10, size=200)
    >>> ec = EdgeCorrection()
    >>> # This is the first batch
    >>> x_t = ec.transform(x)
    >>> print(numpy.round(x_t.mean(), 2), numpy.round(x_t.std(), 2))
    0.0 1.0
    >>> # Give the second batch: mean and std are already fixed according to x batch.
    >>> y = numpy.random.normal(-1, 2, size=300)
    >>> y_t = ec.transform(y)
    >>> print(numpy.round(y_t.mean(), 2), numpy.round(y_t.std(), 2))
    -0.36 0.19

    """

    def __init__(self):
        self.is_initialized = False
        self.config = {
            'offset': None,
            'scale': None
        }

    def get_params(self):
        assert self.is_initialized
        return self.config

    def transform(self, array):
        """
        transform the input variable according to parameters set upon instantiation.

        Parameters
        ----------
        array : array-like
            input array

        Returns
        -------
        transformed_array : array-like
            transformed array by the edge correction
        """
        if not self.is_initialized:
            try:
                assert len(array) >= 500
            except AssertionError:
                warnings.warn(
                    """
                    PARCS calculate normalization statistics from the first input batch,
                    This is the 1st batch, while size < 500. It might lead to instabilities.
                    we recommend to run the first simulation run with greater size
                    """
                )
            self.config['offset'] = array.mean()
            self.config['scale'] = array.std()
            self.is_initialized = True

        return (array - self.config['offset']) / self.config['scale']
