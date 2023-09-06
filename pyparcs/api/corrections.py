#  Copyright (c) 2023. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
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

import numpy
import numpy as np
from scipy.special import expit


class SigmoidCorrection:
    r"""
    This object, transforms the values on :math:`\mathbb{R}` support to a `[L, U]` range
    using the sigmoid function. The Equation is:

    .. math::
        \begin{align}
            x^{'} = ( U - L ) \sigma(x - x_0) + L, \quad \sigma(a) = \frac{1}{1+e^{-a}}
        \end{align}

    where :math:`U` and :math:`L` are user-defined upper and lower bounds for transformed variable,
    and :math:`X_0` is the `offset` which is defined according to user needs, defined by
    ``target_mean`` and ``to_center`` parameters. see the parameter descriptions below for more
    details.

    Parameters
    ----------
    lower, upper : float
        lower and upper bounds for transformed variable
    target_mean : float, default=None
        If a float value (not ``None``), then the mean of transformed value is fixed. This value
        must be in the `[L, U]` range.

    Raises
    ------
    AssertionError
        if the `target_mean` doesn't lie in the lower, upper range or `lower >= upper`

    Examples
    --------
    This class is used internally by PARCS if `correction` parameter is chosen for a node. However,
    to understand the functionality better, we make an example using the class:

    >>> x = np.linspace(-10, 10, 200)
    >>> sc = SigmoidCorrection(lower=-3, upper=2)
    >>> x_t = sc.transform(x)
    >>> print(np.round(x_t.min(), 3), np.round(x_t.max(), 3), np.round(x_t.mean(), 3))
    -3.0 2.0 -0.5
    >>> sc_2 = SigmoidCorrection(lower=0, upper=1, target_mean=0.8)
    >>> x_t = sc_2.transform(x)
    >>> print(np.round(x_t.min(), 3), np.round(x_t.max(), 3), np.round(x_t.mean(), 3))
    0.019 1.0 0.8

    .. note::
        If ``target_mean`` is given, sigmoid correction searches for an offset term to add
        to the input values, such that the required mean is obtained. The process is a manual
        search near the support of data points.
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
    This object normalizes the input variables using the mean and standard deviation
    of the first data batch** that it receives.

    .. math::
        \begin{align}
            x^{'} = \frac{x-\mu_{b_1}}{\sigma_{b_1}}
        \end{align}


    Examples
    --------
    This class is used internally by PARCS if `correction` parameter is chosen for an edge. However,
    to understand the functionality better, we make an example using the class:

    >>> x = np.random.normal(2, 10, size=200)
    >>> ec = EdgeCorrection()
    >>> # This is the first batch
    >>> x_t = ec.transform(x)
    >>> print(np.round(x_t.mean(), 2), np.round(x_t.std(), 2))
    0.0 1.0
    >>> # Give the second batch: mean and std are already fixed according to x batch.
    >>> y = np.random.normal(-1, 2, size=300)
    >>> y_t = ec.transform(y)
    >>> print(np.round(y_t.mean(), 2), np.round(y_t.std(), 2))
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
