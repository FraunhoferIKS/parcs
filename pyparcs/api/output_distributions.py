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

from typing import Callable, List
import numpy as np
import pandas as pd
from scipy import stats as dists
from pyparcs.api.utils import dot_prod
from pyparcs.api.corrections import SigmoidCorrection
from pyparcs.core.exceptions import parcs_assert, DistributionError

DISTRIBUTION_PARAMS = {
    'normal': ['mu_', 'sigma_'],
    'lognormal': ['mu_', 'sigma_'],
    'bernoulli': ['p_'],
    'uniform': ['mu_', 'diff_'],
    'exponential': ['lambda_'],
    'poisson': ['lambda_']
}


class DistParam:
    """**Distribution Parameter class**

    This class represents the parameters of an output distribution

    Parameters
    ----------
    name: str
        name of the parameter
    coef: array-like
        coefficient vector for the parameter. The size of the vector
        is equal to the number of the parents of the node
    corrector: SigmoidCorrection or None
        The correction object to apply node correction for the parameter
    """
    def __init__(self, name: str, coef: np.array, corrector):
        self.name = name
        self.coef = coef
        self.corrector = corrector

    def calculate(self, data: np.ndarray) -> np.ndarray:
        """
        Calculates the value of the parameter given the input data

        Parameters
        ----------
        data: ndarray

        Returns
        -------
        calculated_parameter: ndarray
            N-length parameter value for N data points
        """
        raw = dot_prod(data, self.coef)
        if self.corrector is not None:
            raw = self.corrector.transform(raw)
        return raw


class PARCSDistribution:
    """**PARCS distribution base class**

    This provides a base class for PARCS nodes' output distribution

    Parameters
    ----------
    icdf: callable
        The inverse CDF function of the distribution, called for sampling
        based on uniformly-distributed error terms. The icdf function accepts
        `number of parameters + 1` inputs. The first input is the error terms.
        the next inputs are keyword arguments. See
        :func:`pyparcs.api.output_distributions.PARCSDistribution._parcs_to_icdf_map_param`
        for a note on the parameter names.
    params: list of str
        list of parameter names
    coefs: dict of array-likes
        key is the parameter name, and the value is the coefficient vector
    correctors: dict of None or SigmoidCorrection
        If not None, then the correction class is assigned to the key

    Attributes
    ----------
    params: dict of DistParams
        See :func:`pyparcs.api.output_distributions.DistParam`
    icdf: callable
        the passed iCDF function
    """
    def __init__(self, icdf: Callable, params: List[str], coefs: dict, correctors: dict):
        self.params = {
            p_name: DistParam(
                name=p_name,
                coef=coefs[p_name],
                corrector=None if correctors[p_name] is None else correctors[p_name]
            ) for p_name in params
        }
        self.icdf = icdf

    def _validate_params(self, params):
        pass

    @staticmethod
    def parcs_to_icdf_map_param(params: dict) -> dict:
        """**map from PARCS to iCDF parameter names**

        This method maps from the parameter names that the user uses in the outline
        and the parameter names that the icdf function accepts. If the icdf is a
        custom function, we can simply use the same names and do not change this
        method. But if existing icdf functions, such as scipy stats functions is
        used, then this method helps us to still assign readable names to the
        PARCS distribution parameters, and then pass it to the icdf function.

        Parameters
        ----------
        params: dict of param values
            `{'icdf-param-name': values}` dictionary. The size of values for all
            parameters must be either 1, or equal to number of data points to sample

        Returns
        -------
        renamed_params_dict: dict of param values
            the same params with the renamed keys
        """
        return params

    @staticmethod
    def validate_support(node_data: pd.Series):
        """
        validate if sampled data complies with the support of the distribution.
        This check is necessary as the user directly assigns values to a node
        during the interventions on a graph.

        Parameters
        ----------
        node_data: pandas Series
            the sampled data for the node

        Returns
        ------
        is_valid: bool
            if the node data complies with the distribution support
        """
        return True

    def calculate(self, data: np.ndarray, errors: pd.Series) -> np.ndarray:
        """
        calculate the realizations of the distribution based on given errors.

        Parameters
        ----------
        data : ndarray
            the input data needed to calculate the distribution parameters
        errors : pandas Series
            sampled uniform errors

        Returns
        -------
        sampled realizations : ndarray
            calculated values based on distribution parameters and sampled errors
        """
        param_realization = {
            param: self.params[param].calculate(data)
            for param in self.params
        }  # for each param, calculate the values based on parents values
        self._validate_params(param_realization)  # (e.g. bernoulli p > 1 is invalid)
        # calculate the realizations based on the error
        return self.icdf(errors, **self.parcs_to_icdf_map_param(param_realization))


class BernoulliDistribution(PARCSDistribution):
    """ **Bernoulli distribution**

    Constructed based on
    `Scipy Bernoulli distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html>`_
    distribution.
    """
    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        super().__init__(icdf=dists.bernoulli.ppf, params=['p_'], coefs=coefs,
                         correctors={'p_': SigmoidCorrection(**correction_config) if do_correction
                         else None})

    def _validate_params(self, params):
        p_ = params['p_']
        if isinstance(p_, np.ndarray):
            parcs_assert((p_ <= 1).sum() == len(p_),
                         DistributionError,
                         "Bern(p) probabilities are out of [0, 1] range")
        else:
            parcs_assert(0 <= p_ <= 1, DistributionError,
                         "Bern(p) probabilities are out of [0, 1] range")

    @staticmethod
    def parcs_to_icdf_map_param(params):
        return {'p': params['p_']}

    @staticmethod
    def validate_support(node_data):
        return set(node_data.unique()) == {0, 1}


class GaussianNormalDistribution(PARCSDistribution):
    """ **Gaussian normal distribution**

    Constructed based on
    `Scipy norm distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm>`_
    distribution.
    """
    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        super().__init__(icdf=dists.norm.ppf, params=['mu_', 'sigma_'], coefs=coefs,
                         correctors={
                             'mu_': None,
                             'sigma_': SigmoidCorrection(**correction_config) if do_correction
                             else None,
                         })

    def _validate_params(self, params):
        sigma_ = params['sigma_']
        if isinstance(sigma_, np.ndarray):
            parcs_assert((sigma_ >= 0).sum() == len(sigma_),
                         DistributionError,
                         "Gaussian normal sigma_ has negative values")
        else:
            parcs_assert(sigma_ >= 0, DistributionError,
                         "Gaussian normal sigma_ has negative values")

    @staticmethod
    def parcs_to_icdf_map_param(params):
        return {'loc': params['mu_'], 'scale': params['sigma_']}


class UniformDistribution(PARCSDistribution):
    """ **Uniform distribution**

    Since the distribution of the sampled errors is Uniform, this class takes the samples
    as they are, and does loc-scale to satisfy the given parameters.
    """
    def __init__(self,
                 coefs=None,
                 do_correction=False,
                 correction_config=None):
        parcs_assert(
            not do_correction and not correction_config,
            DistributionError,
            "Uniform distribution does not accept any node correction."
        )

        def yield_icdf():
            def uniform_icdf(errors, mu_=None, diff_=None):
                return mu_ + (np.array(errors) - 0.5) * diff_

            return uniform_icdf
        super().__init__(icdf=yield_icdf(), params=['mu_', 'diff_'], coefs=coefs,
                         correctors={'mu_': None, 'diff_': None})


class ExponentialDistribution(PARCSDistribution):
    """ **Exponential distribution**

    Constructed based on
    `Scipy Exponential distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html>`_
    distribution.
    """
    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        super().__init__(
            icdf=dists.expon.ppf,
            params=['lambda_'],
            coefs=coefs,
            correctors={'lambda_': SigmoidCorrection(**correction_config)
                        if do_correction else None}
        )

    def _validate_params(self, params):
        lambda_ = params['lambda_']
        if isinstance(lambda_, np.ndarray):
            parcs_assert((lambda_ > 0).sum() == len(lambda_),
                         DistributionError,
                         "Exponential lambda has non-positive values")
        else:
            parcs_assert(lambda_ > 0, DistributionError,
                         "Exponential lambda has non-positive values")

    @staticmethod
    def parcs_to_icdf_map_param(params):
        return {'loc': params['lambda_']}

    @staticmethod
    def validate_support(node_data):
        return (node_data >= 0).all()


class PoissonDistribution(PARCSDistribution):
    """ **Poisson distribution**

    Constructed based on
    `Scipy Poisson distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html>`_
    distribution.
    """
    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        super().__init__(
            icdf=dists.poisson.ppf,
            params=['lambda_'],
            coefs=coefs,
            correctors={'lambda_': SigmoidCorrection(**correction_config)
                        if do_correction else None}
        )

    def _validate_params(self, params):
        lambda_ = params['lambda_']
        if isinstance(lambda_, np.ndarray):
            parcs_assert((lambda_ > 0).sum() == len(lambda_),
                         DistributionError,
                         "Poisson lambda has non-positive values")
        else:
            parcs_assert(lambda_ > 0, DistributionError, "Poisson lambda has non-positive values")

    @staticmethod
    def parcs_to_icdf_map_param(params):
        return {'mu': params['lambda_']}

    @staticmethod
    def validate_support(node_data):
        assert (node_data >= 0).all() and ((node_data - node_data.astype('int')) == 0).all()


class LogNormalDistribution(PARCSDistribution):
    """ **Gaussian normal distribution**

    Constructed based on
    `Scipy lognorm distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.lognorm.html>`_
    distribution.
    """
    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        super().__init__(icdf=dists.lognorm.ppf, params=['mu_', 'sigma_'], coefs=coefs,
                         correctors={
                             'mu_': None,
                             'sigma_': SigmoidCorrection(**correction_config) if do_correction
                             else None,
                         })

    def _validate_params(self, params):
        sigma_ = params['sigma_']
        if isinstance(sigma_, np.ndarray):
            parcs_assert((sigma_ >= 0).sum() == len(sigma_),
                         DistributionError,
                         "log normal sigma_ has negative values")
        else:
            parcs_assert(sigma_ >= 0, DistributionError, "log normal sigma_ has negative values")

    @staticmethod
    def validate_support(node_data):
        assert (node_data > 0).all()

    @staticmethod
    def parcs_to_icdf_map_param(params):
        return {'loc': params['mu_'], 's': params['sigma_']}


OUTPUT_DISTRIBUTIONS = {
    'normal': GaussianNormalDistribution,
    'lognormal': LogNormalDistribution,
    'bernoulli': BernoulliDistribution,
    'uniform': UniformDistribution,
    'exponential': ExponentialDistribution,
    'poisson': PoissonDistribution
}
