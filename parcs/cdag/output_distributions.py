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

from parcs.cdag.utils import dot_prod
from parcs.cdag.utils import SigmoidCorrection
from scipy import stats as dists
import numpy as np
from parcs.exceptions import *

DISTRIBUTION_PARAMS = {
    'gaussian': ['mu_', 'sigma_'],
    'lognormal': ['mu_', 'sigma_'],
    'bernoulli': ['p_'],
    'uniform': ['mu_', 'diff_'],
    'exponential': ['lambda_'],
    'poisson': ['lambda_']
}


class GaussianDistribution:
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
        self.params = ['mu_', 'sigma_']
        self.coefs = coefs

        self.do_correction = do_correction
        if do_correction:
            self.sigma_correction = SigmoidCorrection(**correction_config)

    def _correct_param(self, mu_, sigma_):
        sigma_ = self.sigma_correction.transform(sigma_)
        return mu_, sigma_

    def calculate(self, data, errors):
        mu_ = dot_prod(data, self.coefs['mu_'])
        sigma_ = dot_prod(data, self.coefs['sigma_'])
        if self.do_correction:
            mu_, sigma_ = self._correct_param(mu_, sigma_)
        elif isinstance(sigma_, np.ndarray):
            parcs_assert(
                (sigma_ >= 0).sum() == len(sigma_),
                DistributionError,
                'Gaussian sigma_ has negative values'
            )
        else:
            parcs_assert(
                sigma_ >= 0,
                DistributionError,
                'Gaussian sigma_ has negative values'
            )

        samples = dists.norm.ppf(errors, loc=mu_, scale=sigma_)

        return samples


class BernoulliDistribution:
    """ **Gaussian normal distribution**

    Constructed based on
    `Scipy Bernoulli distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html>`_
    distribution.

    """

    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        self.params = ['p_']
        self.coefs = coefs

        self.do_correction = do_correction
        if do_correction:
            self.sigma_correction = SigmoidCorrection(**correction_config)

    def _correct_param(self, p_):
        return self.sigma_correction.transform(p_)

    def calculate(self, data, errors):
        p_ = dot_prod(data, self.coefs['p_'])
        if self.do_correction:
            p_ = self._correct_param(p_)
        elif isinstance(p_, np.ndarray):
            parcs_assert(
                (p_ <= 1).sum() == len(p_),
                DistributionError,
                'Bern(p) probabilities are out of [0, 1] range'
            )
        else:
            parcs_assert(
                0 <= p_ <= 1,
                DistributionError,
                'Bern(p) probabilities are out of [0, 1] range'
            )
        samples = dists.bernoulli.ppf(errors, p_)

        return samples


class UniformDistribution:
    """ **Gaussian normal distribution**

    Since the distribution of the sampled errors is Uniform, this class takes the samples as they are,
    and does loc-scale to satisfy the given parameters.

    """

    def __init__(self,
                 coefs=None,
                 do_correction=False,
                 correction_config=None):
        self.correction_config = correction_config
        self.do_correction = do_correction
        self.params = ['mu_', 'diff_']
        self.coefs = coefs

    def calculate(self, data, errors):
        mu_ = dot_prod(data, self.coefs['mu_'])
        diff_ = dot_prod(data, self.coefs['diff_'])
        low = mu_ - (diff_ / 2)

        samples = errors * diff_ + low

        return samples


class ExponentialDistribution:
    """ **Gaussian normal distribution**

    Constructed based on
    `Scipy Exponential distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html>`_
    distribution.

    """

    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        self.params = ['lambda_']
        self.coefs = coefs

        self.do_correction = do_correction
        if do_correction:
            self.sigma_correction = SigmoidCorrection(**correction_config)

    def _correct_param(self, lambda_):
        return self.sigma_correction.transform(lambda_)

    def calculate(self, data, errors):
        lambda_ = dot_prod(data, self.coefs['lambda_'])
        if self.do_correction:
            lambda_ = self._correct_param(lambda_)
        elif isinstance(lambda_, np.ndarray):
            parcs_assert(
                (lambda_ > 0).sum() == len(lambda_),
                DistributionError,
                'Expon lambda has non-positive values'
            )
        else:
            parcs_assert(
                lambda_ > 0,
                DistributionError,
                'Expon lambda has non-positive values'
            )
        samples = dists.expon.ppf(errors, lambda_)

        return samples


class PoissonDistribution:
    """ **Gaussian normal distribution**

    Constructed based on
    `Scipy Poisson distribution
    <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.poisson.html>`_
    distribution.

    """

    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        self.params = ['lambda_']
        self.coefs = coefs

        self.do_correction = do_correction
        if do_correction:
            self.sigma_correction = SigmoidCorrection(**correction_config)

    def _correct_param(self, lambda_):
        return self.sigma_correction.transform(lambda_)

    def calculate(self, data, errors):
        lambda_ = dot_prod(data, self.coefs['lambda_'])
        if self.do_correction:
            lambda_ = self._correct_param(lambda_)
        elif isinstance(lambda_, np.ndarray):
            parcs_assert(
                (lambda_ >= 0).sum() == len(lambda_),
                DistributionError,
                'Poisson lambda has negative values'
            )
        else:
            parcs_assert(
                lambda_ >= 0,
                DistributionError,
                'Poisson lambda has negative values'
            )
        samples = dists.poisson.ppf(errors, lambda_)

        return samples


class LogNormalDistribution:
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
        self.params = ['mu_', 'sigma_']
        self.coefs = coefs

        self.do_correction = do_correction
        if do_correction:
            self.sigma_correction = SigmoidCorrection(**correction_config)

    def _correct_param(self, mu_, sigma_):
        sigma_ = self.sigma_correction.transform(sigma_)
        return mu_, sigma_

    def calculate(self, data, errors):
        mu_ = dot_prod(data, self.coefs['mu_'])
        sigma_ = dot_prod(data, self.coefs['sigma_'])
        if self.do_correction:
            mu_, sigma_ = self._correct_param(mu_, sigma_)
        elif isinstance(sigma_, np.ndarray):
            parcs_assert(
                (sigma_ >= 0).sum() == len(sigma_),
                DistributionError,
                'Log normal sigma_ has negative values'
            )
        else:
            parcs_assert(
                sigma_ >= 0,
                DistributionError,
                'Log normal sigma_ has negative values'
            )

        samples = dists.lognorm.ppf(errors, loc=mu_, scale=sigma_)

        return samples


OUTPUT_DISTRIBUTIONS = {
    'gaussian': GaussianDistribution,
    'lognormal': LogNormalDistribution,
    'bernoulli': BernoulliDistribution,
    'uniform': UniformDistribution,
    'exponential': ExponentialDistribution,
    'poisson': PoissonDistribution
}
