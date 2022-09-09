from parcs.cdag.utils import dot_prod
from parcs.cdag.utils import SigmoidCorrection
from scipy import stats as dists
import numpy as np

DISTRIBUTION_PARAMS = {
    'gaussian': ['mu_', 'sigma_'],
    'bernoulli': ['p_']
}

class GaussianDistribution:
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

        samples = dists.norm.ppf(errors, loc=mu_, scale=sigma_)

        return samples

class BernoulliDistribution:
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
        else:
            assert (np.abs(p_)<=1).sum() == len(p_), 'Bern(p) probabilities are out of [0, 1] range'
        samples = dists.bernoulli.ppf(errors, p_)

        return samples

OUTPUT_DISTRIBUTIONS = {
    'gaussian': GaussianDistribution,
    'bernoulli': BernoulliDistribution
}