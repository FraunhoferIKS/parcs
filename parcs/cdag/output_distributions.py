import numpy as np
import pandas as pd
from parcs.sem import mapping_functions
from parcs.cdag.utils import dot_prod
from parcs.cdag.utils import SigmoidCorrection


class GaussianDistribution:
    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        self.params = ['mu_', 'sigma_']
        self.coefs = coefs

        self.do_correction = do_correction
        if do_correction:
            self.sigma_correction = SigmoidCorrection(**correction_config['sigma_'])

    def _correct_param(self, mu_, sigma_):
        sigma_ = self.sigma_correction.transform(sigma_)
        return mu_, sigma_

    def calculate_output(self, data, size):
        mu_ = dot_prod(data, self.coefs['mu_'])
        sigma_ = dot_prod(data, self.coefs['sigma_'])
        if self.do_correction:
            mu_, sigma_ = self._correct_param(mu_, sigma_)
        if data.shape[0] == 0:
            return np.random.normal(mu_, sigma_, size=size)
        else:
            return np.random.normal(mu_, sigma_)


class BernoulliDistribution:
    def __init__(self,
                 coefs=None,
                 do_correction=True,
                 correction_config=None):
        self.params = ['p_']
        self.coefs = coefs

        self.do_correction = do_correction
        if do_correction:
            self.sigma_correction = SigmoidCorrection(**correction_config['p_'])

    def _correct_param(self, p_):
        return self.sigma_correction.transform(p_)

    def calculate_output(self, data, size):
        p_ = dot_prod(data, self.coefs['p_'])
        if self.do_correction:
            p_ = self._correct_param(p_)
        if data.shape[0] == 0:
            return np.random.choice([0, 1], p=[1-p_, p_], size=size)
        else:
            return np.array([np.random.choice([0, 1], p=[1-i, i]) for i in p_])