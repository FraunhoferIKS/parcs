import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.special import expit


class IndependentNormalLatents:
    def __init__(self):
        self.data = None
        self.var_list = None

    def set_nodes(self, var_list=None):
        self.var_list = var_list
        return self

    def sample(self, sample_size: int = 100):
        self.data = pd.DataFrame({
            var['name']: np.random.normal(var['mean'], var['sigma'], size=sample_size)
            for var in self.var_list
        })
        return self.data


class IndependentUniformLatents:
    def __init__(self):
        self.data = None
        self.var_list = None

    def set_nodes(self, var_list=None):
        self.var_list = var_list
        return self

    def sample(self, sample_size: int = 100):
        self.data = pd.DataFrame({
            var['name']: np.random.uniform(var['low'], var['high'], size=sample_size)
            for var in self.var_list
        })
        return self.data


class LatentLabelMaker:
    def __init__(self, coef_min: float = 1, coef_max: float = 4, normalize_latent: bool = True):
        self.normalize_latent = normalize_latent
        self.coef_min = coef_min
        self.coef_max = coef_max

        self.coefs_sampled = False
        self.coefs = None

    def _sample_coefs(self, size=None):
        self.coefs = np.random.uniform(self.coef_min, self.coef_max, size=size)
        return self

    @staticmethod
    def _normalize(x):
        return (x - x.mean(axis=0)) / x.std(axis=0)

    def make_label(self, sampled_latents: np.array = None, offset: float = 0):
        if self.normalize_latent:
            sampled_latents = self._normalize(sampled_latents)
        self._sample_coefs(size=sampled_latents.shape[1])
        states = np.dot(sampled_latents.values, self.coefs)
        norm_states = offset + self._normalize(states)
        probs = expit(norm_states)
        return np.array([
            np.random.choice([0, 1], p=[1-prob, prob])
            for prob in probs
        ])
