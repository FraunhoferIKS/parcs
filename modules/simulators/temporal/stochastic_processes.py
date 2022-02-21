import numpy as np
import pandas as pd


class BrownianMotion:
    def __init__(self,
                 geometric: bool = False,
                 sampled_latents: pd.DataFrame = None,
                 drift_column: str = 'drift',
                 scale_column: str = 'scale',
                 init_column: str = 'init'):
        self.latent = sampled_latents
        self.mu = self.latent[drift_column].values.reshape(-1, 1)
        self.sigma = self.latent[scale_column].values.reshape(-1, 1)
        self.s0 = self.latent[init_column].values.reshape(-1, 1)
        self.geometric = geometric
        if geometric:
            self.s0 = np.log(self.s0)

    def _standard_sample(self, seq_len: int = None):
        return self.s0 + np.concatenate(
            np.zeros(shape=(len(self.latent), 1)),
            np.random.normal(
                self.mu, self.sigma, size=(len(self.latent), seq_len-1)
            ).cumsum(axis=1)
        )

    def sample(self, seq_len: int = 10):
        if self.geometric:
            return np.exp(self._standard_sample(seq_len=seq_len))
        else:
            return self._standard_sample(seq_len=seq_len)


class UnmarkedExponentialEvent:
    def __init__(self,
                 sampled_latents: pd.DataFrame = None,
                 rate_column: str = 'rate'):
        self.latent = sampled_latents
        self.rate = self.latent[rate_column].values.reshape(-1, 1)

    def sample(self, num_events: int = 10):
        return np.random.exponential(scale=1 / self.rate, size=(len(self.latent), num_events)).cumsum(axis=1)