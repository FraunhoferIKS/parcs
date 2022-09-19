import numpy as np
import pandas as pd


class BrownianMotion:
    """
    Simulates Brownian motion

    read on user guide... coming soon

    Parameters
    ----------
    geometric : bool, default=False
        set to `True` for geometric brownian motion.
    sampled_latents : pd.DataFrame
        `n x p` size Dataframe of latent realizations for n samples and p latent variables.
    drift_column : str, default='drift'
        column name for drift in `sampled_latents` dataframe
    scale_column : str, default='scale'
        column name for scale in `sampled_latents` dataframe
    init_column : str, default='init'
        column name for initial value of series in `sampled_latents` dataframe

    Examples
    --------
    >>> import numpy as np
    >>> from parcs.simulators.temporal.stochastic_processes import BrownianMotion
    >>> np.random.seed(1)
    >>> latents = pd.DataFrame(
    ...     [
    ...         [1, 1.2, 0],
    ...         [-0.5, 1, 4]
    ...     ], columns=('drift', 'scale', 'init')
    ... )
    >>> bm = BrownianMotion(sampled_latents=latents)
    >>> np.round(bm.sample(seq_len=4), 2)
    array([[ 0.  ,  2.95,  3.22,  3.58],
           [ 4.  ,  2.43,  2.79, -0.01]])
    """
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
        return self.s0 + np.concatenate([
            np.zeros(shape=(len(self.latent), 1)),
            np.random.normal(
                self.mu, self.sigma, size=(len(self.latent), seq_len-1)
            ).cumsum(axis=1)
        ], axis=1)

    def sample(self, seq_len: int = 10):
        """
        sample TSN

        Parameters
        ----------
        seq_len : int, default=10

        Returns
        -------
        D : array-like with shape `(n,t)`
            matrix where rows are samples and columns are time points.
        """
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