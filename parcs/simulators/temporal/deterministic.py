import numpy as np
import pandas as pd


class FourierSeries:
    """
    Simulates Fourier series signals.

    read on user guide... to be continued

    Parameters
    ----------
    sampled_latents : pd.DataFrame
        `n x p` size Dataframe of latent realizations for n samples and p latent variables.
    dominant_amplitude : positive float, default=1
        amplitude of the dominant frequency in the series.
    amplitude_exp_decay_rate : positive float, default=1
        exponential decay rate for the amplitude of next frequencies
    added_noise_sigma_ratio : float, default=0.0
        determines the added noise sigma s.t. `sigma = dominant_amplitude * added_noise_sigma_ratio`
        if set to `0.0`, then no added noise
    frequency_prefix : str, default='w'
        prefix for the frequency columns in `sampled_latents` dataframe
    phaseshift_prefix : str, default='phi'
        prefix for the phase shift columns in `sampled_latents` dataframe


    Examples
    --------
    Simulating two signals with 2 dominant frequencies for 5 time points.
    First signal has the frequencies :math:`\{\pi/50, \pi/70\}` and phase shifts :math:`\{0, \pi/2\}`.
    Second signal has the frequencies :math:`\{\pi/20, \pi/30\}` and phase shifts :math:`\{0, \pi/8\}`.

    >>> from parcs.simulators.temporal.deterministic import FourierSeries
    >>> import numpy as np
    >>> latents = pd.DataFrame([
    ...     [np.pi / 50, np.pi / 70, 0, np.pi / 2],
    ...     [np.pi / 20, np.pi / 30, 0, np.pi / 8],
    ... ], columns=('w_0', 'w_1', 'phi_0', 'phi_1'))
    >>> fs = FourierSeries(sampled_latents=latents)
    >>> np.round(fs.sample(seq_len=5), 2)
    array([[0.37, 0.43, 0.49, 0.55, 0.61],
           [0.14, 0.33, 0.52, 0.69, 0.85]])
    """

    def __init__(self,
                 sampled_latents: pd.DataFrame = None,
                 dominant_amplitude: float = 1,
                 amplitude_exp_decay_rate: float = 1,
                 added_noise_sigma_ratio: float = 0.0,
                 frequency_prefix: str = 'w',
                 phaseshift_prefix: str = 'phi'):
        self.latents = sampled_latents
        self.noise_sigma = added_noise_sigma_ratio
        self.dominant_amplitude = dominant_amplitude

        # get latent columns
        self.frequency_columns = sorted([
            i for i in sampled_latents.columns if i.split('_')[0] == frequency_prefix
        ])
        self.phaseshift_columns = sorted([
            i for i in sampled_latents.columns if i.split('_')[0] == phaseshift_prefix
        ])
        # latent values
        self.frequencies = self.latents[self.frequency_columns].values
        self.phaseshifts = self.latents[self.phaseshift_columns].values

        # setup amplitude
        num_freqs = len(self.frequency_columns)
        self.amplitudes = dominant_amplitude * np.exp(-amplitude_exp_decay_rate * np.arange(num_freqs)) * \
                          np.ones(shape=(len(self.latents), num_freqs))

    def sample(self, seq_len: int = 10):
        """
        sampled Fourier series

        Parameters
        ----------
        seq_len : int, default=10

        Returns
        -------
        D : array-like with shape `(n,t)`
            matrix where rows are samples and columns are time points.
        """
        t = np.arange(seq_len)
        # reshape [n x w_i] -> [n x w_i x 1], because it will be outer product with time
        freqs = self.frequencies.reshape(*self.frequencies.shape, 1)
        amps = self.amplitudes.reshape(*self.amplitudes.shape, 1)
        phis = self.phaseshifts.reshape(*self.phaseshifts.shape, 1)
        # calculate bucket of sins for samples and reshape again to [ n x w x t]
        decomposed = (amps * np.sin(freqs * t + phis)).reshape(*self.frequencies.shape, -1)
        # add sine waves
        data = decomposed.sum(axis=1)
        # add noise
        data = data + np.random.normal(0, self.dominant_amplitude * self.noise_sigma, size=data.shape)
        return data


class TSN:
    """
    Simulates Trend Seasonal Noise signals.

    read on user guide... to be continued

    Parameters
    ----------
    sampled_latents : pd.DataFrame
        `n x p` size Dataframe of latent realizations for n samples and p latent variables.
    slope_column : str, default='slope'
        column name for slope in `sampled_latents` dataframe
    intercept_column : str, default='intercept'
        column name for intercept in `sampled_latents` dataframe
    amplitude_column : str, default='amplitude'
        column name for amplitude in `sampled_latents` dataframe
    frequency_column : str, default='frequency'
        column name for frequency in `sampled_latents` dataframe
    phaseshift_column : str, default='phaseshift'
        column name for phaseshift in `sampled_latents` dataframe
    noise_sigma : float, default=0.5
        scale of additive Gaussian noise

    Examples
    --------
    This code generates two samples and 4 timepoints, with the given TSN parameters given by a latent dataframe

    >>> import numpy as np
    >>> from parcs.simulators.temporal.deterministic import TSN
    >>> np.random.seed(1)
    >>> latents = pd.DataFrame(
    ...     [
    ...         [1, 0, 10, 1, 0],
    ...         [1.2, 1, 10, 2, 0]
    ...     ], columns=('slope', 'intercept', 'amplitude', 'frequency', 'phaseshift')
    ... )
    >>> tsn = TSN(sampled_latents=latents)
    >>> np.round(tsn.sample(seq_len=4), 2)
    array([[ 0.81,  9.11, 10.83,  3.87],
           [ 1.43, 10.14, -3.3 ,  1.43]])
    """

    def __init__(self,
                 sampled_latents: pd.DataFrame = None,
                 slope_column: str = 'slope',
                 intercept_column: str = 'intercept',
                 amplitude_column: str = 'amplitude',
                 frequency_column: str = 'frequency',
                 phaseshift_column: str = 'phaseshift',
                 noise_sigma: float = 0.5):
        self.latent = sampled_latents
        self.n_sample = len(sampled_latents)
        self.theta = self.latent[slope_column].values.reshape(-1, 1)
        self.b = self.latent[intercept_column].values.reshape(-1, 1)
        self.a = self.latent[amplitude_column].values.reshape(-1, 1)
        self.w = self.latent[frequency_column].values.reshape(-1, 1)
        self.phi = self.latent[phaseshift_column].values.reshape(-1, 1)
        self.noise_sigma = noise_sigma

    def sample(self, seq_len: int = 10):
        """
        sampled TSN

        Parameters
        ----------
        seq_len : int, default=10

        Returns
        -------
        D : array-like with shape `(n,t)`
            matrix where rows are samples and columns are time points.
        """
        t = np.arange(seq_len)
        return (self.theta * t + self.b) + \
               (self.a * np.sin(self.w * t + self.phi)) + \
               np.random.normal(0, self.noise_sigma, size=(self.n_sample, seq_len))
