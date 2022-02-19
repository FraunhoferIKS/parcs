import numpy as np
from matplotlib import pyplot as plt
import pandas as pd


class FourierSeries:
    def __init__(self,
                 sampled_latents: pd.DataFrame = None,
                 frequency_prefix: str = 'w',
                 phaseshift_prefix: str = 'phi',
                 amplitude_prefix: str = 'a'):
        self.latents = sampled_latents
        # get latent columns
        self.frequency_columns = sorted([
            i for i in sampled_latents.columns if i.split('_')[0] == frequency_prefix
        ])
        self.phaseshift_columns = sorted([
            i for i in sampled_latents.columns if i.split('_')[0] == phaseshift_prefix
        ])
        self.amplitude_columns = sorted([
            i for i in sampled_latents.columns if i.split('_')[0] == amplitude_prefix
        ])
        # latent values
        self.frequencies = self.latents[self.frequency_columns].values
        self.phaseshifts = self.latents[self.phaseshift_columns].values
        self.amplitudes = self.latents[self.amplitude_columns].values

    def sample(self, seq_len: int = 10):
        t = np.arange(seq_len)
        # reshape [n x w_i] -> [n x w_i x 1], because it will be outer product with time
        freqs = self.frequencies.reshape(*self.frequencies.shape, 1)
        amps = self.amplitudes.reshape(*self.amplitudes.shape, 1)
        phis = self.phaseshifts.reshape(*self.phaseshifts.shape, 1)
        # calculate bucket of sins for samples and reshape again to [ n x w x t]
        decomposed = (amps * np.sin(freqs * t + phis)).reshape(*self.frequencies.shape, -1)
        return decomposed.sum(axis=1)


class TSN:
    def __init__(self,
                 sampled_latents: pd.DataFrame = None,
                 slope_column: str = 'slope',
                 intercept_column: str = 'intercept',
                 amplitude_column: str = 'amplitude',
                 frequency_column: str = 'frequency',
                 phaseshift_column: str = 'phaseshift'):
        self.latent = sampled_latents
        self.theta = self.latent[slope_column].values.reshape(-1, 1)
        self.b = self.latent[intercept_column].values.reshape(-1, 1)
        self.a = self.latent[amplitude_column].values.reshape(-1, 1)
        self.w = self.latent[frequency_column].values.reshape(-1, 1)
        self.phi = self.latent[phaseshift_column].values.reshape(-1, 1)

    def sample(self, seq_len: int = 10):
        t = np.arange(seq_len)
        return (self.theta * t + self.b) + (self.a * np.sin(self.w*t+self.phi))


if __name__ == '__main__':
    # latents = pd.DataFrame([
    #     [np.pi / 10, np.pi / 20, np.pi/2, np.pi/2, 1, 1],
    #     [np.pi / 10, np.pi / 20, 0, 0, 2, 2],
    #     [np.pi / 10, np.pi / 20, 0, 0, 4, 4]
    # ], columns=('w_0', 'w_1', 'phi_0', 'phi_1', 'a_0', 'a_1'))
    # fs = FourierSeries(sampled_latents=latents)
    # for j in range(3):
    #     plt.plot(np.arange(200), fs.sample(seq_len=200)[j])
    # plt.show()
    latents = pd.DataFrame(
        [
            [1, 0, 10, 1, 0],
            [1.2, 1, 10, 2, 0]
        ], columns=('slope', 'intercept', 'amplitude', 'frequency', 'phaseshift')
    )
    tsn = TSN(sampled_latents=latents)
    tsn.sample(seq_len=200)
    plt.plot(np.arange(200), tsn.sample(seq_len=200)[0])
    plt.plot(np.arange(200), tsn.sample(seq_len=200)[1])
    plt.show()
