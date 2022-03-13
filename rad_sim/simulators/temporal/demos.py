import numpy as np
import pandas as pd

from rad_sim.sem.basic import *
from rad_sim.simulators.temporal.deterministic import *


# = TSN
# == latent y
class TsnLatentLogisticYSimulator:
    def __init__(self,
                 tsn_latent_type='uniform',
                 tsn_latent_var_list=None,
                 tsn_noise_sigma=None,
                 y_beta_coef_range=None,
                 y_sigmoid_offset=None,
                 latent_subset_for_label=None):
        # latent simulator
        if tsn_latent_type == 'uniform':
            self.latent_simulator = IndependentUniformLatents()
        elif tsn_latent_type == 'normal':
            self.latent_simulator = IndependentNormalLatents()
        else:
            raise ValueError('latent type unknown')
        self.latent_simulator.set_nodes(var_list=tsn_latent_var_list)
        self.tsn_noise_sigma = tsn_noise_sigma

        # latent label maker
        self.label_maker = LatentLabelMaker(
            coef_min=y_beta_coef_range[0],
            coef_max=y_beta_coef_range[1],
            sigmoid_offset=y_sigmoid_offset
        )
        self.latent_subset_column = latent_subset_for_label

    def sample(self, sample_size=None, seq_len=None):
        # sample latents
        l = self.latent_simulator.sample(sample_size=sample_size)
        # initiate tsn
        tsn = TSN(sampled_latents=l, noise_sigma=self.tsn_noise_sigma)
        # run tsn
        signals = tsn.sample(seq_len=seq_len)

        # make labels
        labels = self.label_maker.make_label(sampled_latents=l[self.latent_subset_column])

        return signals, labels


class TsnLatent2distYSimulator:
    def __init__(self,
                 tsn_latent_type='normal',
                 dist_0_var_list=None,
                 dist_1_var_list=None,
                 tsn_noise_sigma=None,
                 class_ratio=0.5):
        if tsn_latent_type == 'uniform':
            self.l0_simulator = IndependentUniformLatents()
            self.l1_simulator = IndependentUniformLatents()
        elif tsn_latent_type == 'normal':
            self.l0_simulator = IndependentNormalLatents()
            self.l1_simulator = IndependentNormalLatents()
        else:
            raise ValueError('latent type unknown')
        self.l0_simulator.set_nodes(var_list=dist_0_var_list)
        self.l1_simulator.set_nodes(var_list=dist_1_var_list)
        self.tsn_noise_sigma = tsn_noise_sigma
        self.class_ratio = class_ratio

    def sample(self, sample_size=None, seq_len=None):
        n1 = int(sample_size*self.class_ratio)
        n0 = sample_size - n1
        # sample latents
        l0 = self.l0_simulator.sample(sample_size=n0)
        l1 = self.l1_simulator.sample(sample_size=n1)
        # run tsn
        tsn0 = TSN(sampled_latents=l0, noise_sigma=self.tsn_noise_sigma)
        tsn1 = TSN(sampled_latents=l1, noise_sigma=self.tsn_noise_sigma)
        signals = np.concatenate([tsn0.sample(seq_len=seq_len), tsn1.sample(seq_len=seq_len)], axis=0)
        # make labels
        labels = np.concatenate([np.zeros(shape=(n0,)), np.ones(shape=(n1,))]).astype(int)
        # shuffle
        idx = np.arange(signals.shape[0])
        np.random.shuffle(idx)
        signals = signals[idx]
        labels = labels[idx]
        return signals, labels


class TsnShapeletYSimulator:
    def __init__(self,
                 tsn_latent_type='uniform',
                 tsn_latent_var_list=None,
                 tsn_noise_sigma=None,
                 shapelet_window_ratio=0.3,
                 shapelet_num_sin=5,
                 shapelet_added_noise=None,
                 class_ratio=None):
        # latent simulator
        if tsn_latent_type == 'uniform':
            self.latent_simulator = IndependentUniformLatents()
        elif tsn_latent_type == 'normal':
            self.latent_simulator = IndependentNormalLatents()
        else:
            raise ValueError('latent type unknown')
        self.latent_simulator.set_nodes(var_list=tsn_latent_var_list)
        self.tsn_noise_sigma = tsn_noise_sigma

        self.label_maker = ShapeletPlacementLabelMaker(
            window_ratio=shapelet_window_ratio,
            class_ratio=class_ratio,
            shapelet_num_sin=shapelet_num_sin,
            shapelet_added_noise=shapelet_added_noise
        )

    def sample(self, sample_size=None, seq_len=None):
        # make tsn
        # sample latents
        l = self.latent_simulator.sample(sample_size=sample_size)
        # initiate tsn
        tsn = TSN(sampled_latents=l, noise_sigma=self.tsn_noise_sigma)
        raw_signal = tsn.sample(seq_len=seq_len)
        signals, labels = self.label_maker.make_label(signals=raw_signal)
        return signals, labels


class FsLogNormalLatent2distYSimulator:
    def __init__(self,
                 dist_0_frequency_mean=None, dist_1_frequency_mean=None,
                 dist_0_frequency_sigma=None, dist_1_frequency_sigma=None,
                 dist_0_next_frequency_ratio=2, dist_1_next_frequency_ratio=2,
                 fs_phi_latent_type='uniform', dist_0_phi_config=None, dist_1_phi_config=None,
                 dist_0_num_sin=None, dist_1_num_sin=None,
                 dist_0_dominant_amplitude=1, dist_1_dominant_amplitude=1,
                 dist_0_amplitude_exp_decay_rate=1, dist_1_amplitude_exp_decay_rate=1,
                 dist_0_added_noise_sigma_ratio=0.2, dist_1_added_noise_sigma_ratio=0.2,
                 class_ratio=0.5):
        self.fs0_config = {
            'dominant_amplitude': dist_0_dominant_amplitude,
            'added_noise_sigma_ratio': dist_0_added_noise_sigma_ratio,
            'amplitude_exp_decay_rate': dist_0_amplitude_exp_decay_rate
        }
        self.fs1_config = {
            'dominant_amplitude': dist_1_dominant_amplitude,
            'added_noise_sigma_ratio': dist_1_added_noise_sigma_ratio,
            'amplitude_exp_decay_rate': dist_1_amplitude_exp_decay_rate
        }
        self.class_ratio = class_ratio
        # frequencies
        self.l0_freq_simulator = FrequencyLogNormalLatents(
            num_freqs=dist_0_num_sin,
            first_freq_mean=dist_0_frequency_mean,
            next_freq_ratio=dist_0_next_frequency_ratio,
            sigma=dist_0_frequency_sigma
        )
        self.l1_freq_simulator = FrequencyLogNormalLatents(
            num_freqs=dist_1_num_sin,
            first_freq_mean=dist_1_frequency_mean,
            next_freq_ratio=dist_1_next_frequency_ratio,
            sigma=dist_1_frequency_sigma
        )
        # phis
        if fs_phi_latent_type == 'uniform':
            self.l0_phi_simulator = IndependentUniformLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'low': dist_0_phi_config[0], 'high': dist_0_phi_config[1]}
                for i in range(dist_0_num_sin)
            ])
            self.l1_phi_simulator = IndependentUniformLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'low': dist_1_phi_config[0], 'high': dist_1_phi_config[1]}
                for i in range(dist_1_num_sin)
            ])
        elif fs_phi_latent_type == 'normal':
            self.l0_phi_simulator = IndependentNormalLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'mean': dist_0_phi_config[0], 'sigma': dist_0_phi_config[1], 'log': False}
                for i in range(dist_0_num_sin)
            ])
            self.l1_phi_simulator = IndependentNormalLatents().set_nodes(var_list=[
                {'name': 'phi_{}'.format(i), 'mean': dist_1_phi_config[0], 'sigma': dist_1_phi_config[1], 'log': False}
                for i in range(dist_1_num_sin)
            ])
        else:
            raise ValueError('unknown latent type')

    def sample(self, sample_size=None, seq_len=None):
        n1 = int(sample_size * self.class_ratio)
        n0 = sample_size - n1
        l0_freqs = self.l0_freq_simulator.sample(sample_size=n0)
        l1_freqs = self.l1_freq_simulator.sample(sample_size=n1)
        l0_phis = self.l0_phi_simulator.sample(sample_size=n0)
        l1_phis = self.l1_phi_simulator.sample(sample_size=n1)

        fs0 = FourierSeries(
            sampled_latents=pd.concat([l0_freqs, l0_phis], axis=1),
            **self.fs0_config
        )
        fs1 = FourierSeries(
            sampled_latents=pd.concat([l1_freqs, l1_phis], axis=1),
            **self.fs1_config
        )
        signals = np.concatenate([fs0.sample(seq_len=seq_len), fs1.sample(seq_len=seq_len)], axis=0)
        # make labels
        labels = np.concatenate([np.zeros(shape=(n0,)), np.ones(shape=(n1,))]).astype(int)
        # shuffle
        idx = np.arange(signals.shape[0])
        np.random.shuffle(idx)
        signals = signals[idx]
        labels = labels[idx]
        return signals, labels



