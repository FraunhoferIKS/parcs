import numpy as np


def function_lookup(function_name):
    table = {
        'gaussian_0': gaussian_0,
        'gaussian_1': gaussian_1
    }
    return table[function_name]


def function_params_lookup(function_name):
    table = {
        'gaussian_0': [
            {'name': 'noise_sigma', 'range': [0.1, 0.9]}
        ],
        'gaussian_1': [
            {'name': 'noise_sigma', 'range': [0.1, 0.2]}
        ]
    }
    return table[function_name]


def gaussian_0(noise_sigma=None):
    return np.random.normal(0, noise_sigma)


def gaussian_1(noise_sigma=None):
    return np.random.normal(1, noise_sigma)
