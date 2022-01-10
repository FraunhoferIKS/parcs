import numpy as np


def function_lookup(function_name):
    table = {
        'sine': sine_function
    }
    return table[function_name]


def function_params_lookup(function_name):
    table = {
        'sine': [
            {'name': 'seasonality_w', 'range': [-np.pi/3, np.pi/3]},
            {'name': 'seasonality_phi', 'range': [-np.pi/2, np.pi/2]},
            {'name': 'seasonality_magnitude', 'range': [1, 5]}
        ]
    }
    return table[function_name]


def sine_function(seasonality_w=None, seasonality_phi=None, seasonality_magnitude=None, t=None):
    return seasonality_magnitude * np.sin(seasonality_w * t + seasonality_phi)
