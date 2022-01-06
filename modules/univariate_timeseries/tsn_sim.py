import numpy as np


def tsn_sample(time_range=None,
               trend_function=None, trend_params=None,
               seasonality_function=None, seasonality_params=None,
               noise_function=None, noise_params=None):
    data = np.array([
        trend_function(t=t, **trend_params) +
        seasonality_function(t=t, **seasonality_params) +
        noise_function(noise_function, **noise_params)
        for t in time_range
    ])
    return data.transpose()


class TSN:
    def __init__(self):
        pass

    def sample(self, size=None, time_range=None):
        pass



