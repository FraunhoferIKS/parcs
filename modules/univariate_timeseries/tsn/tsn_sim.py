import numpy as np
import pandas as pd


class TSNCore:
    def __init__(self, composition=None,
                 trend_function=None, trend_params=None,
                 seasonality_function=None, seasonality_params=None,
                 noise_function=None, noise_params=None):
        self.trend_function = trend_function
        self.trend_params = trend_params

        self.seasonality_function = seasonality_function
        self.seasonality_params = seasonality_params

        self.noise_function = noise_function
        self.noise_params = noise_params

        self.composition = composition

    def _sample_additive(self, time_range=None):
        data = np.array([
            self.trend_function(t=t, **self.trend_params) +
            self.seasonality_function(t=t, **self.seasonality_params) +
            self.noise_function(**self.noise_params)
            for t in time_range
        ])
        return data.transpose()

    def _sample_multiplicative(self, time_range=None):
        data = np.array([
            self.trend_function(t=t, **self.trend_params) *
            self.seasonality_function(t=t, **self.seasonality_params) *
            self.noise_function(**self.noise_params)
            for t in time_range
        ])
        return data.transpose()

    def sample(self, time_range=None):
        if self.composition == 'additive':
            return self._sample_additive(time_range=time_range)
        elif self.composition == 'multiplicative':
            return self._sample_multiplicative(time_range=time_range)
        else:
            raise (ValueError, 'only additive and multiplicative compositions')


if __name__ == '__main__':
    def trend_func(a=None, b=None, t=None):
        return a*t + b

    def seasonality_func(w=None, phi=None, t=None):
        return np.sin(w*t + phi)

    def noise_func_1(sigma=None):
        return 1

    def noise_func_0(sigma=None):
        return 0

    sim = TSNCore(
        composition='multiplicative',
        trend_function=trend_func,
        seasonality_function=seasonality_func,
        noise_function=noise_func_1,
        trend_params={'a': np.array([1, 10]), 'b': np.array([0, 1])},
        seasonality_params={'w': np.array([0, 1]), 'phi': np.array([0, 0])},
        noise_params={'sigma': 0}
    )
    print(sim.sample(time_range=range(4)))


