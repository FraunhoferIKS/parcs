import numpy as np
import pandas as pd
from modules.simulators.temporal.utils import (
    seasonality_functions
)
from modules.simulators.temporal.utils import noise_functions, trend_functions
from modules.sem.structures import SimpleStructure


class TSNCore:
    def __init__(self, composition=None, trend_function=None,
                 seasonality_function=None, noise_function=None,):
        self.trend_function = trend_function
        self.seasonality_function = seasonality_function
        self.noise_function = noise_function

        self.composition = composition

    def _calculate_additive(self, time_range=None, param_samples=None):
        data = np.array([
            self.trend_function(t=t, **param_samples['trend']) +
            self.seasonality_function(t=t, **param_samples['seasonality']) +
            self.noise_function(**param_samples['noise'])
            for t in time_range
        ])
        return data.transpose()

    def _calculate_multiplicative(self, time_range=None, param_samples=None):
        data = np.array([
            self.trend_function(t=t, **param_samples['trend']) *
            self.seasonality_function(t=t, **param_samples['seasonality']) *
            self.noise_function(**param_samples['noise'])
            for t in time_range
        ])
        return data.transpose()

    def calculate(self, time_range=None, param_samples=None):
        if self.composition == 'additive':
            return self._calculate_additive(time_range=time_range, param_samples=param_samples)
        elif self.composition == 'multiplicative':
            return self._calculate_multiplicative(time_range=time_range, param_samples=param_samples)
        else:
            raise (ValueError, 'only additive and multiplicative compositions')


class TSN:
    def __init__(self,
                 trend_function=None, seasonality_function=None, noise_function=None,
                 composition=None):
        self.composition = composition

        self.param_samples = None
        self.timeseries_samples = None

        self.trend_function = trend_functions.function_lookup(trend_function)
        self.trend_function_params = trend_functions.function_params_lookup(trend_function)

        self.seasonality_function = seasonality_functions.function_lookup(seasonality_function)
        self.seasonality_function_params = seasonality_functions.function_params_lookup(seasonality_function)

        self.noise_function = noise_functions.function_lookup(noise_function)
        self.noise_function_params = noise_functions.function_params_lookup(noise_function)

        # parameters to simulate
        params = []
        params += [
            param['name'] for param in self.trend_function_params
        ]
        params += [
            param['name'] for param in self.seasonality_function_params
        ]
        params += [
            param['name'] for param in self.noise_function_params
        ]
        num_params = len(params)

        # param constraints
        t_param_constraints = {
            param['name']: param['range']
            for param in self.trend_function_params
        }
        s_param_constraints = {
            param['name']: param['range']
            for param in self.seasonality_function_params
        }
        n_param_constraints = {
            param['name']: param['range']
            for param in self.noise_function_params
        }
        self.param_constraints = {**t_param_constraints, **s_param_constraints, **n_param_constraints}

        self.sem_sim = SimpleStructure(
            adj_matrix=pd.DataFrame(
                np.zeros(shape=(num_params, num_params)),
                index=params,
                columns=params
            ),
            node_types={param: 'continuous' for param in params}
        )
        self.core = TSNCore(
            trend_function=self.trend_function,
            seasonality_function=self.seasonality_function,
            noise_function=self.noise_function,
            composition=self.composition
        )

    @staticmethod
    def _recale_param(values=None, ranges=None):
        [min_f, max_f] = ranges
        [min_i, max_i] = [values.min(), values.max()]
        # rescale-locate
        values = (values - min_i)/(max_i - min_i)
        values *= (max_f - min_f)
        values += min_f
        return values

    def sample(self, size=None, time_range=None):
        # sample first
        self.param_samples = self.sem_sim.sample(size=size)
        # re-scale
        for c in self.param_samples.columns:
            self.param_samples[c] = self._recale_param(
                values=self.param_samples[c].values,
                ranges=self.param_constraints[c]
            )

        # prep param samples
        param_samples = {
            'trend': {
                name: self.param_samples[name].values
                for name in self.param_samples if name.split('_')[0] == 'trend'
            },
            'seasonality': {
                name: self.param_samples[name].values
                for name in self.param_samples if name.split('_')[0] == 'seasonality'
            },
            'noise': {
                name: self.param_samples[name].values
                for name in self.param_samples if name.split('_')[0] == 'noise'
            }
        }
        self.timeseries_samples = self.core.calculate(
            time_range=time_range,
            param_samples=param_samples
        )
        return self.timeseries_samples


if __name__ == '__main__':
    tsn = TSN(
        trend_function='linear',
        seasonality_function='sine',
        noise_function='gaussian_1',
        composition='multiplicative'
    )
    from numpy import linspace
    ts_samples = tsn.sample(size=3, time_range=linspace(0, 10, 50))
    from matplotlib import pyplot as plt
    for ts in ts_samples:
        plt.plot(linspace(0, 10, 50), ts)
    plt.show()
