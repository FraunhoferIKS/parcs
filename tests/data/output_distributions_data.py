from scipy import stats as dists
import numpy as np
from numpy import array


class BaseClassTestData:
    # tests DistParams
    # inputs: param name, input data, coef
    dist_param = [
        ('p_1',
         array([[2], [1]]),
         {'bias': 1, 'linear': [-1], 'interactions': [2]}),
        ('p_2',
         array([[2, 1],
                [1, 1]]),
         {'bias': 0, 'linear': [1, 1], 'interactions': [0, 0, 2]}),
        ('p_1',
         array([]),
         {'bias': 2.7, 'linear': [], 'interactions': []})
    ]

    # tests PARCSDistributions
    # inputs: icdf, params, coefs, errors, data
    parcs_dist = [
        (dists.bernoulli.ppf,
         ['p'],
         {'p': {'bias': 1, 'linear': [], 'interactions': []}},
         [0.2, 0.3],
         array([])),
        (dists.norm.ppf,
         ['loc', 'scale'],
         {'loc': {'bias': 0, 'linear': [1, 0], 'interactions': [2, 0, 0]},
          'scale': {'bias': 1, 'linear': [0, 0], 'interactions': [0, 0, 0]}},
         [0.2, 0.3, 0.5],
         array([[1, 2], [2, 4], [1, 5]])),
    ]


class DistributionsData:
    # tests bernoulli
    # inputs: coefficients, errors, input data
    bernoulli_data = [
        ({'p_': {'bias': 0.2, 'linear': [], 'interactions': []}},
         [0.2, 0.3],
         array([])),
        ({'p_': {'bias': 0, 'linear': [2], 'interactions': [0]}},
         [0.1, 0.9],
         array([[0.1], [0.3]])),
        ({'p_': {'bias': 0, 'linear': [1, 2], 'interactions': [0.2, 0.1, 0.4]}},
         [0.3, 0.1],
         array([[0.1, 0.05], [0.03, 0.02]]))
    ]

    # tests bernoulli with correction
    # inputs: coefs, errors, data, correction configs
    bernoulli_correction_data = [
        ({'p_': {'bias': 0.2, 'linear': [], 'interactions': []}},
         [0.2, 0.8],
         array([]),
         {'lower': 0, 'upper': 1, 'target_mean': None}),
        ({'p_': {'bias': 2, 'linear': [1, 3], 'interactions': [-1, 2, 2.5]}},
         np.linspace(0, 1, 9),
         np.random.uniform(size=(9, 2)),
         {'lower': 0, 'upper': 1, 'target_mean': None})
    ]

    # tests uniform
    # inputs: coefs, errors, data
    uniform_data = [
        ({'mu_': {'bias': 0, 'linear': [], 'interactions': []},
          'diff_': {'bias': 1, 'linear': [], 'interactions': []}},
         [0.2, 0.3],
         array([])),
        ({'mu_': {'bias': 0, 'linear': [1, -0.5], 'interactions': [1, 0, 2]},
          'diff_': {'bias': 3, 'linear': [0, 0], 'interactions': [-1, -2, 0]}},
         [0.01],
         array([[1, 4]])),
    ]

    # tests normal
    # inputs: coefs, errors, data
    normal_data = [
        ({'mu_': {'bias': 0, 'linear': [], 'interactions': []},
          'sigma_': {'bias': 1, 'linear': [], 'interactions': []}},
         [0.2, 0.3],
         array([])),
        ({'mu_': {'bias': 0, 'linear': [1, -0.5], 'interactions': [1, 0, 2]},
          'sigma_': {'bias': 3, 'linear': [0, 0], 'interactions': [1, 2, 0]}},
         [0.01],
         array([[1, 4]]))
    ]

    # tests exponential
    # inputs: coefs, errors, data
    exponential_data = [
        ({'lambda_': {'bias': 0.2, 'linear': [], 'interactions': []}},
         [0.2, 0.3],
         array([])),
        ({'lambda_': {'bias': 0, 'linear': [2], 'interactions': [0]}},
         [0.1, 0.9],
         array([[0.1], [0.3]])),
        ({'lambda_': {'bias': 0, 'linear': [1, 2], 'interactions': [3, 0.1, 4]}},
         [0.3, 0.1],
         array([[0.1, 0.5], [0.3, 2]]))
    ]

    # tests poisson
    # inputs: coefs, errors, data
    poisson_data = [
        ({'lambda_': {'bias': 0.2, 'linear': [], 'interactions': []}},
         [0.2, 0.3],
         array([])),
        ({'lambda_': {'bias': 0, 'linear': [2], 'interactions': [0]}},
         [0.1, 0.9],
         array([[0.1], [0.3]])),
        ({'lambda_': {'bias': 0, 'linear': [1, 2], 'interactions': [3, 0.1, 4]}},
         [0.3, 0.1],
         array([[0.1, 0.5], [0.3, 2]]))
    ]

    # tests lognormal
    # inputs: coefs, errors, data
    lognormal_data = [
        ({'mu_': {'bias': 0, 'linear': [], 'interactions': []},
          'sigma_': {'bias': 1, 'linear': [], 'interactions': []}},
         [0.2, 0.7],
         array([])),
        ({'mu_': {'bias': 0, 'linear': [1, -0.5], 'interactions': [1, 0, 2]},
          'sigma_': {'bias': 3, 'linear': [0, 0], 'interactions': [1, 2, 0]}},
         [0.1],
         array([[1, 4]]))
    ]