import numpy as np


def get_edge_params():
    return {
        'sigmoid': {
            'alpha': [1, 6],
            'beta': [-0.8, 0.8],
            'gamma': {0, 1},
            'tau': {1, 3, 5},
            'rho': [0.05, 0.2]
        },
        'gaussian_rbf': {
            'alpha': [1, 6],
            'beta': [-0.8, 0.8],
            'gamma': {0, 1},
            'tau': {2, 4, 6},
            'rho': [0.05, 0.2]
        },
        'identity': {},
        'beta_noise': {
            'rho': [0.05, 0.3]
        }
    }


def get_state_params():
    return {
        'linear': {
            'coefs': [0, 3]
        },
        'poly1_interactions': {
            'coefs': [0, 3]
        }
    }


def get_output_params():
    return {
        'gaussian_noise': {
            'rho': [0.01, 0.5]
        },
        'gamma_noise': {
            'rho': [0.01, 0.5]
        },
        'bernoulli': {
            'rho': [0.01, 0.07],
            'gamma': {0, 1},
            'mean_': [0.1, 0.9]
        },
        'multinomial': {
            'rho': [0.01, 0.07],
            'gamma': {0, 1},
            'centers': {tuple(np.random.uniform(size=np.random.randint(low=3, high=9))) for _ in range(10)}
        }
    }


def get_edge_functions():
    return {
        'binary_input': ('identity', 'beta_noise'),
        'continuous_input': ('sigmoid', 'gaussian_rbf')
    }


def get_node_functions():
    return {
        'state': ('linear', 'poly1_interactions'),
        'output': {
            'binary': ('bernoulli', ),
            'continuous': ('gaussian_noise', 'gamma_noise'),
            'categorical': ('multinomial', )
        }
    }


def get_node_dtypes():
    return ['binary', 'categorical', 'continuous']
