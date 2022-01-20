import pandas as pd

from modules.sem.utils import exp_prob
from modules.sem import mapping_functions
import numpy as np
from scipy.special import comb


class Node:
    def __init__(self,
                 name=None,
                 parents=None):
        # basic attributes
        self.name = name
        self.parents = parents

        # selected activation functions attributes
        self.state_function = {
            'name': None,
            'function': mapping_functions.state_empty,
            'params': {}
        }
        self.output_function = {
            'name': None,
            'function': mapping_functions.output_empty,
            'params': {}
        }
        # helper
        self.functions = {'state': self.state_function, 'output': self.output_function}

        # value attributes
        self.value = {
            'state': np.array([]),
            'output': np.array([])
        }

        # options lists
        self.function_list = {
            'state': {},
            'output': {}
        }
        self.make_function_list()

    def get_configs(self):
        return {
            'name': self.name,
            'state_function': self.state_function,
            'output_function': self.output_function
        }

    def get_function_options(self, function_type=None):
        return []

    def get_param_options(self, function_type=None, function_name=None):
        return {}

    def make_function_list(self):
        pass

    @staticmethod
    def _get_param_size(num_parents=None, function_name=None, function_type=None):
        if function_type == 'output':
            return None
        else:
            if function_name == 'linear':
                return num_parents
            elif function_name == 'poly1_interactions':
                # interactions: True, Bias: True
                return 1 + num_parents + comb(num_parents, 2, exact=True)
            else:
                raise ValueError('Function {} not implemented'.format(function_name))

    def _set_function(self, function_type=None, function_name=None):
        self.functions[function_type]['name'] = function_name
        self.functions[function_type]['function'] = \
            self.function_list[function_type][function_name]
        return self

    def set_state_function(self, function_name=None):
        return self._set_function(function_type='state', function_name=function_name)

    def set_output_function(self, function_name=None):
        return self._set_function(function_type='output', function_name=function_name)

    def _set_function_params(self, function_type=None, params=None):
        self.functions[function_type]['params'] = params
        return self

    def set_state_params(self, params=None):
        return self._set_function_params(function_type='state', params=params)

    def set_output_params(self, params=None):
        return self._set_function_params(function_type='output', params=params)

    def calc_state(self, inputs=None, size=None):
        try:
            assert len(inputs) != 0
            result = self.state_function['function'](
                inputs=inputs,
                parents_order=self.parents,
                **self.state_function['params']
            )
            self.value['state'] = result
        except AssertionError:
            self.value['state'] = np.random.normal(size=size)
        return self.value['state']

    def calc_output(self):
        self.value['output'] = self.output_function['function'](
            array=self.value['state'],
            **self.output_function['params']
        )
        return self.value['output']


class ContinuousNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_function_options(self, function_type=None):
        options = {
            'state': ('linear', 'poly1_interactions'),
            'output': ('gaussian_noise', 'gamma_noise')
        }
        return options[function_type]

    def get_param_options(self, function_type=None, function_name=None):
        options = {
            'state': {
                'linear': {
                    'coefs': [0, 3]
                },
                'poly1_interactions': {
                    'coefs': [0, 3]
                }
            },
            'output': {
                'gaussian_noise': {
                    'rho': [0.01, 0.5]
                },
                'gamma_noise': {
                    'rho': [0.01, 0.5]
                }
            }
        }
        return options[function_type][function_name]

    def make_function_list(self):
        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'gaussian_noise': mapping_functions.output_gaussian_noise,
                'gamma_noise': mapping_functions.output_gamma_noise
            }
        }
        return None


class BinaryNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_function_options(self, function_type=None):
        options = {
            'state': ('linear', 'poly1_interactions'),
            'output': ('bernoulli', )
        }
        return options[function_type]

    def get_param_options(self, function_type=None, function_name=None):
        options = {
            'state': {
                'linear': {
                    'coefs': [0, 3]
                },
                'poly1_interactions': {
                    'coefs': [0, 3]
                }
            },
            'output': {
                'bernoulli': {
                    'rho': [0.01, 0.07],
                    'gamma': {0, 1},
                    'mean_': [0.1, 0.9]
                }
            }
        }
        return options[function_type][function_name]

    def make_function_list(self):
        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'bernoulli': mapping_functions.output_binary
            }
        }
        return None


class CategoricalNode(Node):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_function_options(self, function_type=None):
        options = {
            'state': ('linear', 'poly1_interactions'),
            'output': ('multinomial', )
        }
        return options[function_type]

    def get_param_options(self, function_type=None, function_name=None):
        options = {
            'state': {
                'linear': {
                    'coefs': [0, 3]
                },
                'poly1_interactions': {
                    'coefs': [0, 3]
                }
            },
            'output': {
                'multinomial': {
                    'rho': [0.01, 0.07],
                    'gamma': {0, 1},
                    'centers': {tuple(np.random.uniform(size=np.random.randint(low=3, high=9))) for _ in range(10)}
                }
            }
        }
        return options[function_type][function_name]

    def make_function_list(self):
        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'multinomial': mapping_functions.output_multinomial
            }
        }
        return None


if __name__ == '__main__':
    # check continuous node
    node = ContinuousNode(name='x0', parents=['x1', 'x2'])
    node.set_state_function(function_name='linear')
    node.set_output_function(function_name='gaussian_noise')

    node.set_state_params(params={'coefs': np.array([1, 2])})
    node.set_output_params(params={'rho': 0.02})

    node.calc_state(inputs=pd.DataFrame([[1, 2], [2, 1], [1, 1], [0, 0]], columns=('x1', 'x2')))
    print(node.value['state'])
    node.calc_output()
    print(node.value['output'])
