from modules.sem.utils import exp_prob
from modules.sem import mapping_functions
import numpy as np
from scipy.special import comb


class Node:
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None):
        # basic attributes
        self.name = name
        self.parents = parents
        self.complexity = complexity

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
        self.is_random = {
            'state_function': False,
            'state_params': False,
            'output_function': False,
            'output_params': False
        }

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
            'complexity': self.complexity,
            'state_function': self.state_function,
            'output_function': self.output_function,
            'is_random': self.is_random
        }

    def get_function_options(self, function_type=None):
        return []

    def get_param_options(self, function_type=None, function_name=None):
        return {}

    def make_function_list(self):
        pass

    def get_function_probs(self, function_type=None):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_function_options(function_type=function_type))
        )

    def _set_random_function(self, function_type=None):
        name = np.random.choice(
         self.get_function_options(function_type=function_type),
         p=self.get_function_probs(function_type=function_type)
        )
        self.functions[function_type]['name'] = name
        self.functions[function_type]['function'] = self.function_list[function_type][name]

        self.is_random['{}_function'.format(function_type)] = True
        return self

    def set_random_state_function(self):
        return self._set_random_function(function_type='state')

    def set_random_output_function(self):
        return self._set_random_function(function_type='output')

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

    def _set_random_params(self, function_type=None):
        param_options = self.get_param_options(
            function_type=function_type,
            function_name=self.functions[function_type]['name']
        )
        # for state functions, we need an array of random values
        size_ = self._get_param_size(
            num_parents=len(self.parents),
            function_name=self.functions[function_type]['name'],
            function_type=function_type
        )
        for param in param_options:
            options = param_options[param]
            if isinstance(options, list):
                param_value = np.random.uniform(
                    low=options[0],
                    high=options[1],
                    size=size_
                )
            elif isinstance(options, set):
                param_value = np.random.choice(
                    list(options),
                    size=size_
                )
            else:
                raise TypeError
            self.functions[function_type]['params'][param] = param_value

        self.is_random['{}_params'.format(function_type)] = True
        return self

    def set_random_state_params(self):
        return self._set_random_params(function_type='state')

    def set_random_output_params(self):
        return self._set_random_params(function_type='output')

    def _set_function(self, function_type=None, function_name=None):
        self.functions[function_type]['name'] = function_name
        self.functions[function_type]['function'] = \
            self.function_list[function_type][function_name]
        self.is_random['{}_function'] = False
        return self

    def set_state_function(self, function_name=None):
        return self._set_function(function_type='state', function_name=function_name)

    def set_output_function(self, function_name=None):
        return self._set_function(function_type='output', function_name=function_name)

    def _set_function_params(self, function_type=None, params=None):
        self.functions[function_type]['params'] = params
        self.is_random['{}_params'] = False
        return self

    def set_state_params(self, params=None):
        return self._set_function_params(function_type='state', params=params)

    def set_output_params(self, params=None):
        return self._set_function_params(function_type='output', params=params)

    def random_initiate(self):
        self.set_random_state_function()
        self.set_random_output_function()
        self.set_random_state_params()
        self.set_random_output_params()
        return self

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
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None,
                 **kwargs):
        super().__init__(
            name=name,
            parents=parents,
            complexity=complexity
        )

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
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None,
                 **kwargs):
        super().__init__(
            name=name,
            parents=parents,
            complexity=complexity
        )

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
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None,
                 **kwargs):
        super().__init__(
            name=name,
            parents=parents,
            complexity=complexity
        )

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