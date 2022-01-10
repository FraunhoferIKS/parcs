from modules.sem.utils import exp_prob
from modules.sem import mapping_functions
import numpy as np


class Edge:
    def __init__(self,
                 parent=None,
                 child=None,
                 complexity=None):
        self.parent = parent
        self.child = child
        if complexity is None:
            self.complexity = 0
        else:
            self.complexity = complexity

        self.edge_function = {
            'name': None,
            'function': None,
            'params': {}
        }
        self.is_random = {
            'function': False,
            'params': False
        }

        # options list attributes
        self.function_list = {}
        self.make_function_list()

        # output values
        self.value = np.array([])

    def get_configs(self):
        return {
            'parent': self.parent,
            'child': self.child,
            'complexity': self.complexity,
            'edge_function': self.edge_function,
            'is_random': self.is_random
        }

    def get_function_options(self):
        return []

    def get_param_options(self, function=None):
        return {}

    def make_function_list(self):
        pass

    def get_function_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_function_options())
        )

    def set_random_function(self):
        function_name = np.random.choice(
            self.get_function_options(),
            p=self.get_function_probs()
        )
        self.is_random['function'] = True
        self.edge_function['name'] = function_name
        self.edge_function['function'] = self.function_list[function_name]
        return self

    def set_random_function_params(self):
        param_options = self.get_param_options(
            function=self.edge_function['name']
        )
        for param in param_options:
            options = param_options[param]
            if isinstance(options, list):
                param_value = np.random.uniform(low=options[0], high=options[1])
            elif isinstance(options, set):
                param_value = np.random.choice(list(options))
            else:
                raise TypeError
            self.edge_function['params'][param] = param_value

        self.is_random['params'] = True
        return self

    def set_function(self, function_name=None):
        self.edge_function = {
            'name': function_name,
            'function': self.function_list[function_name]
        }
        self.is_random['function'] = False
        return self

    def set_function_params(self, params=None):
        self.edge_function['params'] = params
        self.is_random['params'] = False
        return self

    def random_initiate(self):
        self.set_random_function()
        self.set_random_function_params()
        return self

    def map(self, array=None):
        self.value = self.edge_function['function'](
            array=array,
            **self.edge_function['params']
        )
        return self.value


class BinaryInputEdge(Edge):
    def __init__(self,
                 parent=None,
                 child=None,
                 complexity=None):
        super().__init__(
            parent=parent,
            child=child,
            complexity=complexity
        )

    def get_function_options(self):
        return (
            'identity',
            'beta_noise'
        )

    def get_param_options(self, function=None):
        options = {
            'identity': {},
            'beta_noise': {
                'rho': [0.05, 0.3]
            }
        }
        return options[function]

    def make_function_list(self):
        self.function_list = {
            'identity': mapping_functions.edge_binary_identity,
            'beta_noise': mapping_functions.edge_binary_beta
        }


class ContinuousInputEdge(Edge):
    def __init__(self,
                 parent=None,
                 child=None,
                 complexity=None):
        super().__init__(
            parent=parent,
            child=child,
            complexity=complexity
        )

    def get_function_options(self):
        return (
            'sigmoid',
            'gaussian_rbf'
        )

    def get_param_options(self, function=None):
        options = {
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
            }
        }
        return options[function]

    def make_function_list(self):
        self.function_list = {
            'sigmoid': mapping_functions.edge_sigmoid,
            'gaussian_rbf': mapping_functions.edge_gaussian_rbf
        }