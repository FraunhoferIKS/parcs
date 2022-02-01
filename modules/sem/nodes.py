import pandas as pd
from modules.sem import mapping_functions
import numpy as np


class Node:
    def __init__(self,
                 name=None,
                 node_type=None,
                 parents=None):
        # basic attributes
        self.node_type = node_type
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

        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'gaussian_noise': mapping_functions.output_gaussian_noise,
                'gamma_noise': mapping_functions.output_gamma_noise,
                'bernoulli': mapping_functions.output_binary,
                'multinomial': mapping_functions.output_multinomial
            }
        }

    def get_configs(self):
        return {
            'name': self.name,
            'node_type': self.node_type,
            'state_function': self.state_function,
            'output_function': self.output_function
        }

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


if __name__ == '__main__':
    # check continuous node
    node = Node(name='x0', node_type='continuous', parents=['x1', 'x2'])
    node.set_state_function(function_name='linear')
    node.set_output_function(function_name='gaussian_noise')

    node.set_state_params(params={'coefs': np.array([1, 2])})
    node.set_output_params(params={'rho': 0.02})

    node.calc_state(inputs=pd.DataFrame([[1, 2], [2, 1], [1, 1], [0, 0]], columns=('x1', 'x2')))
    print(node.value['state'])
    node.calc_output()
    print(node.value['output'])
