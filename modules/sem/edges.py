from modules.sem import mapping_functions
import numpy as np


class Edge:
    def __init__(self,
                 parent: str = 'dummy',
                 child: str = 'dummy',
                 edge_input_type: str = 'dummy'):
        self.parent = parent
        self.child = child
        self.edge_input_type = edge_input_type

        self.edge_function = {
            'name': None,
            'function': mapping_functions.edge_empty,
            'params': {}
        }

        # options list attributes
        self.function_list = {
            'identity': mapping_functions.edge_binary_identity,
            'beta_noise': mapping_functions.edge_binary_beta,
            'sigmoid': mapping_functions.edge_sigmoid,
            'gaussian_rbf': mapping_functions.edge_gaussian_rbf
        }

        # output values
        self.value = np.array([])

    def get_configs(self):
        return {
            'parent': self.parent,
            'child': self.child,
            'edge_input_type': self.edge_input_type,
            'edge_function': self.edge_function
        }

    def _make_function_list(self):
        pass

    def set_function(self, function_name=None):
        self.edge_function = {
            'name': function_name,
            'function': self.function_list[function_name]
        }
        return self

    def set_function_params(self, params=None):
        self.edge_function['params'] = params
        return self

    def map(self, array=None):
        self.value = self.edge_function['function'](
            array=array,
            **self.edge_function['params']
        )
        return self.value


if __name__ == '__main__':
    # check binary input edge
    edge = Edge(parent='a', child='b', edge_input_type='binary')
    edge.set_function(function_name='beta_noise').set_function_params(params={'rho': 0.2})
    output = edge.map(array=np.array([1, 0, 0, 0, 1, 0]))
    print(np.round(output, 3))

    # check continuous input edge
    edge = Edge(parent='a', child='b', edge_input_type='continuous')
    edge.set_function(function_name='sigmoid').set_function_params(
        params={'alpha': 2, 'beta': -0.3, 'gamma': 0, 'tau': 1, 'rho': 0.07}
    )
    output = edge.map(array=np.array([1.8, 2, -1, 0.02, 1, 0]))
    print(np.round(output, 3))
