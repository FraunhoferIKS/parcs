import numpy as np
import pandas as pd
from itertools import product
from modules.sem import mapping_functions
from modules.sem.utils import topological_sort


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

        self.allowed_functions = {
            'continuous': ('gaussian_noise', 'gamma_noise'),
            'binary': ('bernoulli', ),
            'categorical': ('multinomial', )
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
        assert function_name in self.allowed_functions[self.node_type],\
            '{} doesn\'t match with {} node output type'.format(function_name, self.node_type)
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
        self.allowed_input_functions = {
            'continuous': ('sigmoid', 'gaussian_rbf'),
            'binary': ('identity', 'beta_noise')
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

    def set_function(self, function_name=None):
        assert function_name in self.allowed_input_functions[self.edge_input_type],\
            '{} doesn\'t match with {} edge type'.format(function_name, self.edge_input_type)
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


class BaseGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adj_matrix = pd.DataFrame([])
        self.data = {}

    def set_nodes(self, nodes_list=None):
        for item in nodes_list:
            node = Node(
                name=item['name'],
                node_type=item['output_type'],
                parents=item['parents']
            )
            node.set_state_function(
                function_name=item['state_function_name']
            ).set_output_function(
                function_name=item['output_function_name']
            ).set_state_params(
                params=item['state_params']
            ).set_output_params(
                params=item['output_params']
            )
            self.nodes[item['name']] = node
        return self

    def set_edges(self, info_matrix: pd.DataFrame = None):
        self.adj_matrix = (info_matrix != 0).astype(int)
        for node_pair in product(info_matrix.index, info_matrix.columns):
            try:
                info = info_matrix.loc[node_pair[0], node_pair[1]]
                # TODO: this must be implemented better
                assert info != 0
                parent_node_type = self.nodes[node_pair[0]].get_configs()['node_type']

                edge = Edge(parent=node_pair[0], child=node_pair[1], edge_input_type=parent_node_type)
                edge.set_function(
                    function_name=info['function_name']
                ).set_function_params(
                    params=info['function_params']
                )
                self.edges['{}->{}'.format(node_pair[0], node_pair[1])] = edge
            except AssertionError:
                continue

    def sample(self, size=None):
        # TODO: add a "check-all" step for all the info if they match
        assert size is not None, 'Specify size for sample'
        for node in topological_sort(adj_matrix=self.adj_matrix):
            v = self.nodes[node]
            inputs = pd.DataFrame({
                p: self.edges['{}->{}'.format(p, v.name)].map(
                    array=self.nodes[p].value['output']
                ) for p in v.parents
            })
            v.calc_state(inputs=inputs, size=size)
            v.calc_output()
        self.data = pd.DataFrame({v: self.nodes[v].value['output'] for v in self.nodes})
        return self.data


if __name__ == '__main__':
    # =========== EDGE check ===========
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

    # =========== NODE check ===========
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

    # =========== GRAPH check ===========
    graph = BaseGraph()
    graph.set_nodes(nodes_list=[
        {
            'name': 'A1',
            'parents': [],
            'output_type': 'continuous',
            'state_function_name': 'linear',
            'output_function_name': 'gaussian_noise',
            'state_params': {},
            'output_params': {'rho': 0.02}
        },
        {
            'name': 'A2',
            'parents': [],
            'output_type': 'continuous',
            'state_function_name': 'linear',
            'output_function_name': 'gaussian_noise',
            'state_params': {},
            'output_params': {'rho': 0.02}
        },
        {
            'name': 'B1',
            'parents': ['A1', 'A2'],
            'output_type': 'binary',
            'state_function_name': 'linear',
            'output_function_name': 'bernoulli',
            'state_params': {'coefs': np.array([1, 1])},
            'output_params': {'rho': 0.02, 'gamma': 0}
        }
    ])
    edge_param = {'alpha': 1, 'beta': 0, 'gamma': 0, 'tau': 1, 'rho': 0.02}
    graph.set_edges(info_matrix=pd.DataFrame(
        [
            [0, 0, {'function_name': 'sigmoid', 'function_params': edge_param}],
            [0, 0, {'function_name': 'sigmoid', 'function_params': edge_param}],
            [0, 0, 0]
        ],
        columns=['A1', 'A2', 'B1'],
        index=['A1', 'A2', 'B1']
    ))
    data = graph.sample(size=3000)
    print(data)
