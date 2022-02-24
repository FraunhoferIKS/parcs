import numpy as np
import pandas as pd
from itertools import product
from modules.sem import mapping_functions
from modules.sem.utils import topological_sort


NODE_OUTPUT_TYPES = ['continuous', 'binary', 'categorical']
ALLOWED_EDGE_FUNCTIONS = {
    'continuous': ('sigmoid', 'gaussian_rbf'),
    'binary': ('identity', 'beta_noise'),
    'categorical': ('identity', )
}
ALLOWED_STATE_FUNCTIONS = ['linear', 'poly1_interactions']
ALLOWED_OUTPUT_FUNCTIONS = {
    'continuous': ('gaussian_noise', 'gamma_noise'),
    'binary': ('bernoulli',),
    'categorical': ('multinomial',)
}


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

        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'gaussian_noise': mapping_functions.output_gaussian_noise,
                'gamma_noise': mapping_functions.output_gamma_noise,
                'bernoulli': mapping_functions.output_bernoulli,
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


class Edge:
    def __init__(self,
                 parent: str = 'dummy',
                 child: str = 'dummy'):
        self.parent = parent
        self.child = child

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
            'edge_function': self.edge_function
        }

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


class BaseGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adj_matrix = pd.DataFrame([])
        self.data = {}

    def set_nodes(self, nodes_list=None):
        self.nodes = {
            item['name']: Node(
                name=item['name'], parents=item['parents']
            ).set_state_function(
                function_name=item['state_function']
            ).set_output_function(
                function_name=item['output_function']
            ).set_state_params(
                params=item['state_params']
            ).set_output_params(
                params=item['output_params']
            ) for item in nodes_list
        }
        return self

    def set_edges(self, adj_matrix: pd.DataFrame = None, function_specs: dict = None):
        self.adj_matrix = adj_matrix
        for node_pair in product(adj_matrix.index, adj_matrix.columns):
            try:
                info = adj_matrix.loc[node_pair[0], node_pair[1]]
                edge_symbol = '{} -> {}'.format(node_pair[0], node_pair[1])
                assert info != 0
                self.edges[edge_symbol] = Edge(
                    parent=node_pair[0], child=node_pair[1]
                ).set_function(
                    function_name=function_specs[edge_symbol]['function_name']
                ).set_function_params(
                    params=function_specs[edge_symbol]['params']
                )
            except AssertionError:
                continue
        return self

    def sample(self, size=None):
        # TODO: add a "check-all" step for all the info if they match
        assert size is not None, 'Specify size for sample'
        for node in topological_sort(adj_matrix=self.adj_matrix):
            print('node is: ', node)
            v = self.nodes[node]
            print('v is: ', v)
            inputs = pd.DataFrame({
                p: self.edges['{} -> {}'.format(p, v.name)].map(
                    array=self.nodes[p].value['output']
                ) for p in v.parents
            })
            print('inputs are: ', inputs)
            v.calc_state(inputs=inputs, size=size)
            print('state calculated')
            v.calc_output()
            print('output calculated')
            print('==========')
        self.data = pd.DataFrame({v: self.nodes[v].value['output'] for v in self.nodes})
        return self.data


if __name__ == '__main__':
    # =========== EDGE check ===========
    # check binary input edge
    edge = Edge(parent='a', child='b')
    edge.set_function(function_name='beta_noise').set_function_params(params={'rho': 0.2})
    output = edge.map(array=np.array([1, 0, 0, 0, 1, 0]))
    print(np.round(output, 3))

    # check continuous input edge
    edge = Edge(parent='a', child='b')
    edge.set_function(function_name='sigmoid').set_function_params(
        params={'alpha': 2, 'beta': -0.3, 'gamma': 0, 'tau': 1, 'rho': 0.07}
    )
    output = edge.map(array=np.array([1.8, 2, -1, 0.02, 1, 0]))
    print(np.round(output, 3))

    # =========== NODE check ===========
    # check continuous node
    node = Node(name='x0', parents=['x1', 'x2'])
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
            'state_function': 'linear',
            'output_function': 'gaussian_noise',
            'state_params': {},
            'output_params': {'rho': 0.02}
        },
        {
            'name': 'A2',
            'parents': [],
            'output_type': 'continuous',
            'state_function': 'linear',
            'output_function': 'gaussian_noise',
            'state_params': {},
            'output_params': {'rho': 0.02}
        },
        {
            'name': 'B1',
            'parents': ['A1', 'A2'],
            'output_type': 'binary',
            'state_function': 'linear',
            'output_function': 'bernoulli',
            'state_params': {'coefs': np.array([1, 1])},
            'output_params': {'rho': 0.02, 'gamma': 0}
        }
    ])
    edge_param = {'alpha': 1, 'beta': 0, 'gamma': 0, 'tau': 1, 'rho': 0.02}
    graph.set_edges(
        adj_matrix=pd.DataFrame(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 0, 0]
            ],
            columns=['A1', 'A2', 'B1'],
            index=['A1', 'A2', 'B1']
        ),
        function_specs={
            'A1 -> B1': {'function_name': 'sigmoid', 'function_params': edge_param},
            'A2 -> B1': {'function_name': 'sigmoid', 'function_params': edge_param}
        }
    )
    data = graph.sample(size=3000)
    print(data)