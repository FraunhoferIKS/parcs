import numpy as np
import pandas as pd
from itertools import product
from rad_sim.scm import mapping_functions
from rad_sim.scm.utils import topological_sort
from rad_sim.scm.output_distributions import GaussianDistribution, BernoulliDistribution


OUTPUT_DISTRIBUTIONS = {
    'gaussian': GaussianDistribution,
    'bernoulli': BernoulliDistribution
}

EDGE_FUNCTIONS = {
    'identity': mapping_functions.edge_binary_identity,
    'sigmoid': mapping_functions.edge_sigmoid,
    'gaussian_rbf': mapping_functions.edge_gaussian_rbf
}


class Node:
    def __init__(self,
                 name=None,
                 parents=None,
                 output_distribution=None,
                 dist_configs=None,
                 dist_params_coefs=None):
        # basic attributes
        self.info = {
            'name': name,
            'output_distribution': output_distribution,
            'parents': parents
        }
        self.output_distribution = OUTPUT_DISTRIBUTIONS[output_distribution](
            **dist_configs,
            coefs=dist_params_coefs
        )

    def sample(self, data, size):
        return self.output_distribution.calculate_output(
            data[self.info['parents']].values,
            size
        )


class Edge:
    def __init__(self,
                 parent=None,
                 child=None,
                 function_name=None,
                 function_params=None):
        self.parent = parent
        self.child = child

        self.edge_function = {
            'name': function_name,
            'function': EDGE_FUNCTIONS[function_name],
            'params': function_params
        }

    def map(self, array=None):
        return self.edge_function['function'](
            array=array,
            **self.edge_function['params']
        )


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
                    function_params=function_specs[edge_symbol]['function_params']
                )
            except AssertionError:
                continue
        return self

    def sample(self, size=None):
        assert size is not None, 'Specify size for sample'
        for node in topological_sort(adj_matrix=self.adj_matrix):
            v = self.nodes[node]
            inputs = pd.DataFrame({
                p: self.edges['{} -> {}'.format(p, v.name)].map(
                    array=self.nodes[p].value['output']
                ) for p in v.parents
            })
            v.calc_state(inputs=inputs, size=size)
            v.calc_output()
        self.data = pd.DataFrame({v: self.nodes[v].value['output'] for v in self.nodes})
        return self.data


# if __name__ == '__main__':
#     # =========== EDGE check ===========
#     # check binary input edge
#     edge = Edge(parent='a', child='b')
#     edge.set_function(function_name='beta_noise').set_function_params(params={'rho': 0.2})
#     output = edge.map(array=np.array([1, 0, 0, 0, 1, 0]))
#     print(np.round(output, 3))
#
#     # check continuous input edge
#     edge = Edge(parent='a', child='b')
#     edge.set_function(function_name='sigmoid').set_function_params(
#         params={'alpha': 2, 'beta': -0.3, 'gamma': 0, 'tau': 1, 'rho': 0.07}
#     )
#     output = edge.map(array=np.array([1.8, 2, -1, 0.02, 1, 0]))
#     print(np.round(output, 3))
#
#     # =========== NODE check ===========
#     # check continuous node
#     node = Node(name='x0', parents=['x1', 'x2'])
#     node.set_state_function(function_name='linear')
#     node.set_output_function(function_name='gaussian_noise')
#
#     node.set_state_params(params={'coefs': np.array([1, 2])})
#     node.set_output_params(params={'rho': 0.02})
#
#     node.calc_state(inputs=pd.DataFrame([[1, 2], [2, 1], [1, 1], [0, 0]], columns=('x1', 'x2')))
#     print(node.value['state'])
#     node.calc_output()
#     print(node.value['output'])
#
#     # =========== GRAPH check ===========
#     graph = BaseGraph()
#     graph.set_nodes(nodes_list=[
#         {
#             'name': 'A1',
#             'parents': [],
#             'output_type': 'continuous',
#             'state_function': 'linear',
#             'output_function': 'gaussian_noise',
#             'state_params': {},
#             'output_params': {'rho': 0.02}
#         },
#         {
#             'name': 'A2',
#             'parents': [],
#             'output_type': 'continuous',
#             'state_function': 'linear',
#             'output_function': 'gaussian_noise',
#             'state_params': {},
#             'output_params': {'rho': 0.02}
#         },
#         {
#             'name': 'B1',
#             'parents': ['A1', 'A2'],
#             'output_type': 'binary',
#             'state_function': 'linear',
#             'output_function': 'bernoulli',
#             'state_params': {'coefs': np.array([1, 1])},
#             'output_params': {'rho': 0.02, 'gamma': 0}
#         }
#     ])
#     edge_param = {'alpha': 1, 'beta': 0, 'gamma': 0, 'tau': 1, 'rho': 0.02}
#     graph.set_edges(
#         adj_matrix=pd.DataFrame(
#             [
#                 [0, 0, 1],
#                 [0, 0, 1],
#                 [0, 0, 0]
#             ],
#             columns=['A1', 'A2', 'B1'],
#             index=['A1', 'A2', 'B1']
#         ),
#         function_specs={
#             'A1 -> B1': {'function_name': 'sigmoid', 'function_params': edge_param},
#             'A2 -> B1': {'function_name': 'sigmoid', 'function_params': edge_param}
#         }
#     )
#     data = graph.sample(size=3000)
#     print(data)