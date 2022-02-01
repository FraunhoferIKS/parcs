import numpy as np
import pandas as pd
from itertools import product
from matplotlib import pyplot as plt
from modules.sem.utils import topological_sort
from modules.sem.graph_objects import Node, Edge


class BaseStructure:
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
    structure = BaseStructure()
    structure.set_nodes(nodes_list=[
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
    structure.set_edges(info_matrix=pd.DataFrame(
        [
            [0, 0, {'function_name': 'sigmoid', 'function_params': edge_param}],
            [0, 0, {'function_name': 'sigmoid', 'function_params': edge_param}],
            [0, 0, 0]
        ],
        columns=['A1', 'A2', 'B1'],
        index=['A1', 'A2', 'B1']
    ))
    data = structure.sample(size=3000)
    print(data)

    plt.scatter(data.A1, data.A2, c=data.B1)
    plt.show()
