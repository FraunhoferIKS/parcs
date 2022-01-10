from modules.sem.utils import is_acyclic, topological_sort
from modules.sem.graph_objects.nodes import *
from modules.sem.graph_objects.edges import *
from itertools import product
import pandas as pd
import numpy as np


class Structure:
    def __init__(self,
                 adj_matrix: pd.DataFrame = None,
                 node_types=None,
                 complexity=None):
        if complexity is None:
            complexity = 0
        self.nodes = {}
        self.edges = {}
        assert adj_matrix.columns.tolist() == adj_matrix.index.tolist()
        self.adj_matrix = adj_matrix
        self.node_dict = {
            'continuous': ContinuousNode,
            'binary': BinaryNode,
            'categorical': CategoricalNode
        }
        self.edge_dict = {
            'continuous': ContinuousInputEdge,
            'binary': BinaryInputEdge
        }
        self.complexity = complexity

        self.data = pd.DataFrame([], columns=adj_matrix.columns)

        self._set_nodes(node_types=node_types)._set_edges(node_types=node_types)._random_initiate()

    def _set_nodes(self, node_types=None):
        adjm = self.adj_matrix

        self.nodes = {
            c: self.node_dict[node_types[c]](
                name=c,
                parents=adjm[adjm[c] == 1].index.tolist(),
                complexity=self.complexity
            ) for c in adjm.columns
        }
        return self

    def _set_edges(self, node_types=None):
        adjm = self.adj_matrix
        self.edges = {
            '{}->{}'.format(c[0], c[1]): self.edge_dict[node_types[c[0]]](
                parent=c[0],
                child=c[1],
                complexity=self.complexity
            ) for c in product(adjm.index, adjm.columns)
            if adjm.loc[c[0], c[1]] == 1
        }
        return self

    def _random_initiate(self):
        for n in self.nodes:
            self.nodes[n].random_initiate()
        for e in self.edges:
            self.edges[e].random_initiate()
        return self

    def sample(self, size=None):
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
    # TODO: what is this down here?!
    # def determinize_node(self, ):


class AutoEncoderSimulator(Structure):
    def __init__(self,
                 num_latent_vars=None,
                 latent_nodes_adjm=None,
                 num_nodes_in_hidden_layers=None,
                 output_layer_dtype_list=None,
                 complexity=0):
        latent_nodes = ['latent_{}'.format(i) for i in range(num_latent_vars)]
        hidden_layer_nodes = [
            ['hidden{}_{}'.format(num_hidden+1, i+1)
            for i in range(num_nodes_in_hidden_layers[num_hidden])]
            for num_hidden in range(len(num_nodes_in_hidden_layers))
        ]
        output_nodes = ['x{}'.format(i+1) for i in range(len(output_layer_dtype_list))]
        total_nodes = latent_nodes + [item for sublist in hidden_layer_nodes for item in sublist] + output_nodes
        num_nodes = len(total_nodes)

        adj_matrix = pd.DataFrame(np.zeros(shape=(num_nodes, num_nodes)), columns=total_nodes, index=total_nodes)

        if latent_nodes_adjm:
            assert is_acyclic(adj_matrix=pd.DataFrame(latent_nodes_adjm, columns=latent_nodes, index=latent_nodes))
            adj_matrix.loc[latent_nodes, latent_nodes] = latent_nodes_adjm

        # connect hidden to first layer
        adj_matrix.loc[latent_nodes, hidden_layer_nodes[0]] = 1
        # interconnect hidden layers
        for i in range(len(hidden_layer_nodes)-1):
            adj_matrix.loc[hidden_layer_nodes[i], hidden_layer_nodes[i+1]] = 1
        # last hidden layer to output
        adj_matrix.loc[hidden_layer_nodes[-1], output_nodes] = 1
        dtypes = ['continuous']*(num_nodes-len(output_nodes)) + output_layer_dtype_list
        node_types = {name:type_ for name, type_ in zip(total_nodes, dtypes)}

        super().__init__(
            adj_matrix=adj_matrix,
            node_types=node_types,
            complexity=complexity
        )

    def get_latent_vars(self):
        cols = [c for c in self.data.columns if c[:6]=='latent']
        return self.data[cols]

    def get_output_vars(self):
        cols = [c for c in self.data.columns if c[0]=='x']
        return self.data[cols]

    def get_full_vars(self):
        return self.data


# if __name__ == '__main__':
#     a = AutoEncoderSimulator(
#         num_latent_vars=2,
#         num_nodes_in_hidden_layers=[2, 5, 10],
#         output_layer_dtype_list=['continuous']*20,
#         complexity=0
#     )
#     a.sample(size=1000)
#     plt.scatter(a.data.x1, a.data.x2, c=a.data.x3)
#     plt.show()


if __name__ == '__main__':
    e = Edge(complexity=0.8)
    print(e.complexity)

# if __name__ == '__main__':
#     from pprint import pprint
#     node_names = ['x{}'.format(i) for i in range(6)]
#     adjm = pd.DataFrame(
#         [
#             [0, 1, 0, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1],
#             [0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0]
#
#         ],
#         index=node_names,
#         columns=node_names
#     )
#     node_types = {'x0': 'continuous', 'x1': 'continuous', 'x2': 'continuous', 'x3': 'continuous', 'x4': 'continuous', 'x5': 'categorical'}
#     s = Structure(adj_matrix=adjm, node_types=node_types)
#     for n in s.nodes:
#         print(n, '========')
#         pprint(s.nodes[n].get_configs())
#     for e in s.edges:
#         print(e, '========')
#         pprint(s.edges[e].get_configs())
#
#     s.sample(size=1000)
#     print(s.nodes['x5'].get_configs())
#     # plt.scatter(s.data.x3, s.data.x4, c=s.data.x5)
#     plt.hist(s.data.x5)
#     plt.show()