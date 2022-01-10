from modules.sem.utils import topological_sort
from matplotlib import pyplot as plt
from modules.sem.nodes import *
from modules.sem.edges import *
from itertools import product
import pandas as pd


class SimpleStructure:
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


if __name__ == '__main__':
    from pprint import pprint
    node_names = ['x{}'.format(i) for i in range(6)]
    adjm = pd.DataFrame(
        [
            [0, 1, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]

        ],
        index=node_names,
        columns=node_names
    )
    node_types = {
        'x0': 'continuous', 'x1': 'continuous', 'x2': 'continuous',
        'x3': 'continuous', 'x4': 'continuous', 'x5': 'categorical'
    }
    s = SimpleStructure(adj_matrix=adjm, node_types=node_types)
    for n in s.nodes:
        print(n, '========')
        pprint(s.nodes[n].get_configs())
    for e in s.edges:
        print(e, '========')
        pprint(s.edges[e].get_configs())

    s.sample(size=1000)
    print(s.nodes['x5'].get_configs())
    plt.scatter(s.data.x3, s.data.x4, c=s.data.x5)
    plt.show()
