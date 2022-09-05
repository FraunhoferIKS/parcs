import numpy as np
import pandas as pd
from warnings import warn


class Randomizer:
    def __init__(self):
        self.num_nodes = None
        self.node_names = []
        self.adj_matrix = pd.DataFrame([])

        self.nodes = []
        self.edges = []

    def randomize(self, guideline=None, start_from=None, **kwargs):
        possibles = ['scratch', 'num_nodes', 'adj_matrix', 'incomplete_list']
        requires = ['guideline_dir']

    def post_fix(self):
        pass



    def _randomize_num_nodes(self, low=None, high=None, name_prefix=None):
        # sample a number
        self.num_nodes = np.random.uniform(low, high)
        # name list
        self.node_names = ['{}_{}'.format(name_prefix, i) for i in range(self.num_nodes)]
        return self

    def randomize_adj_matrix(self, num_nodes=None, density=0.5):
        if self.num_nodes and num_nodes:
            warn(
                'number of nodes already fixed (n={}), given num_nodes argument will be ignored'.format(self.num_nodes)
            )
        if not self.num_nodes and not num_nodes:
            raise KeyError('num_nodes must be given.')

        # create a fully connected and then mask based on sparsity
        shape_ = (self.num_nodes, self.num_nodes)
        self.adj_matrix = pd.DataFrame(
            np.triu(np.ones(shape=shape_), k=1)
        )
        mask = np.random.choice([0, 1], p=[1-density, density], size=shape_)
        self.adj_matrix = np.multiply(self.adj_matrix, mask)
        return self

    def randomize_edge_functions(self, guideline=None):
        pass