import numpy as np
import pandas as pd
from warnings import warn
from parcs.graph_builder import utils

class FixedRandomizer:
    def __init__(self, guideline_dir):

class FreeRandomizer:
    def __init__(self, guideline_dir):
        guide = utils.config_parser(guideline_dir)
        # 1. set num nodes
        self.num_nodes = self._guideline_sampler(guide['graph']['num_nodes'])
        # 2. set adj_matrix

    @staticmethod
    def _guideline_sampler(item):
        if isinstance(item, list):
            if item[0] is 'i-range':
                options = [i for i in range(item[1], item[2]+1)]
                return np.random.choice(options, p=[1/len(options)]*options)
            elif item[0] is 'f-range':
                return np.random.uniform(item[1], item[2])
            elif item[0] is 'choice':
                options = item[1:]
                return np.random.choice(options, p=[1 / len(options)] * options)
            else:
                raise ValueError('first element is other than i-range/f-range/choice')
        else:
            return item




class Randomizer:
    def __init__(self):
        self.num_nodes = None
        self.node_names = []
        self.adj_matrix = pd.DataFrame([])

        self.nodes = []
        self.edges = []

    def randomize(self, guideline=None,
                  num_nodes=None, adj_matrix=None, nodes=None, edges=None,
                  randomize_num_nodes : bool=False):
        pass

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


if __name__ == '__main__':
    rand = Randomizer()
    rand.randomize(guideline='simple_guideline')
