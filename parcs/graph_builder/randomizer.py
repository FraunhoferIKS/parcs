import numpy as np
import pandas as pd
from warnings import warn
from parcs.cdag.output_distributions import DISTRIBUTION_PARAMS
from parcs.cdag.mapping_functions import FUNCTION_PARAMS
from parcs.graph_builder import parsers, utils

class ParamRandomizer:
    def __init__(self, graph_dir=None, guideline_dir=None):
        # read randomization guideline
        self.guideline = parsers.guideline_parser(guideline_dir)
        # read fixed nodes and edges
        self.nodes, self.edges = parsers.graph_file_parser(graph_dir)
        # setup parent dictionary
        self.nodes_parents = {
            node['name']: sorted([
                e['name'].split('->')[0] for e in self.edges
                if e['name'].split('->')[1] == node['name']
            ])
            for node in self.nodes
        }

        # 1. pick distributions and edge functions for '?' values
        self._set_node_distributions()._set_edge_functions()
        # 2. fill in all '?' params
        self._fill_in_edges()
        print(self.edges)

    @staticmethod
    def __random_picker(options):
        p_options = np.ones(shape=(len(options),))/len(options)
        return np.random.choice(options, p=p_options)


    def __directive_picker(self, directive):
        if isinstance(directive, list):
            if directive[0] == 'choice':
                options = directive[1:]
                return self.__random_picker(options)
            elif directive[0] == 'i-range':
                return np.random.randint(low=directive[1], high=directive[2]+1)
            elif directive[0] == 'f-range':
                return np.random.uniform(low=directive[1], high=directive[2])
            else:
                return ValueError
        else:
            return directive

    def _set_edge_functions(self):
        for e_ind in range(len(self.edges)):
            # check if distribution is set
            if self.edges[e_ind]['function_name'] == '?':
                # pick a function
                func = self.__random_picker(list(self.guideline['edges'].keys()))
                self.edges[e_ind]['function_name'] = func
                # set empty dist param coefs
                self.edges[e_ind]['function_params'] = {
                    param: '?' for param in FUNCTION_PARAMS[func]
                }
        return self

    def _set_node_distributions(self):
        for node in self.nodes:
            # check if distribution is set
            if node['output_distribution'] == '?':
                # pick a distribution
                dist = self.__random_picker(list(self.guideline['nodes'].keys()))
                node['output_distribution'] = dist
                # set empty dist param coefs
                node['dist_params_coefs'] = {
                    param: '?' for param in DISTRIBUTION_PARAMS[dist]
                }
        return self

    def _fill_in_edges(self):
        for edge in self.edges:
            for p in edge['function_params']:
                if edge['function_params'][p] == '?':
                    directive = self.guideline['edges'][edge['function_name']][p]
                    edge['function_params'][p] = self.__directive_picker(directive)


# class FreeRandomizer:
#     def __init__(self, guideline_dir):
#         guide = utils.config_parser(guideline_dir)
#         # 1. set num nodes
#         self.num_nodes = self._guideline_sampler(guide['graph']['num_nodes'])
#         # 2. set adj_matrix
#
#     @staticmethod
#     def _guideline_sampler(item):
#         if isinstance(item, list):
#             if item[0] == 'i-range':
#                 options = [i for i in range(item[1], item[2]+1)]
#                 return np.random.choice(options, p=[1/len(options)]*options)
#             elif item[0] == 'f-range':
#                 return np.random.uniform(item[1], item[2])
#             elif item[0] == 'choice':
#                 options = item[1:]
#                 return np.random.choice(options, p=[1 / len(options)] * options)
#             else:
#                 raise ValueError('first element is other than i-range/f-range/choice')
#         else:
#             return item


# class OldRandomizer:
#     def __init__(self):
#         self.num_nodes = None
#         self.node_names = []
#         self.adj_matrix = pd.DataFrame([])
#
#         self.nodes = []
#         self.edges = []
#
#     def randomize(self, guideline=None,
#                   num_nodes=None, adj_matrix=None, nodes=None, edges=None,
#                   randomize_num_nodes : bool=False):
#         pass
#
#     def post_fix(self):
#         pass
#
#     def _randomize_num_nodes(self, low=None, high=None, name_prefix=None):
#         # sample a number
#         self.num_nodes = np.random.uniform(low, high)
#         # name list
#         self.node_names = ['{}_{}'.format(name_prefix, i) for i in range(self.num_nodes)]
#         return self
#
#     def randomize_adj_matrix(self, num_nodes=None, density=0.5):
#         if self.num_nodes and num_nodes:
#             warn(
#                 'number of nodes already fixed (n={}), given num_nodes argument will be ignored'.format(self.num_nodes)
#             )
#         if not self.num_nodes and not num_nodes:
#             raise KeyError('num_nodes must be given.')
#
#         # create a fully connected and then mask based on sparsity
#         shape_ = (self.num_nodes, self.num_nodes)
#         self.adj_matrix = pd.DataFrame(
#             np.triu(np.ones(shape=shape_), k=1)
#         )
#         mask = np.random.choice([0, 1], p=[1-density, density], size=shape_)
#         self.adj_matrix = np.multiply(self.adj_matrix, mask)
#         return self
#
#     def randomize_edge_functions(self, guideline=None):
#         pass


if __name__ == '__main__':
    rand = ParamRandomizer(
        graph_dir='../../graph_templates/causal_triangle.yml',
        guideline_dir='../../guidelines/simple_guideline.yml'
    )

