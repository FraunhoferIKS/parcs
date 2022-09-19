import numpy as np
import pandas as pd
from parcs.cdag.output_distributions import DISTRIBUTION_PARAMS
from parcs.cdag.mapping_functions import FUNCTION_PARAMS
from parcs.graph_builder import parsers
from parcs.cdag.utils import get_interactions_length, topological_sort
from itertools import combinations as comb
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class ParamRandomizer:
    def __init__(self, graph_dir=None, guideline_dir=None):
        # read randomization guideline
        self.guideline = parsers.guideline_parser(guideline_dir)
        # read fixed nodes and edges
        self.nodes, self.edges = parsers.graph_file_parser(graph_dir)

    def directive_picker(self, directive):
        if isinstance(directive, list):
            if directive[0] == 'choice':
                options = directive[1:]
                return np.random.choice(options)
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
                func = np.random.choice(list(self.guideline['edges'].keys()))
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
                dist = np.random.choice(list(self.guideline['nodes'].keys()))
                node['output_distribution'] = dist
                # set empty dist param coefs
                node['dist_params_coefs'] = {
                    param: {k: '?' for k in ['bias', 'linear', 'interactions']}
                    for param in DISTRIBUTION_PARAMS[dist]
                }
        return self

    def _fill_in_edges(self):
        for edge in self.edges:
            for p in edge['function_params']:
                if edge['function_params'][p] == '?':
                    directive = self.guideline['edges'][edge['function_name']][p]
                    edge['function_params'][p] = self.directive_picker(directive)
        return self

    def _fill_in_nodes(self):
        for node in self.nodes:
            dist = node['output_distribution']
            num_parents = len(self.nodes_parents[node['name']])
            num_interactions = get_interactions_length(num_parents)
            for param in node['dist_params_coefs']:
                if node['dist_params_coefs'][param]['bias'] == '?':
                    node['dist_params_coefs'][param]['bias'] = \
                        self.directive_picker(self.guideline['nodes'][dist][param][0])
                if node['dist_params_coefs'][param]['linear'] == '?':
                    node['dist_params_coefs'][param]['linear'] = np.array([
                        self.directive_picker(self.guideline['nodes'][dist][param][0])
                        for _ in range(num_parents)
                    ])
                if node['dist_params_coefs'][param]['interactions'] == '?':
                    node['dist_params_coefs'][param]['interactions'] = np.array([
                        self.directive_picker(self.guideline['nodes'][dist][param][0])
                        for _ in range(num_interactions)
                    ])
        return self

    def _setup(self):
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
        self._fill_in_edges()._fill_in_nodes()

        return self

    def get_graph_params(self):
        self._setup()
        return self.nodes, self.edges


class ExtendRandomizer(ParamRandomizer):
    def __init__(self,
                 graph_dir=None, guideline_dir=None):
        super().__init__(graph_dir=graph_dir, guideline_dir=guideline_dir)
        # pick number of nodes
        num_node_directive = self.guideline['graph']['num_nodes']
        if isinstance(num_node_directive, list):
            assert num_node_directive[1] >= len(self.nodes)
        else:
            assert num_node_directive >= len(self.nodes)
        self.num_nodes = self.directive_picker(num_node_directive)
        # pick names
        self.node_names = [
            '{}_{}'.format(self.guideline['graph']['node_name_prefix'], i) for i in range(self.num_nodes)
        ]
        # randomize adj matrix
        self.adj_matrix = self._random_adj_matrix()
        # get local adj_matrix
        local_node_order = self._local_topological_sort()
        # assign places
        indices = np.sort(np.random.choice(range(self.num_nodes), replace=False, size=len(local_node_order)))
        mapper = {
            '{}_{}'.format(self.guideline['graph']['node_name_prefix'], i): n for i, n in zip(indices, local_node_order)
        }
        self.adj_matrix.rename(columns=mapper, index=mapper, inplace=True)
        # update adj matrix elements
        for e in self.edges:
            par, child = e['name'].split('->')
            self.adj_matrix.loc[par, child] = 1

        # extend nodes
        for i in range(self.num_nodes):
            if i not in indices:
                node_name = '{}_{}'.format(self.guideline['graph']['node_name_prefix'], i)
                parents = sorted(list(self.adj_matrix[self.adj_matrix[node_name]==1].index))
                self.nodes.append({
                    'name': node_name,
                    **parsers.node_parser('random', parents)
                })
        # extend edges
        current_edges = [e['name'] for e in self.edges]
        for par, child in comb(self.adj_matrix.columns, 2):
            if '{}->{}'.format(par, child) not in current_edges and self.adj_matrix.loc[par, child]==1:
                self.edges.append({
                    'name': '{}->{}'.format(par, child),
                    **parsers.edge_parser('random')
                })

    def _local_topological_sort(self):
        # local adj
        num_local_node = len(self.nodes)
        local_adj_matrix = pd.DataFrame(
            np.zeros(shape=(num_local_node, num_local_node)),
            columns=[n['name'] for n in self.nodes],
            index=[n['name'] for n in self.nodes]
        )
        for e in self.edges:
            par, child = e['name'].split('->')
            local_adj_matrix.loc[par, child] = 1

        return topological_sort(local_adj_matrix)
    def _random_adj_matrix(self):
        density = self.directive_picker(self.guideline['graph']['graph_density'])
        adj_matrix = np.random.choice([0, 1], p=[1-density, density], size=(self.num_nodes, self.num_nodes))
        mask = np.triu(adj_matrix, k=1)
        adj_matrix = np.multiply(adj_matrix, mask)
        return pd.DataFrame(adj_matrix, columns=self.node_names, index=self.node_names)


class FreeRandomizer(ExtendRandomizer):
    def __init__(self, guideline_dir=None):
        super().__init__(graph_dir=None, guideline_dir=guideline_dir)


if __name__ == '__main__':
    from parcs.cdag.graph_objects import Graph
    import numpy as np
    np.random.seed(1)

    rand = ParamRandomizer(
        graph_dir='../../graph_templates/causal_triangle.yml',
        guideline_dir='../../guidelines/simple_guideline.yml'
    )
    nodes, edges = rand.get_graph_params()

    g = Graph(nodes=nodes, edges=edges)
    data, errors = g.sample(size=500, cache_sampling=True, cache_name='exp_1', return_errors=True)
    print(data)
    from matplotlib import pyplot as plt
    plt.scatter(data['C'], data['A'], c=data['Y'])
    plt.show()

