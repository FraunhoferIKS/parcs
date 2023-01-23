#  Copyright (c) 2022. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
#  acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses/>.
#
#  https://www.gnu.de/documents/gpl-2.0.de.html
#
#  Contact: alireza.zamanian@iks.fraunhofer.de

from copy import deepcopy
from pyparcs.cdag.mapping_functions import FUNCTION_PARAMS
from pyparcs.graph_builder import parsers
from pyparcs.cdag.utils import get_interactions_length, topological_sort
from itertools import product, combinations
from pyparcs.graph_builder.utils import config_parser, config_dumper
from pyparcs.cdag.output_distributions import OUTPUT_DISTRIBUTIONS, DISTRIBUTION_PARAMS
from pyparcs.exceptions import parcs_assert, DescriptionFileError
import pandas as pd
import numpy as np
import re
import os
import warnings
from pathlib import Path
from typing import Union, Optional, Iterable
from typeguard import typechecked

warnings.simplefilter(action='ignore', category=FutureWarning)


@typechecked
class ParamRandomizer:
    def __init__(self,
                 graph_dir: Optional[Union[str, Path]] = None,
                 guideline_dir: Optional[Union[str, Path]] = None):
        # read randomization guideline
        self.guideline = parsers.guideline_parser(guideline_dir)
        # read fixed nodes and edges
        self.nodes, self.edges = parsers.graph_file_parser(graph_dir)

    @staticmethod
    def directive_picker(directive: Union[list, int, float]):
        if isinstance(directive, list):
            if directive[0] == 'choice':
                options = directive[1:]
                return np.random.choice(options)
            else:
                ranges = directive[1:]
                # if multiple ranges are given
                parcs_assert(len(ranges) % 2 == 0, DescriptionFileError,
                             f'The number of range numbers should be even, got {len(ranges)}')
                num_ranges = int(len(ranges) / 2)
                # pick the range
                range_ = np.random.randint(num_ranges)
                low, high = ranges[range_ * 2], ranges[range_ * 2 + 1]
                if directive[0] == 'i-range':
                    return np.random.randint(low=low, high=high)
                elif directive[0] == 'f-range':
                    return np.random.uniform(low=low, high=high)
                else:
                    raise ValueError
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
            try:
                assert 'output_distribution' in node
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
            except AssertionError:
                continue
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
            try:
                assert 'output_distribution' in node
                dist = node['output_distribution']
                num_parents = len(self.nodes_parents[node['name']])
                num_interactions = get_interactions_length(num_parents)
                for param in node['dist_params_coefs']:
                    if node['dist_params_coefs'][param]['bias'] == '?':
                        node['dist_params_coefs'][param]['bias'] = \
                            self.directive_picker(self.guideline['nodes'][dist][param][0])
                    if node['dist_params_coefs'][param]['linear'] == '?':
                        node['dist_params_coefs'][param]['linear'] = np.array([
                            self.directive_picker(self.guideline['nodes'][dist][param][1])
                            for _ in range(num_parents)
                        ])
                    if node['dist_params_coefs'][param]['interactions'] == '?':
                        node['dist_params_coefs'][param]['interactions'] = np.array([
                            self.directive_picker(self.guideline['nodes'][dist][param][2])
                            for _ in range(num_interactions)
                        ])
            except AssertionError:
                continue
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


@typechecked
class ExtendRandomizer(ParamRandomizer):
    def __init__(self,
                 graph_dir: Optional[Union[str, Path]] = None,
                 guideline_dir: Optional[Union[str, Path]] = None):
        super().__init__(graph_dir=graph_dir, guideline_dir=guideline_dir)

        # pick number of nodes:
        self.num_nodes = self._set_num_nodes()
        # pick names
        if 'node_name_prefix' not in self.guideline['graph']:
            self.guideline['graph']['node_name_prefix'] = 'H'
        self.node_names = self._set_node_names()
        # randomize main adj matrix
        self.adj_matrix = self._random_adj_matrix()
        # get local node order of the user-given graph, given already by self.nodes and self.edges
        user_node_order = self._user_topological_sort()
        # assign places
        indices = np.sort(np.random.choice(range(self.num_nodes), replace=False, size=len(user_node_order)))
        mapper = {
            '{}_{}'.format(self.guideline['graph']['node_name_prefix'], i): n for i, n in zip(indices, user_node_order)
        }
        self.adj_matrix.rename(columns=mapper, index=mapper, inplace=True)
        # update adj matrix elements
        for e in self.edges:
            par, child = e['name'].split('->')
            self.adj_matrix.loc[par, child] = 1
        # extend nodes and edges
        self._extend_nodes(indices)._extend_edges()
        # modify param coefs based on newly added parents
        self._update_param_coefs()

    def _set_num_nodes(self):
        num_node_directive = self.guideline['graph']['num_nodes']
        if isinstance(num_node_directive, list):
            assert num_node_directive[1] >= len(self.nodes)
        else:
            assert num_node_directive >= len(self.nodes)
        return self.directive_picker(num_node_directive)

    def _set_node_names(self):
        try:
            return [
                '{}_{}'.format(self.guideline['graph']['node_name_prefix'], i) for i in range(self.num_nodes)
            ]
        except KeyError:
            return [
                'H_{}'.format(i) for i in range(self.num_nodes)
            ]

    def _user_topological_sort(self):
        # local adj
        num_user_node = len(self.nodes)
        user_adj_matrix = pd.DataFrame(
            np.zeros(shape=(num_user_node, num_user_node)),
            columns=[n['name'] for n in self.nodes],
            index=[n['name'] for n in self.nodes]
        )
        for e in self.edges:
            par, child = e['name'].split('->')
            user_adj_matrix.loc[par, child] = 1

        return topological_sort(user_adj_matrix)

    def _random_adj_matrix(self):
        density = self.directive_picker(self.guideline['graph']['graph_density'])
        adj_matrix = np.random.choice([0, 1], p=[1 - density, density], size=(self.num_nodes, self.num_nodes))
        mask = np.triu(adj_matrix, k=1)
        adj_matrix = np.multiply(adj_matrix, mask)
        return pd.DataFrame(adj_matrix, columns=self.node_names, index=self.node_names)

    def _extend_nodes(self, indices: np.ndarray):
        for i in range(self.num_nodes):
            if i not in indices:
                node_name = '{}_{}'.format(self.guideline['graph']['node_name_prefix'], i)
                parents = sorted(list(self.adj_matrix[self.adj_matrix[node_name] == 1].index))
                self.nodes.append({
                    'name': node_name,
                    **parsers.node_parser('random', parents)
                })
        return self

    def _extend_edges(self):
        current_edges = [e['name'] for e in self.edges]
        for par, child in combinations(self.adj_matrix.columns, 2):
            if '{}->{}'.format(par, child) not in current_edges and self.adj_matrix.loc[par, child] == 1:
                self.edges.append({
                    'name': '{}->{}'.format(par, child),
                    **parsers.edge_parser('random')
                })
        return self

    def _update_param_coefs(self):
        return self


@typechecked
class FreeRandomizer(ExtendRandomizer):
    def __init__(self, guideline_dir: Optional[Union[str, Path]] = None):
        super().__init__(graph_dir=None, guideline_dir=guideline_dir)


@typechecked
class ConnectRandomizer(ParamRandomizer):
    def __init__(self,
                 parent_graph_dir: Optional[Union[str, Path]] = None,
                 child_graph_dir: Optional[Union[str, Path]] = None,
                 guideline_dir: Optional[Union[str, Path]] = None,
                 adj_matrix_mask: pd.DataFrame = None,
                 delete_temp_graph_description: bool = True):
        pgd = config_parser(parent_graph_dir)
        cgd = config_parser(child_graph_dir)
        n_p = [n for n in pgd if '->' not in n]
        l_p = len(n_p)
        n_c = [n for n in cgd if '->' not in n]
        e_c = [n for n in cgd if '->' in n]
        l_c = len(n_c)

        guideline = config_parser(guideline_dir)

        # sample connection adj_matrix
        density = self.directive_picker(guideline['graph']['graph_density'])
        adj_matrix = np.random.choice([0, 1], p=[1 - density, density], size=(l_p, l_c))
        adj_matrix = np.multiply(adj_matrix, adj_matrix_mask.values)
        adj_matrix = pd.DataFrame(adj_matrix, index=adj_matrix_mask.index, columns=adj_matrix_mask.columns)
        # make additional edges
        e_opt = list(guideline['edges'].keys())
        add_edges = {
            '{}->{}'.format(p, c): '{}(?), correction[]'.format(np.random.choice(e_opt))
            for p, c in product(n_p, n_c) if adj_matrix.loc[p, c] == 1
        }
        # modify receiving child nodes
        for n in n_c:
            if adj_matrix[n].sum() == 0:  # no added edge
                # delete all stars and move on
                cgd[n] = cgd[n].replace('*', '')
                continue
            dist, arg, rest = re.split('[()]', cgd[n].replace(' ', ''))
            assert dist in OUTPUT_DISTRIBUTIONS, 'non stochastic child node is receiving additional edge'
            params = arg.split(',')
            star_flag = False
            for i in range(len(params)):
                if params[i][0] != '*':
                    if i == len(params) - 1 and not star_flag:  # last param and still no star
                        # remove the new edges which doesn't do anything
                        e_names = [i for i in add_edges if i.split('->')[1] == n]
                        for e in e_names:
                            del add_edges[e]
                    continue
                star_flag = True  # has at least one star param
                # remove '*'
                params[i] = params[i][1:]
                k, v = params[i].split('=')
                if v == '?':
                    continue
                # add new parents
                current_par = [p for p in n_c if '{}->{}'.format(p, n) in e_c]
                new_par = [p for p in n_p if adj_matrix.loc[p, n] == 1]

                # linear terms
                coefs = [self.directive_picker(guideline['nodes'][dist][k][1]) for _ in range(len(new_par))]
                for coef, par in zip(coefs, new_par):
                    if coef > 0:
                        v += '+{}{}'.format(np.round(coef, 2), par)
                    elif coef < 0:
                        v += '{}{}'.format(np.round(coef, 2), par)
                # interaction terms
                terms = [i for i in combinations(new_par, 2)] + [i for i in product(new_par, current_par)]
                terms = [''.join(i) for i in terms]
                coefs = [self.directive_picker(guideline['nodes'][dist][k][2]) for _ in range(len(terms))]
                for coef, par in zip(coefs, terms):
                    if coef > 0:
                        v += '+{}{}'.format(np.round(coef, 2), par)
                    elif coef < 0:
                        v += '{}{}'.format(np.round(coef, 2), par)
                params[i] = k + '=' + v
            arg = ','.join(params)
            cgd[n] = '{}({}){}'.format(dist, arg, rest)
        gd = {**cgd, **pgd, **add_edges}
        config_dumper(gd, 'combined_gdf.yml')
        super().__init__(graph_dir='combined_gdf.yml', guideline_dir=guideline_dir)
        if delete_temp_graph_description:
            os.remove('combined_gdf.yml')


@typechecked
def guideline_iterator(guideline_dir: Optional[Union[str, Path]] = None, to_iterate: str = None, steps: int = None,
                       repeat: int = 1):
    @typechecked
    def _get_iterable(directive: list, steps: Optional[int]):
        # assert isinstance(directive, list), 'GuidelineIterator received fixed value as the directive'
        if directive[0] == 'f-range':
            assert len(directive) == 3, 'multirange doesn\'t work in guideline iterator'
            return np.linspace(directive[1], directive[2], steps)
        elif directive[0] == 'i-range':
            assert len(directive) == 3, 'multirange doesn\'t work in guideline iterator'
            return range(directive[1], directive[2] + 1)
        elif directive[0] == 'choice':
            return directive[1:]
        # else:
        #     raise ValueError

    @typechecked
    def _get_directive(dict_: dict, path: list[str]):
        directive = dict_[path[0]]
        if len(path) == 1:
            return directive
        for step in path[1:]:
            directive = directive[step]
        return directive

    @typechecked
    def _set_directive(dict_: dict, path: list[str], value: Union[np.float, float, int]):
        new_guideline = deepcopy(dict_)
        if isinstance(value, np.float):
            value = float(value)
        replacement = value
        for i in range(1, len(path)):
            temp = _get_directive(dict_, path[:-i])
            temp[path[-i]] = replacement
            replacement = temp
        new_guideline[path[0]] = replacement
        return new_guideline

    @typechecked
    def _generator(dict_: dict, iterable: Iterable, path: list[str], repeat: int):
        for i in iterable:
            for epoch in range(repeat):
                new_guideline = _set_directive(dict_, path, i)
                dir_ = './temp_analysis_guideline.yml'
                config_dumper(new_guideline, dir_)
                yield dir_, epoch, i
        os.remove('./temp_analysis_guideline.yml')

    guideline = config_parser(guideline_dir)
    path = to_iterate.split('/')
    directive = _get_directive(guideline, path)
    iterable = _get_iterable(directive, steps)
    generator = _generator(guideline, iterable, path, repeat)
    return generator


if __name__ == '__main__':
    from pyparcs.cdag.graph_objects import Graph
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

    # for dir_, epoch, value in guideline_iterator(guideline_dir='../../guidelines/simple_guideline.yml',
    #                                             to_iterate='graph/num_nodes',
    #                                             repeat=2):
    #     print('num_nodes:', value)
    #     print('\t EPOCH:', epoch)

    #     rndz = FreeRandomizer(
    #         guideline_dir=dir_
    #     )
    #     nodes, edges = rndz.get_graph_params()
    #     g = Graph(nodes=nodes, edges=edges)
    #     samples = g.sample(size=1)
    #     print(samples)
    #     print('=====')
