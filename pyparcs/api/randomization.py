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

import warnings
import numpy as np
import pandas as pd
from copy import deepcopy
from typing import Union, Optional, Iterable
from typeguard import typechecked
from pyparcs.api.utils import get_interactions_length, get_interactions_names
from pyparcs.api.mapping_functions import FUNCTION_PARAMS
from pyparcs.api.output_distributions import DISTRIBUTION_PARAMS

warnings.simplefilter(action='ignore', category=FutureWarning)


def is_eligible(guideline_tag, line_tags):
    """checks if a line must be addressed by the randomizer"""
    # both line and randomizer are untagged
    if guideline_tag is None and line_tags == []:
        return True
    # tags match
    elif guideline_tag in line_tags:
        return True
    else:
        return False


def randomize_edge_function(edge, guideline):
    """handles randomization of the edge functions"""
    if edge['function_name'] == '?':
        # pick a function
        edge['function_name'] = guideline.sample_keys(which='edges')
        # set empty dist param coefs
        edge['function_params'] = {param: '?' for param in FUNCTION_PARAMS[edge['function_name']]}
    return edge


def randomize_edge_function_parameters(edge, guideline):
    """handles randomization of the edge function parameters"""
    for e_p in edge['function_params']:
        if edge['function_params'][e_p] == '?':
            edge['function_params'][e_p] = guideline.sample_values(
                f"edges.{edge['function_name']}.{e_p}"
            )
    return edge


def randomize_node_distribution(node, guideline):
    """handles randomization of the node output distributions"""
    if node['output_distribution'] == '?':
        # pick a distribution
        node['output_distribution'] = guideline.sample_keys(which='nodes')
        # set empty dist param coefs
        node['dist_params_coefs'] = {
            param: {k: '?' for k in ['bias', 'linear', 'interactions']}
            for param in DISTRIBUTION_PARAMS[node['output_distribution']]
        }


def randomize_node_distribution_parameters(node, parent_list, guideline):
    """handles randomization of the node distribution parameters"""
    dist = node['output_distribution']
    num_parents = len(parent_list)
    num_interactions = get_interactions_length(num_parents)

    # there are two types of call for randomization: 1. '?' or 2. ['?', 1, '?', ...]

    for param in node['dist_params_coefs']:
        if node['dist_params_coefs'][param]['bias'] == '?':
            node['dist_params_coefs'][param]['bias'] = guideline.sample_values(f'nodes.{dist}.{param}.0')
        if node['dist_params_coefs'][param]['linear'] == '?':  # case-1 linear
            node['dist_params_coefs'][param]['linear'] = np.array([
                guideline.sample_values(f'nodes.{dist}.{param}.1')
                for _ in range(num_parents)
            ])
        elif '?' in node['dist_params_coefs'][param]['linear']:  # case-2 linear
            node['dist_params_coefs'][param]['linear'] = list(map(
                lambda coef: guideline.sample_values(f'nodes.{dist}.{param}.1')
                if coef == '?' else coef,
                node['dist_params_coefs'][param]['linear']
            ))
        if node['dist_params_coefs'][param]['interactions'] == '?':  # case-1 interactions
            node['dist_params_coefs'][param]['interactions'] = np.array([
                guideline.sample_values(f'nodes.{dist}.{param}.2')
                for _ in range(num_interactions)
            ])
        elif '?' in node['dist_params_coefs'][param]['interactions']:  # case-2 interactions
            node['dist_params_coefs'][param]['interactions'] = list(map(
                lambda coef: guideline.sample_values(f'nodes.{dist}.{param}.2')
                if coef == '?' else coef,
                node['dist_params_coefs'][param]['interactions']
            ))


def random_adj_matrix(node_names, density):
    # this shuffles the argument, but that's fine
    np.random.shuffle(node_names)
    # initiate random raw matrix
    adj_matrix = np.random.choice([0, 1], p=[1 - density, density],
                                  size=(num_nodes := len(node_names), num_nodes))
    # create the acyclic mask
    mask = np.triu(adj_matrix, k=1)
    # apply mask
    adj_matrix = np.multiply(adj_matrix, mask)

    return pd.DataFrame(adj_matrix, columns=node_names, index=node_names).astype(int)


def random_connection_adj_matrix(parent_nodes, child_nodes, density, mask):
    """creates a random adjacency matrix for connecting two subgraphs"""
    shape = [len(parent_nodes), len(child_nodes)]
    adj_matrix = pd.DataFrame(np.random.choice([0, 1], p=[1 - density, density], size=shape),
                              index=parent_nodes,
                              columns=child_nodes)
    # apply mask
    if isinstance(mask, pd.DataFrame):
        assert set(adj_matrix.index) == set(mask.index) and \
               set(adj_matrix.columns) == set(mask.columns)
        adj_matrix = adj_matrix.mul(mask.reindex_like(adj_matrix))
    return adj_matrix


def get_new_terms(parents):
    """Gives new terms to be added for connect randomizer"""
    interactions = list(map(lambda i: ''.join(i) if i[0] != i[1] else f'{i[0]}^2',
                            get_interactions_names(parents)))

    return '+'.join([f'?{i}' for i in parents + interactions])
