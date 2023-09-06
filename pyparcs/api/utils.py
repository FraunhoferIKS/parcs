#  Copyright (c) 2023. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
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
from typing import Union, List
import yaml
import numpy as np
import pandas as pd
from pyparcs.core.exceptions import parcs_assert, DescriptionError
from itertools import combinations_with_replacement as comb_w_repl


def digest_outline_input(input_: Union[str, dict]) -> dict:
    """
    Gives the dict version of the outline, for both str or dict
    inputs.

    Parameters
    ----------
    input_: str or dict
        the path to YAML file, or the outline dict

    Returns
    -------
    outline: dict
    """
    # if Dict
    if isinstance(input_, dict):
        return input_
    # if Path
    with open(input_, 'r') as stream:
        config_loaded = yaml.safe_load(stream)
    return config_loaded


def get_adj_matrix(node_list: list, parents_lists: dict) -> pd.DataFrame:
    """
    Creates adjacency matrix based on the node/parent lists.

    Parameters
    ----------
    node_list : str
        sorted list of node names
    parents_lists : dict
        with the format of {node_name: sorted list of its parents}

    Returns
    -------
    get_adj_matrix: pandas DataFrame

    See Also
    --------
    :ref:`flowchart <flowchart_get_adj_matrix>` for `get_adj_matrix()`
    """
    num_n = len(node_list)

    adj_matrix = pd.DataFrame(np.zeros(shape=(num_n, num_n)),
                              index=node_list, columns=node_list)
    for n in node_list:
        adj_matrix.loc[parents_lists[n], n] = 1

    return adj_matrix.astype(int)


def topological_sort(adj_matrix: pd.DataFrame) -> list:
    """
    performs topological sorting of a given adjacency matrix.
    It is an implementation of Kahn's algorithm.

    Parameters
    ----------
    adj_matrix : pandas DataFrame
        the adjacency matrix to be sorted

    Returns
    -------
    sorted_nodes : list(str)
        sorted list of all nodes


    Raises
    ------
    ValueError
        If index and column names doesn't match
    DescriptionError
        If the adjacency matrix is not acyclic

    See Also
    --------
    :ref:`flowchart <flowchart_topological_sort>` for `topological_sort()`
    """
    parcs_assert(set(adj_matrix.columns) == set(adj_matrix.index),
                 ValueError,
                 "index and column names in the adjacency matrix doesn't match")

    adjm = adj_matrix.copy(deep=True).values
    ordered_list = []
    covered_nodes = 0
    while covered_nodes < adjm.shape[0]:
        # sum r/x -> r edges
        sum_c = adjm.sum(axis=0)
        # find nodes with no parents
        parent_inds = list(np.where(sum_c == 0)[0])
        parcs_assert(len(parent_inds) != 0,
                     DescriptionError,
                     "Adjacency matrix is not acyclic")

        covered_nodes += len(parent_inds)
        # add to the list
        ordered_list += parent_inds
        # remove parent edges
        adjm[parent_inds, :] = 0
        # eliminate from columns by assigning values
        adjm[:, parent_inds] = 10
    return [adj_matrix.columns.tolist()[idx] for idx in ordered_list]


def get_interactions_values(data: np.ndarray) -> np.ndarray:
    """
    Returns the columns of product of interaction terms. The interaction terms are of length 2.
    The order of interaction terms follow the order
    of `itertools: combination with replacement
    <https://docs.python.org/3/library/itertools.html#itertools.combinations_with_replacement>`_
    module.

    Parameters
    ----------
    data: array-like

    Returns
    -------
    interaction_data: ndarray
        2D array of NxM, where `N` is the row number of data, and `M`
        is the number of interaction terms, determined by
        :func:`pyparcs.api.utils.get_interactions_length`

    Examples
    --------
    >>> get_interactions_values([[1, 2],
    ...                          [3, 9]])
    array([[ 1,  2,  4],
           [ 9, 27, 81]])
    """
    index_comb = list(comb_w_repl(range(data.shape[1]), 2))
    out = np.array([
        np.prod([data[:, ind[0]], data[:, ind[1]]], axis=0)
        for ind in index_comb
    ]).transpose()

    return out


def get_interactions_length(raw_data_len: int) -> int:
    """
    Returns length of interaction terms

    Parameters
    ----------
    raw_data_len: int
        length of the data input

    Returns
    -------
    interactions_len: int
        length of the interactions of the data

    Examples
    --------
    >>> get_interactions_length(2)
    3
    """
    dummy_data = np.ones(shape=(raw_data_len,))
    return len([np.prod(i) for i in comb_w_repl(dummy_data, 2)])


def get_interactions_names(parents: List[str]) -> List[list]:
    """ **Gives parents pairings for each interaction term**

    This function is used to trace which parents are making an interaction term in some index.

    Parameters
    ----------
    parents : list of str
        list of parents

    Returns
    -------
    pairings : list of tuple
        list of all parent pairings, where the index corresponds to the interaction data

    Examples
    --------
    >>> get_interactions_names(['A', 'B', 'C'])
    [['A', 'A'], ['A', 'B'], ['A', 'C'], ['B', 'B'], ['B', 'C'], ['C', 'C']]
    """
    return [sorted(i) for i in comb_w_repl(parents, 2)]


def dot_prod(data: np.ndarray, coef: dict) -> np.ndarray:
    """
    dot product of data and bias/linear/interactions coefs

    Parameters
    ----------
    data: ndarray
    coef: dict
        including three keys `bias`, `linear` and `interactions`:
        - bias is a scalar number
        - linear is a list, with the size of the input dimension
        - interactions is a list, with the size of the interactions length

    Returns
    -------
    result: ndarray
        N-length data result

    Examples
    --------
    >>> some_coef = {'bias': 1, 'linear': [1, 3], 'interactions': [0, 1, 0]}
    >>> dot_prod(np.array([[1, 2], [10, 20]]), some_coef)
    array([ 10, 271])
    """
    if data.shape[0] == 0:
        data_augmented = [np.array([]) for _ in range(3)]
    else:
        data_augmented = [data, get_interactions_values(data)]

    return coef['bias'] + np.array([
        np.dot(d, coef[i])
        for (i, d) in zip(['linear', 'interactions'], data_augmented)
    ]).sum(axis=0)
