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
#  Contact: andreas.binder@iks.fraunhofer.de
import time
import re
import os
import yaml
from pyparcs.api.utils import digest_outline_input
from pyparcs.api.parsers import outline_splitter
from pyparcs.core.exceptions import DescriptionError


NEGATION_PREFIX = 'neg'


def temporal_edge_parser(old_key, old_value, resolved_file, n_timesteps):
    """
    takes in the old key and the old value and writes the new version into
    the resolved_file

    Parameters
    ----------
    old_key : str
        the dictionary key e.g. 'BP_{t-1}->BP_{t}'
    old_value : str
        the dictionary value e.g. 'identity()'
    resolved_file: dict
        the new dictionary
    n_timesteps: int
        number of time steps to generate the graph

    Returns
    -------
    resolved_file: dict
        the new dictionary after the update
    """
    split_expansion_sequence = r'\{(t.*?)\}'

    # TODO: currently only parses key dependencies, might need assertion
    for t in range(1, n_timesteps+1):
        new_key = str()
        new_value = old_value
        for split in re.split(split_expansion_sequence, old_key):
            if split[0:2] == 't-':
                delay = int(split[2:])  # remove 't-'
                if t >= delay:
                    new_key += str(t - delay)
                else:
                    new_key += NEGATION_PREFIX + str(delay - t)
            elif split == 't':
                new_key += str(t)
            else:
                new_key += split
        resolved_file[new_key] = new_value
    return resolved_file


def temporal_node_parser(old_key, old_value, resolved_file, n_timesteps):
    """
    takes in the old key and the old value and writes the new version into
    the resolved_file

    Parameters
    ----------
    old_key : str
        the dictionary key e.g. 'Drug_{t}'
    old_value : str
        the dictionary value e.g. 'bernoulli(p_=BP_{t}Age), correction[]'
    resolved_file: dict
        the new dictionary
    n_timesteps: int
        number of time steps to generate the graph

    Returns
    -------
    resolved_file: dict
        the new dictionary after the update
    """
    split_blv_temporal = r'\{(.*?)\}'
    split_initial_expansions = r'\{(\d*)\}|\{(-\d*)\}'
    split_expansion_sequence = r'\{(t.*?)\}'

    # is BLV
    if not re.search(split_blv_temporal, old_key):
        resolved_file[old_key] = old_value
    # is Temporal
    else:
        # is initial
        # TODO assumes no dependencies to be parsed in value
        if re.search(split_initial_expansions, old_key):
            new_key = re.sub('\{|\}', '', old_key)
            # remove potential negative values
            new_key = re.sub('-', NEGATION_PREFIX, new_key)
            # assign
            resolved_file[new_key] = old_value
        # is expansion (either recursive or non-recursive)
        else:
            for t in range(1,n_timesteps+1):
                new_key = re.sub('\{t\}', str(t), old_key)
                new_value = str()
                # splits old_value into previous timesteps (t-X), current timesteps (t) and
                # the remaining value phrase
                for split in re.split(split_expansion_sequence, old_value):
                    if split[0:2] == 't-':
                        # remove 't-'
                        delay = int(split[2:])
                        # check if delay < initial timesteps
                        if t >= delay:
                            new_value += str(t - delay)
                        # replace '-' with neg for correct value parsing
                        else:
                            new_value += NEGATION_PREFIX + str(delay - t)
                    elif split == 't':
                        new_value += str(t)
                    else:
                        new_value += split

                resolved_file[new_key] = new_value
    return resolved_file


def temporal(temporal_nodes, oldest):
    """
    The 'temporal' decorator for deterministc nodes with temporal inputs

    Using this decorator, the user writes columns like 'X_{t-2}' in the custom function.
    Then when data is passed as `X_5`, the decorator turns `X_5` into `X_{t-2}` properly.
    For that, the user needs to give the list of temporal nodes, and the oldest time index manually

    Parameters
    ----------
    temporal_nodes : list(str)
        list of temporal nodes that appear in the deterministic node
    oldest : str
        in the form of t-1, t-2, ..., marking the earliest t offset that appears in the custom function
    """
    def decorator(func):
        def wrapper(data):
            # find t-indexed temporal nodes
            temp_cols = [i for i in data.columns if i.rsplit('_', 1)[0] in temporal_nodes]
            # resolve the 'neg' sign
            temp_cols = [i.replace('neg', '-') for i in temp_cols]
            # get minimum t-index
            min_t = min(int(i.rsplit('_', 1)[1]) for i in temp_cols)
            # find out t
            if oldest != 't':
                min_t_offset = int(oldest.split('-')[1])
            else:
                min_t_offset = 0
            current_t = min_t_offset + min_t
            # apply t to all columns
            rename_dict = {}
            for c in temp_cols:
                offset = current_t - int(c.rsplit('_', 1)[1])
                # 'neg' sign back to normal, because we want to replace the names
                c = c.replace('-', 'neg')
                if offset == 0:
                    rename_dict[c] = f"{c.rsplit('_', 1)[0]}_{{t}}"
                else:
                    rename_dict[c] = f"{c.rsplit('_', 1)[0]}_{{t-{offset}}}"
            data.rename(columns=rename_dict, inplace=True)
            # Call the decorated function with updated data
            return func(data)
        return wrapper
    return decorator


def temporal_outline_parser(outline, n_timesteps):
    """Parser for temporal outlines

    This function reads the temporal outline and returns the time-flattened outline.

    Parameters
    ----------
    outline: dict or path
        temporal outline
    n_timesteps: int
        number of time steps to generate the graph

    Returns
    -------
    flat_outline: dict
    """

    outline = digest_outline_input(outline)
    nodes, edges = outline_splitter(outline)

    new_outline = {}

    for node, line in nodes.items():
        new_outline = temporal_node_parser(node, line, new_outline, n_timesteps)

    for edge, line in edges.items():
        new_outline = temporal_edge_parser(edge, line, new_outline, n_timesteps)

    return new_outline