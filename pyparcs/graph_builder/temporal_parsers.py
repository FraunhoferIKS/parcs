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
#  Contact: andreas.binder@iks.fraunhofer.de
import yaml
import re
import os
import time  

from pyparcs.graph_builder.parsers import graph_file_parser

from pyparcs.exceptions import (parcs_assert, DescriptionFileError, ExternalResourceError,
                                GuidelineError)


NEGATION_PREFIX = 'neg'

def config_parser(dir_):
    with open(dir_, 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf

def temporal_edge_parser(old_key, old_value, resolved_file):
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

    Returns
    -------
    resolved_file: dict
        the new dictionary after the update
    """
    n_timesteps = resolved_file['n_timesteps']

    split_expansion_sequence = r'\{(t.*?)\}'

    # TODO currently only parses key dependencies, might need assertion
    for t in range(1,n_timesteps+1):
        new_key = str() 
        new_value = old_value
        for split in re.split(split_expansion_sequence, old_key):            
            if split[0:2] == 't-':
                delay = int(split[2:]) # remove 't-'
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


def temporal_node_parser(old_key, old_value, resolved_file):
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

    Returns
    -------
    resolved_file: dict
        the new dictionary after the update
    """
    split_BLV_Temporal = r'\{(.*?)\}'
    split_Initial_Expansions = r'\{(\d*)\}|\{(-\d*)\}'
    split_expansion_sequence = r'\{(t.*?)\}'

    n_timesteps = resolved_file['n_timesteps']

    # is BLV
    if not re.search(split_BLV_Temporal, old_key):
        resolved_file[old_key] = old_value
    # is Temporal
    else: 
        # is initial
        # TODO assumes no dependencies to be parsed in value
        if re.search(split_Initial_Expansions, old_key):
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


def temporal_graph_file_parser(file_dir):
    """**Parser for temporal graph description YAML files**
    This function reads the temporal graph description `.yml` file and returns the list of nodes and edges.
    These lists are used to instantiate a :func:`~pyparcs.cdag.graph_objects.Graph` object.

    See Also
    --------

    Parameters
    ----------
    file_dir : str
        directory of the description file.

    Returns
    -------
    nodes : list of dicts
        List of dictionaries whose keys are kwargs of the :func:`~pyparcs.cdag.graph_objects.Node`
        object.
    edges : list of dicts
        List of dictionaries whose keys are kwargs of the :func:`~pyparcs.cdag.graph_objects.Edge`
        object.

    """
    if file_dir is None:
        return [], []
    try:
        file = config_parser(file_dir)
    except Exception as exc:
        raise DescriptionFileError("Error in parsing YAML file.") from exc
    
    resolved_file = dict()

    # overwrite 
    resolved_file['n_timesteps'] = file['n_timesteps']

    for key in file:
        # is edge
        if '->' in key:
            resolved_file = temporal_edge_parser(key, file[key], resolved_file)
        # is node
        else:
            resolved_file = temporal_node_parser(key, file[key], resolved_file)

    # remove n_timesteps  
    del resolved_file['n_timesteps']    

    # TODO not smoothest approach 
    SAVE_PATH = f'tmp_{time.time()}.yaml'
    with open(SAVE_PATH, 'w') as file:
        documents = yaml.dump(resolved_file, file)

    # call parcs graph_file_parser with SAVE_PATH
    nodes, edges =  graph_file_parser(SAVE_PATH)
 
    # remove tmp file 
    os.remove(SAVE_PATH)

    return nodes, edges
            