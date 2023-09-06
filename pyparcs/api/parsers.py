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

import re
import sys
from typing import List, Tuple, Union
import numpy as np
from pyparcs.api.output_distributions import DISTRIBUTION_PARAMS
from pyparcs.api.mapping_functions import EDGE_FUNCTIONS, FUNCTION_PARAMS
from pyparcs.api.utils import get_interactions_length, get_interactions_names
from pyparcs.core.exceptions import (parcs_assert, DescriptionError, ExternalResourceError)


def augment_line(line: str, addition: str, limit: bool = False) -> Tuple[bool, str]:
    """Augments line for receiving extra edges

    This functions takes the node line and the additional terms to add to the line
    and returns the augmented line. This function is used when a defined line needs
    to receive new parents.

    Parameters
    ----------
    line: str
        the node outline line
    addition: str
        The extra term to be added
    limit: bool, default=False
        If `True`, the line is augmented only if the parameter is not limited
        by the trailing ``!`` character behind the parameter name

    Returns
    -------
    is_augmented: bool
        If the line is augmented (because it could be limited)
    augmented_line: str
        The new line, augmented with the addition term

    """
    line = line.replace(' ', '')

    join_param_keys = '|'.join(DISTRIBUTION_PARAMS.keys())
    stoch_pattern = re.compile(fr'({join_param_keys})\((.*)\)')  # stochastic node regex pattern
    try:
        res = stoch_pattern.search(line)
        params = res.group(2)
    except AttributeError:  # it's not stochastic or it is random
        return False, line
    # based on the returned params, find the index of the last character
    # because we want to add new terms at that index
    # reversed by [::-1] because if we add from beginning, indices get shifted
    add_index = [re.search(item.replace('?', '\?').
                           replace('+', '\+').
                           replace('^', '\^'), line).span()[1]
                 for item in params.split(',')][::-1]
    limited = ['!' in item for item in params.split(',')][::-1]
    is_added = False
    for ind, is_limited in zip(add_index, limited):
        if not (limit and is_limited):
            is_added = True
            line = line[:ind] + f'+{addition}' + line[ind:]
    return is_added, line.replace('!', '')


def term_parser(term: str, possible_parents: List[str]) -> Tuple[list, Union[float, str]]:
    """
    parses the terms (between plus signs). gives list of parents and
    the multiplicative coef of the term

    Parameters
    ----------
    term : str
        the term to be processed
    possible_parents : list(str)
        the list of possible parents to extract from the term

    Returns
    -------
    term_parents : list(str)
        list of present parents in the term
    coef : float
        the multiplicative coefficient

    Raises
    ------
    DescriptionError
        If the terms are invalid. Cases: non-existing parents, terms other than bias, linear,
        interactions, etc.
    """
    # check that the term is either a bias term, or ends with a parents name
    # try:
    #     coef = float(term)
    #     return [], coef
    # except ValueError:
    #     pattern = rf'(?:{"|".join(re.escape(sub) for sub in possible_parents)}$'
    #     parcs_assert(re.search(pattern, term),
    #                  DescriptionError,
    #                  f'Error in a term: unknown substring in {term}'
    #                  'The term ends with a number which is not in any parent name. '
    #                  'If it is a coefficient, they must be put always at the beginning'
    #                  'of the term')

    # check if the term has the "<text><underscore><number>" in the end
    # it suppresses the correct pattern for A^2
    digit_match = re.search(r'[A-Za-z]+_*(\d+)$', term)
    # either the term doesn't end with that pattern, or if it does, it's a parent name
    parcs_assert(not digit_match or any([term.endswith(p) for p in possible_parents]),
                 DescriptionError,
                 f'Unknown ending in {term}: '
                 'The term ends with a number which is not in any parent name, nor a power. '
                 'If it is a coefficient, they must be put always at the beginning'
                 'of the term')

    err_msg = (f"The term {term} in the description file is invalid. Another possibility "
               "is that a node name exists in another node\'s parameter, while it is "
               "not marked as a parent by the described edges; in this case, "
               "check for the nodes/edges consistency.")  # for potential errors
    term_parents = []
    # sort possible parents by their length, to solve the risk of inclusive names: B, B_1
    for var in sorted(possible_parents, key=len, reverse=True):
        # check if X^2 exists:
        res = re.search(rf'{var}\^2', term)
        if res:  # a term like 'X^2' exists
            # quad parent must be the only parent
            parcs_assert(len(term_parents) == 0, DescriptionError, err_msg)
            inds = res.span(0)
            # '^2' must be the last char
            parcs_assert(len(term) == inds[1], DescriptionError, err_msg)
            term_parents = [var, var]
            term = term[0:inds[0]] + term[inds[1]:]
            break  # do not check other variables because we cannot have 'YX^2'

        # it's not quadratic. check other possibilities
        res = re.search(fr'{var}(?!\d+)', term)
        if res:
            inds = res.span(0)
            term_parents.append(var)
            term = term[0:inds[0]] + term[inds[1]:]
    # what remains is the coefficient
    non_number_possibilities = {
        '-': -1,  # something like '-X'
        '': 1,  # something like 'X'
        '?': '?'  # randomizing the coefficient
    }
    if term in non_number_possibilities:
        coef = non_number_possibilities[term]
    else:
        try:
            coef = float(term)
        except ValueError as exc:
            # there are other chars apart from quad term and coef.
            raise DescriptionError(err_msg) from exc
    return term_parents, coef


def equation_parser(equation: str, vars_: List[str]) -> List[Tuple[List[str], Union[float, str]]]:
    """
    parses the equation strings as lists of terms. each term includes tuples of parent lists
    and coefficient.

    Parameters
    ----------
    equation : str
        the equation string e.g. '1 + 2X_1 - X_2^2'
    vars_ : list(str)
        list of potential variables in the equation

    Returns
    -------
    parsed equation : list(tuple(list(str), float))
        list of parsed terms. each term is a tuple of parent list and the coefficient float or '?'
    """
    # 0. remove spaces
    equation = equation.replace(' ', '')
    # 1. we split by +, so negative terms are + - ...
    equation = equation.replace('-', '+-')
    if equation[0] == '+':
        equation = equation[1:]
    equation = equation.split('+')
    # 2. split by vars
    equation = [term_parser(term, vars_) for term in equation]
    # 3. check for duplicates
    parents = [tuple(sorted(e[0])) for e in equation]
    parcs_assert(len(set(parents)) == len(parents),
                 DescriptionError, f"Duplicated terms exist in equation {equation}")

    return equation


def tags_parser(line: str) -> list:
    """Parses tag keywords in edge and node lines"""
    tag_pattern = re.compile(r'tags\[(.*)]')
    # 0-1 finding tags
    res = tag_pattern.search(line.replace(' ', ''))
    # default values
    try:
        tags = res.group(1).split(',')
        parcs_assert('' not in tags,
                     DescriptionError,
                     'empty tag not allowed. '
                     'empty tag might have been generated due to comma separation.\n'
                     f'bad tag is: `tags[{res.group(1)}]`')
        return tags
    except AttributeError:  # no tag
        return []


def detect_parse_random_node(line: str) -> Tuple[bool, dict]:
    """detects and (if is random) returns the parsed config dict"""
    # with or without tag... correction is not allowed
    if line == 'random' or line[:12] == 'random,tags[':
        return True, {'output_distribution': '?', 'do_correction': True, 'correction_config': {}}
    return False, {}


def detect_parse_data_node(line: str, parents: list) -> Tuple[bool, dict]:
    """detects and (if is data node) returns the parsed config dict"""
    # data node regex pattern
    data_pattern = re.compile(r'data\((.*)\)')
    try:
        res = data_pattern.search(line)
        dir_, col = res.group(1).split(',')
        # IF "dir_" is not null, then next pyparcs assertion must be made
        parcs_assert(parents == [],
                     DescriptionError,
                     "data nodes cannot have parents.")
        return True, {'csv_dir': dir_, 'col': col}
    except AttributeError:
        # it's not data node
        pass
    return False, {}


def detect_parse_constant_node(line: str) -> Tuple[bool, dict]:
    """detects and (if is constant node) returns the parsed config dict"""
    # constant node regex pattern
    const_pattern = re.compile(r'constant\((-?[0-9]+(\.[0-9]+)?)\)')
    try:
        res = const_pattern.search(line)
        raw_value = res.group(1)
        if len(res.groups()) == 2:
            return True, {'value': float(raw_value)}
        if len(res.groups()) == 1:  # doesn't have floating part
            return True, {'value': int(raw_value)}
        raise AttributeError
    except AttributeError:
        # it's not constant
        pass
    return False, {}


def detect_parse_deterministic_node(line: str) -> Tuple[bool, dict]:
    """detects and (if is deterministic node) returns the parsed config dict

    Raises
    ------
    ExternalResourceError
        If there is a problem reading the python script of function
    """
    # deterministic node regex pattern
    det_pattern = re.compile(r'deterministic\((.*),(.*)\)')
    try:
        res = det_pattern.search(line)
        directory = res.group(1)
        parcs_assert(
            directory[-3:] == '.py',
            ExternalResourceError,
            'module directory must end with the module name, i.e. with .py '
        )
        if '/' in directory:
            # strip the module name which is the last element after the last '/'
            path_to_module = '/'.join(directory.split('/')[:-1])
            sys.path.append(path_to_module)
            # update directory for the next step, remaining only the module name
            directory = directory.split('/')[-1]
        # strip .py extension
        directory = directory[:-3]
        function_name = res.group(2)
        try:
            function_file = __import__(directory)
        except ModuleNotFoundError as exc:
            raise ExternalResourceError(f'Python script {directory} containing the function does '
                                        f'not exist.') from exc
        try:
            function = getattr(function_file, function_name)
        except AttributeError as exc:
            raise ExternalResourceError(f'Python function {function_name} not existing in script '
                                        f'{directory}') from exc
        return True, {'function': function}
    except AttributeError:
        # it's not deterministic
        pass
    return False, {}


def detect_parse_stochastic_node(line: str, parents: list) -> Tuple[bool, dict]:
    """detects and (if is stochastic node) returns the parsed config dict
    Raises
    ------
    DescriptionError
        if the parameters of the stochastic node are not written correctly
    """
    interactions_dict = get_interactions_names(parents)  # get order of interactions
    join_param_keys = '|'.join(DISTRIBUTION_PARAMS.keys())
    stoch_pattern = re.compile(fr'({join_param_keys})\((.*)\)')  # stochastic node regex pattern
    correction_pattern = re.compile(r'correction\[(.*)]')  # correction regex pattern
    try:
        res = stoch_pattern.search(line)
        dist = res.group(1)
        params = res.group(2)
    except AttributeError:
        return False, {}

    # split into param - value
    try:
        keys_ = [p.split('=')[0] for p in params.split(',')]
        values_ = [p.split('=')[1] for p in params.split(',')]
    except IndexError:
        # all ?
        parcs_assert(
            params == '?',
            DescriptionError,
            (f'Error in parameter "({params})" for the distribution "{dist}". '
             'The convention to parameterize a distribution is: '
             'dist(p1=..., p2=..., ...) even for distributions with 1 parameter, '
             'unless the parameters are all randomized via question mark: "dist(?)"')
        )
        keys_ = DISTRIBUTION_PARAMS[dist]
        # dist(?) -> dist(p1_=?, p2_=?, ...)
        values_ = ['?'] * len(keys_)

    parcs_assert(
        set(keys_) == set(DISTRIBUTION_PARAMS[dist]),
        DescriptionError,
        f"params {set(keys_)} not valid for distribution '{dist}'"
    )
    parcs_assert(
        len(set(keys_)) == len(keys_),
        DescriptionError,
        f"duplicate params for distribution '{dist}'"
    )
    params = dict(zip(keys_, values_))

    # process params
    for param in params:
        if params[param] == '?':
            params[param] = {
                'bias': '?', 'linear': '?', 'interactions': '?'
            }
            continue
        terms = equation_parser(params[param], parents)
        params[param] = {
            'bias': 0,
            'linear': [0] * len(parents),
            'interactions': [0] * get_interactions_length(len(parents))
        }

        for term in terms:
            term_parents, coef = term
            if len(term_parents) == 0:
                params[param]['bias'] = coef
            elif len(term_parents) == 1:
                ind = parents.index(term_parents[0])
                params[param]['linear'][ind] = coef
            else:
                ind = interactions_dict.index(sorted(term_parents))
                params[param]['interactions'][ind] = coef

    # do correction
    res = correction_pattern.search(line)
    # default values
    do_correction = False
    correction_config = {}
    try:
        corrs = res.group(1)
        correction_config = {
            conf.split('=')[0]: float(conf.split('=')[1])
            for conf in corrs.split(',')
        }
        do_correction = True
    except IndexError:  # correction given but with default params
        correction_config = {}
        do_correction = True
    except AttributeError:  # default values
        pass
    except ValueError as exc:  # the values are not float
        raise DescriptionError(f"correction params are invalid: {res}") from exc

    return True, {
        'output_distribution': dist,
        'dist_params_coefs': params,
        'do_correction': do_correction,
        'correction_config': correction_config
    }


def node_parser(line: str, parents: List[str]) -> dict:
    """
    Parses a line in description file, and gives the appropriate dictionary to initiate a node.
    Keys are:
    - stochastic node: 'output_distribution', 'dist_params_coefs', 'do_correction',
    'correction_config'
    - constant node: 'value'
    - data node: 'csv_dir'
    - deterministic node: 'function'

    Parameters
    ----------
    line : str
        one line of the graph description file
    parents : list(str)
        list of parents of the corresponding node

    Returns
    -------
    node config : dict
        depending on the node type, returns the dict to initiate the node. see main description.

    Raises
    ------
    DescriptionError
        If a line cannot be parsed as one of the possible node types
    """
    # 0. preliminaries
    line = line.replace(' ', '')  # remove spaces
    tags = tags_parser(line)
    # remove tag substring
    line = re.sub(r',tags\[[^\]]+\]', '', line)

    # 1. random node (stochastic)
    is_random, node_config = detect_parse_random_node(line)
    if is_random:
        return {**node_config, 'node_type': 'stochastic', 'tags': tags}
    # 2. data node
    is_data, node_config = detect_parse_data_node(line, parents)
    if is_data:
        return {**node_config, 'node_type': 'data', 'tags': tags}
    # 3. constant node
    is_constant, node_config = detect_parse_constant_node(line)
    if is_constant:
        return {**node_config, 'node_type': 'constant', 'tags': tags}
    # 4. deterministic node
    is_deterministic, node_config = detect_parse_deterministic_node(line)
    if is_deterministic:
        return {**node_config, 'node_type': 'deterministic', 'tags': tags}
    # 5. stochastic (non "random") nodes
    is_stochastic, node_config = detect_parse_stochastic_node(line, parents)
    if is_stochastic:
        return {**node_config, 'node_type': 'stochastic', 'tags': tags}
    # 6. if none
    raise DescriptionError(
        f'''description line '{line}' cannot be parsed. check the following:
            - distribution names
            - input arguments for a node type
            - unwanted invalid characters
        ''')


def edge_parser(line: str) -> dict:
    """Parses edge lines

    parses the lines for `X->Y` keys. potential values are `random` plus available
    edge functions. `correction[]` and `tags[]` are allowed, all comma separated

    Raises
    ------
    DescriptionError
        If an edge line cannot be parsed because of unknown function or bad parameter
    """
    line = line.replace(' ', '')
    tags = tags_parser(line)
    # 1. function is "?"
    if line == 'random' or line[:11] == 'random,tags[':
        return {
            'function_name': '?',
            'do_correction': True,
            'tags': tags
        }

    # find the func(p1=core, ...) pattern
    edge_func_keys = '|'.join(EDGE_FUNCTIONS.keys())
    output_params_pattern = re.compile(rf'({edge_func_keys})\((.*)\)')
    try:
        res = output_params_pattern.search(line)
        func = res.group(1)
        params = res.group(2)
    except AttributeError as exc:
        func = line.split('(')[0]
        raise DescriptionError(f"edge function '{func}' unknown") from exc

    # split into param - value
    try:
        assert params != '?'
        func_params = {}
        for e_p in params.split(','):
            try:
                func_params[e_p.split('=')[0]] = float(e_p.split('=')[1])
            except ValueError:
                # only ? is allowed as non float
                parcs_assert(e_p.split('=')[1] == '?', DescriptionError,
                             f"wrong param input: {line}")
                func_params[e_p.split('=')[0]] = '?'
        parcs_assert(
            set(func_params.keys()) == set(FUNCTION_PARAMS[func]),
            DescriptionError,
            f'''
            edge function params are invalid or incomplete. if you want to leave a parameter random,
            specify it explicitly by param=?.
            faulty description file line is: {line}
            '''
        )
    except IndexError:
        # function has no params
        func_params = {}
    except AssertionError:
        # all ?
        func_params = {
            k: '?' for k in FUNCTION_PARAMS[func]
        }
    # do correction:
    correct = re.compile(r'correction\[]')
    res = correct.findall(line)
    if len(res) == 0:
        do_correction = False
    else:
        do_correction = True

    return {
        'function_name': func,
        'function_params': func_params,
        'do_correction': do_correction,
        'tags': tags
    }


def description_parser(desc_dict: dict, infer_edges: bool = False) -> Tuple[dict, dict]:
    """**Parser for graph description dictionaries**
    This function reads a description object and returns the list of nodes and edges.
    These lists are used to instantiate a :func:`~pyparcs.cdag.graph_objects.Graph` object.

    See Also
    --------

    Parameters
    ----------
    desc_dict: dict
        a dictionary of nodes and edges description
    infer_edges: bool, default=False
        If true, then the missing edges are inferred and added to the list

    Returns
    -------
    nodes : dict
        `{name: node_spec ...}`
    edges : dict
        `{edge:edge_spec ...}`

    Raises
    ------
    DescriptionError
        If the namings are wrong or node/edge parents are not consistent
    """
    nodes_sublist, edges_sublist = outline_splitter(desc_dict)
    if infer_edges:
        edges_sublist = infer_missing_edges(nodes_sublist, edges_sublist)

    # == PARCS ASSERTS ==
    # 0. names are standard
    name_standard = r'^[a-zA-Z](?:[a-zA-Z0-9_]*[a-zA-Z0-9])?$'
    parcs_assert(
        all(re.match(name_standard, node_name) for node_name in nodes_sublist),
        DescriptionError,
        "One or more node names does not follow the PARCS naming conventions. Please see the docs."
    )
    # 1. node in edges are also in node list
    node_in_edge = {i for element in edges_sublist.keys() for i in element.split('->')}
    parcs_assert(
        node_in_edge.issubset(set(nodes_sublist.keys())),
        DescriptionError,
        "A parent/child node in the edge list does not exist in node lines."
    )

    edges = {e: {**edge_parser(specs)} for (e, specs) in edges_sublist.items()}

    parent_dict = {
        node: sorted([
            e.split('->')[0] for e in edges_sublist
            if e.split('->')[1] == node
        ])
        for node in nodes_sublist
    }
    nodes = {n: {**node_parser(specs, parent_dict[n])} for (n, specs) in nodes_sublist.items()}

    return nodes, edges


def infer_missing_edges(nodes_sublist: dict, edges_sublist: dict) -> dict:
    """Infers missing edges in the description file

    The method is to search the "string" name of the potential
    parents in the "string" description line of the nodes,
    and check if an edge exists in case of a finding.
    If not, it adds one

    Parameters
    ----------
    nodes_sublist : dict
        the format of a key-value pair is: `'A': 'normal(...)'`,
        which means that the values has not been parsed
    edges_sublist : dict
        the format of a key-value pair is: `'A->B': 'sigmoid(...)'`,
        which means that the values has not been parsed

    Returns
    -------
    augmented edge_sublist
    """
    # extracts list of nodes that appear in the params of the nodes
    # the following dict comprehension, checks the existence of the node name
    # by searching in the string line (it happens at: `if parent in nodes[node]`)
    existing_nodes = {}
    for node, arg in nodes_sublist.items():
        arg_split = re.split(r'[()]', arg)
        # what is between the parenthesis unless it is `random`
        arg = arg_split[1] if len(arg_split) == 3 else ''

        # avoid finding the wrong name which is the substring of the actual name in the node params
        existing_nodes[node] = []
        for parent in sorted(nodes_sublist.keys(), key=len, reverse=True):
            if parent in arg:
                existing_nodes[node].append(parent)
                arg = arg.replace(parent, '')

    # add missing edges
    for node in nodes_sublist:
        for parent in existing_nodes[node]:
            if f'{parent}->{node}' not in edges_sublist.keys():
                edges_sublist[f'{parent}->{node}'] = 'identity()'

    return edges_sublist


def outline_splitter(graph_dict: dict) -> Tuple[dict, dict]:
    """splits the desc dictionary kv pairs into nodes and edges (does not parse)"""
    edges_sublist = {k: v for k, v in graph_dict.items() if '->' in k}
    nodes_sublist = {k: v for k, v in graph_dict.items() if k not in edges_sublist}

    return nodes_sublist, edges_sublist


def is_partial(nodes: dict, edges: dict) -> bool:
    """
    Checks if the passed nodes and edges lists need randomization. True if they need.
    """
    if '?' in str(nodes) + str(edges) or 'random' in str(nodes) + str(edges):
        return True
    return False


def term_synthesizer(parents: List[str], coef: float, round_digit: int = 2) -> str:
    """
    converts term parser output back to an outline term

    Parameters
    ----------
    parents : list(str)
        the term to be processed
    coef : float
        the list of possible parents to extract from the term
    round_digit: int
        decimal for rounding

    Returns
    -------
    term: str
    """
    if coef == 0:
        return ''

    coef = (
        '?' if coef == '?' else
        '' if coef == 1 and len(parents) != 0 else
        '-' if coef == -1 and len(parents) != 0 else
        np.round(coef, round_digit)
    )

    if len(set(parents)) == len(parents) - 1:  # A^2
        return f'{coef}{parents[0]}^2'
    else:  # other types including bias, 1 or more parents
        return f'{coef}{"".join(parents)}'


def equation_synthesizer(parsed_equation: List[Tuple[list, float]]) -> str:
    """
    converts equation parser output back to an outline equation

    Parameters
    ----------
    parsed_equation: list of tuple(list(str), float/str)

    Returns
    -------
    outline_equation: str

    """
    terms = [term_synthesizer(term[0], term[1]) for term in parsed_equation]
    terms = list(filter(None, terms))

    term_string = '+'.join(terms).replace('+-', '-')
    if term_string == '':
        return str(0.0)
    else:
        return term_string


def stochastic_node_synthesizer(node: dict, parents: list, tags: List[str]) -> str:
    """
    converts the stochastic node info dict to a line.

    Parameters
    ----------
    node: dict
        see ref:`graph_objects_schema`
    parents:
        sorted list of parents
    tags: list(str)
        list of node tags

    Returns
    -------
    node_line: str
    """
    if node['output_distribution'] == '?':
        return 'random'
    # make dist params
    args = []
    for param, coefs in node['dist_params_coefs'].items():
        bias_terms = [([], coefs['bias'])]
        linear_terms = [
            ([parent], coef) for parent, coef in zip(parents, coefs['linear'])
        ]
        interactions_terms = [
            (pair, coef)
            for pair, coef in zip(get_interactions_names(parents), coefs['interactions'])
        ]
        equation = equation_synthesizer(bias_terms + linear_terms + interactions_terms)
        args.append(f'{param}={equation}')
    # make correction
    if node['do_correction']:
        # the way we read correction_config relies on the fact that currently we only
        # have maximum one dist parameter for correction
        # the logic must change if distributions have more params to correct
        config = [f'{k}={v}' for k, v in node['correction_config'].items()]
        correction = f'correction[{", ".join(config)}]'
    else:
        correction = ''

    # make the line
    line = f"{node['output_distribution']}({', '.join(args)})"
    if node['do_correction']:
        line += f", {correction}"
    if len(tags) != 0:
        line += f", tags[{', '.join(tags)}]"
    return line


def edge_synthesizer(edge: dict, tags: List[str]) -> str:
    """
    converts edge dict to an edge outline line

    Parameters
    ----------
    edge: dict
        see ref:`graph_objects_schema`
    tags: list(str)
        list of edge tags

    Returns
    -------
    synthesized_line: str
    """
    params = [f'{k}={v}' for k, v in edge['function_params'].items()]
    line = f"{edge['function_name']}({', '.join(params)})"
    if edge['do_correction']:
        line += f", correction[]"
    if len(tags) != 0:
        line += f", tags[{', '.join(tags)}]"

    return line
