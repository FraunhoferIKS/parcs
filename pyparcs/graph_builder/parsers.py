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
import numpy as np
from pyparcs.graph_builder.utils import config_parser
from pyparcs.cdag.output_distributions import DISTRIBUTION_PARAMS
from pyparcs.cdag.mapping_functions import EDGE_FUNCTIONS, FUNCTION_PARAMS
from pyparcs.cdag.utils import get_interactions_length, get_interactions_dict
from pyparcs.exceptions import *
from typeguard import typechecked
from typing import List, Tuple


@typechecked
def term_parser(term: str, vars_: List[str]) -> Tuple[list, float]:
    """
    parses the terms (between plus signs). gives list of parents and the multiplicative coef of the term

    Parameters
    ----------
    term : str
        the term to be processed
    vars_ : list(str)
        the list of possible parents to extract from the term

    Returns
    -------
    pars : list(str)
        list of present parents in the term
    coef : float
        the multiplicative coefficient

    Raises
    ------
    DescriptionFileError
        If the terms are invalid. Cases: non-existing parents, terms other than bias, linear, interactions, etc.
    """
    err_msg = '''The term {} in the description file is invalid.
    Another possibility is that a node name exists in another node\'s parameter, while it is not marked as
    a parent by the described edges; in this case, check for the nodes/edges consistency.
    '''.format(term)  # for potential errors
    pars = []
    for var in vars_:
        # check if X^2 exists:
        res = re.search(r'{}\^2'.format(var), term)
        if res is not None:  # a term like 'X^2' exists
            parcs_assert(len(pars) == 0, DescriptionFileError, err_msg)  # quad parent must be the only parent
            inds = res.span(0)
            parcs_assert(len(term) == inds[1], DescriptionFileError, err_msg)  # '^2' must be the last char
            pars = [var, var]
            term = term[0:inds[0]] + term[inds[1]:]
            break  # do not check other variables because we cannot have 'YX^2'
        # it's not quadratic. check other possibilities
        res = re.search(r'{}'.format(var), term)
        if res is not None:
            inds = res.span(0)
            pars.append(var)
            term = term[0:inds[0]] + term[inds[1]:]
    if term == '-':  # something like '-X'
        coef = -1
    elif term == '':  # something like 'X'
        coef = 1
    else:
        try:
            coef = float(term)
        except ValueError:
            raise DescriptionFileError(err_msg)  # there are other chars apart from quad term and coef.
    return pars, coef


def equation_parser(eq: str, vars_: List[str]) -> List[Tuple[List[str], float]]:
    """
    parses the equation strings as lists of terms. each term includes tuples of parent lists and coefficient.
    Parameters
    ----------
    eq : str
        the equation string e.g. '1 + 2X_1 - X_2^2'
    vars_ : list(str)
        list of potential variables in the equation

    Returns
    -------
    parsed equation : list(tuple(list(str), float))
        list of parsed terms. each term is a tuple of parent list and the coefficient float
    """
    # 0. remove spaces
    eq = eq.replace(' ', '')
    # 1. we split by +, so negative terms are + - ...
    eq = eq.replace('-', '+-')
    if eq[0] == '+':
        eq = eq[1:]
    eq = eq.split('+')
    # 2. split by vars
    eq = [term_parser(term, vars_) for term in eq]
    # 3. check for duplicates
    parents = [frozenset(e[0]) for e in eq]
    parcs_assert(len(set(parents)) == len(parents),
                 DescriptionFileError, "Duplicated terms exist in equation {}".format(eq))

    return eq


@typechecked
def node_parser(line: str, parents: List[str]) -> dict:
    """
    Parses a line in graph description file, and gives the appropriate dictionary to initiate a node. Keys are
    - for stochastic node: 'output_distribution', 'dist_params_coefs', 'do_correction', 'correction_config'
    - for constant node: 'value'
    - for data node: 'csv_dir'
    - for deterministic node: 'function'

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

    """
    # 0. preliminaries
    interactions_dict = get_interactions_dict(parents)  # get order of interactions
    line = line.replace(' ', '')  # remove spaces

    data_pattern = re.compile(r'data\((.*)\)')  # data node regex pattern
    const_pattern = re.compile(r'constant\((-?[0-9]+(\.[0-9]+)?)\)')  # constant node regex pattern
    det_pattern = re.compile(
        r'deterministic\((.*),(.*)\)'.format('|'.join(DISTRIBUTION_PARAMS.keys())))  # deterministic node regex pattern
    stoch_pattern = re.compile(
        r'({})\((.*)\)'.format('|'.join(DISTRIBUTION_PARAMS.keys())))  # stochastic node regex pattern
    correction_pattern = re.compile(r'correction\[(.*)]')  # correction regex pattern, only for stochastic

    # 1. random node (stochastic)
    if line == 'random':
        return {
            'output_distribution': '?',
            'do_correction': True
        }

    # 2. data node
    try:
        res = data_pattern.search(line)
        dir_ = res.group(1)
        # IF "dir_" is not null, then next pyparcs assertion must be made
        parcs_assert(
            parents == [],
            DescriptionFileError,
            "data nodes cannot have parents."
        )
        return {'csv_dir': dir_}
    except AttributeError:
        # it's not constant
        pass

    # 3. constant node
    try:
        res = const_pattern.search(line)
        raw_value = res.group(1)
        if len(res.groups()) == 2:
            return {'value': float(raw_value)}
        elif len(res.groups()) == 1:  # doesn't have floating part
            return {'value': int(raw_value)}
        else:
            raise AttributeError
    except AttributeError:
        # it's not constant
        pass

    # 4. deterministic node
    try:
        res = det_pattern.search(line)
        directory = res.group(1)
        assert directory[-3:] == '.py'
        directory = directory[:-3]
        function_name = res.group(2)
        try:
            function_file = __import__(directory)
        except ModuleNotFoundError:
            raise ExternalResourceError('Python script {} containing the function does not exist.'.format(directory))
        try:
            function = getattr(function_file, function_name)
        except AttributeError:
            raise ExternalResourceError('Python function {} not existing in script {}'.format(function_name, directory))
        return {
            'function': function
        }
    except AttributeError:
        # it's not deterministic
        pass

    # 5. stochastic (non "random") nodes
    try:
        res = stoch_pattern.search(line)
        dist = res.group(1)
        params = res.group(2)
    except AttributeError:
        raise DescriptionFileError(
            '''description line '{}' cannot be parsed. check the following:
                - distribution names
                - input arguments for a node type
                - unwanted invalid characters
            '''.format(line)
        )
    # split into param - value
    try:
        keys_ = [p.split('=')[0] for p in params.split(',')]
        values_ = [p.split('=')[1] for p in params.split(',')]
    except IndexError:
        # all ?
        assert params == '?'
        keys_ = DISTRIBUTION_PARAMS[dist]
        values_ = ['?'] * len(keys_)

    parcs_assert(
        set(keys_) == set(DISTRIBUTION_PARAMS[dist]),
        DescriptionFileError,
        "params {} not valid for distribution '{}'".format(set(keys_), dist)
    )
    parcs_assert(
        len(set(keys_)) == len(keys_),
        DescriptionFileError,
        "duplicate params for distribution '{}'".format(dist)
    )
    params = {
        k: v for k, v in zip(keys_, values_)
    }

    # process params
    for p in params:
        if params[p] == '?':
            params[p] = {
                'bias': '?', 'linear': '?', 'interactions': '?'
            }
            continue
        terms = equation_parser(params[p], parents)
        params[p] = {
            'bias': 0,
            'linear': np.zeros(shape=(len(parents, ))),
            'interactions': np.zeros(shape=get_interactions_length(len(parents)), )
        }

        for term in terms:
            pars, coef = term
            if len(pars) == 0:
                params[p]['bias'] = coef
            elif len(pars) == 1:
                ind = parents.index(pars[0])
                params[p]['linear'][ind] = coef
            else:
                ind = interactions_dict.index(sorted(pars))
                params[p]['interactions'][ind] = coef

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
    except ValueError:  # the values are not float
        raise DescriptionFileError("correction params are invalid: {}".format(res))

    return {
        'output_distribution': dist,
        'dist_params_coefs': params,
        'do_correction': do_correction,
        'correction_config': correction_config
    }


def edge_parser(line):
    line = line.replace(' ', '')
    # 1. function is "?"
    if line == 'random':
        return {
            'function_name': '?',
            'do_correction': True
        }

    # find the func(p1=v1, ...) pattern
    output_params_pattern = re.compile(
        r'({})\((.*)\)'.format('|'.join(EDGE_FUNCTIONS.keys()))
    )
    try:
        res = output_params_pattern.search(line)
        func = res.group(1)
        params = res.group(2)
    except AttributeError:
        raise DescriptionFileError("edge function '{}' unknown".format(line.split('(')[0]))

    # split into param - value
    try:
        assert params != '?'
        func_params = {}
        for p in params.split(','):
            try:
                func_params[p.split('=')[0]] = float(p.split('=')[1])
            except ValueError:
                # only ? is allowed as non float
                parcs_assert(p.split('=')[1] == '?', DescriptionFileError, "wrong param input: {}".format(line))
                func_params[p.split('=')[0]] = '?'
        parcs_assert(
            set(func_params.keys()) == set(FUNCTION_PARAMS[func]),
            DescriptionFileError,
            '''
            edge function params are invalid or incomplete. if you want to leave a parameter random,
            specify it explicitly by param=?.
            faulty description file line is: {}
            '''.format(line)
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
        'do_correction': do_correction
    }


def graph_file_parser(file_dir):
    """**Parser for graph description YAML files**
    This function reads the graph description ``.yml`` files and returns the list of nodes and edges.
    These lists are used to instantiate a :func:`~pyparcs.cdag.graph_objects.Graph` object.

    See Also
    --------

    Parameters
    ----------
    file_dir : str
        directory of the description file.

    Returns
    -------

    """
    # if file is empty
    if file_dir is None:
        return [], []
    try:
        file = config_parser(file_dir)
    except Exception as e:
        raise DescriptionFileError("Error in parsing YAML file. The native error from yaml package: {}".format(e))
    # edges
    edges = [{
        'name': e,
        **edge_parser(file[e])
    } for e in file if '->' in e]
    # node list
    node_list = [n for n in file if '->' not in n]

    # PARCS asserts:
    # 0. names are standard
    name_standard = r'^[a-zA-Z](?:[a-zA-Z0-9_]*[a-zA-Z0-9])?$'
    parcs_assert(
        all(re.match(name_standard, node_name) for node_name in node_list),
        DescriptionFileError,
        "One or more node names does not follow the PARCS naming conventions. Please see the docs."
    )
    # 1. node in edges are also in node list
    edge_names = [e['name'] for e in edges]
    node_in_edge = set([i for element in edge_names for i in element.split('->')])
    parcs_assert(
        node_in_edge.issubset(set(node_list)),
        DescriptionFileError,
        "A parent/child node in the edge list does not exist in node lines."
    )

    parent_dict = {
        node: sorted([
            e['name'].split('->')[0] for e in edges
            if e['name'].split('->')[1] == node
        ])
        for node in node_list
    }
    # nodes
    nodes = [{
        'name': n,
        **node_parser(file[n], parent_dict[n])
    } for n in file if '->' not in n]

    return nodes, edges


def guideline_parser(file_dir):
    try:
        return config_parser(file_dir)
    except Exception as e:
        raise GuidelineError("Error in parsing YAML file. The native error from yaml package: {}".format(e))
