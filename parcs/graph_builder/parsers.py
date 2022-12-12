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
from parcs.graph_builder.utils import config_parser
from parcs.cdag.output_distributions import DISTRIBUTION_PARAMS
from parcs.cdag.mapping_functions import EDGE_FUNCTIONS, FUNCTION_PARAMS
from parcs.cdag.utils import get_interactions_length, get_interactions_dict


def term_parser(term, vars_):
    pars = []
    for var in vars_:
        res = re.search(r'{}'.format(var), term)
        if res is not None:
            inds = res.span(0)
            pars.append(var)
            term = term[0:inds[0]] + term[inds[1]:]
    if term == '-':
        coef = -1
    elif term == '':
        coef = 1
    else:
        coef = float(term)
    return pars, coef


def equation_parser(eq, vars_):
    # 1. we split by +, so negative terms are + - ...
    eq = eq.replace('-', '+-')
    if eq[0] == '+':
        eq = eq[1:]
    eq = eq.split('+')
    # 2. split by vars
    eq = [term_parser(term, vars_) for term in eq]
    return eq


def node_parser(line, parents):
    # preliminary: get order of interactions
    interactions_dict = get_interactions_dict(parents)
    # remove spaces
    line = line.replace(' ', '')
    # First check: if dist = random
    if line == 'random':
        return {
            'output_distribution': '?',
            'do_correction': True
        }
    # check if data node
    try:
        data_pattern = re.compile(
            r'data\((.*)\)'
        )
        res = data_pattern.search(line)
        return {'csv_dir': res.group(1)}
    except AttributeError:
        # it's not constant
        pass

    # check if node is constant
    try:
        const_pattern = re.compile(
            r'constant\(([0-9]+(\.[0-9]+)?)\)'
        )
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
    # check if node is deterministic
    try:
        det_pattern = re.compile(
            r'deterministic\((.*),(.*)\)'.format('|'.join(DISTRIBUTION_PARAMS.keys()))
        )
        res = det_pattern.search(line)
        directory = res.group(1)
        assert directory[-3:] == '.py'
        directory = directory[:-3]
        function_name = res.group(2)
        function_file = __import__(directory)
        function = getattr(function_file, function_name)
        return {
            'function': function
        }
    except AttributeError:
        # it's not deterministic
        pass

    # find the dist(p1=v1, ...) pattern
    output_params_pattern = re.compile(
        r'({})\((.*)\)'.format('|'.join(DISTRIBUTION_PARAMS.keys()))
    )
    try:
        res = output_params_pattern.search(line)
        dist = res.group(1)
        params = res.group(2)
    except AttributeError:
        raise NameError('distribution "{}" unknown'.format(line.split('(')[0]))
    # split into param - value
    try:
        keys_ = [p.split('=')[0] for p in params.split(',')]
        values_ = [p.split('=')[1] for p in params.split(',')]
    except IndexError:
        # all ?
        assert params == '?'
        keys_ = DISTRIBUTION_PARAMS[dist]
        values_ = ['?'] * len(keys_)

    try:
        assert set(keys_) == set(DISTRIBUTION_PARAMS[dist])
    except AssertionError:
        raise KeyError('params not valid')
    try:
        assert len(set(keys_)) == len(keys_)
    except AssertionError:
        raise KeyError('duplicate params')
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
                ind = interactions_dict.index(set(pars))
                params[p]['interactions'][ind] = coef

    # do correction
    pattern = re.compile(r'correction\[(.*)]')
    res = pattern.search(line)
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
    except IndexError:
        # correction given but with default params
        correction_config = {}
        do_correction = True
    except AttributeError:
        # default values
        pass
    finally:
        return {
            'output_distribution': dist,
            'dist_params_coefs': params,
            'do_correction': do_correction,
            'correction_config': correction_config
        }


def edge_parser(line):
    line = line.replace(' ', '')
    # First check: if dist = ?
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
        raise NameError('edge function "{}" unknown'.format(line.split('(')[0]))

    # split into param - value
    try:
        assert params != '?'
        func_params = {}
        for p in params.split(','):
            try:
                func_params[p.split('=')[0]] = float(p.split('=')[1])
            except ValueError:
                func_params[p.split('=')[0]] = p.split('=')[1]
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
    These lists are used to instantiate a :func:`~parcs.cdag.graph_objects.Graph` object.

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

    file = config_parser(file_dir)
    # edges
    edges = [{
        'name': e,
        **edge_parser(file[e])
    } for e in file if '->' in e]
    # node list
    node_list = [n for n in file if '->' not in n]
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
    return config_parser(file_dir)


if __name__ == '__main__':
    obj = node_parser('gaussian(mu_=?, sigma_=1)', ['A', 'B', 'C'])
    print(obj)
    # obj = edge_parser('sigmoid(alpha=2.0, beta=1.8), correct[]')
    # graph_file_parser('../../graph_templates/causal_triangle.yml')
