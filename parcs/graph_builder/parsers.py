import re
import numpy as np
from pprint import pprint
from parcs.graph_builder.utils import config_parser
from parcs.cdag.output_distributions import DISTRIBUTION_PARAMS
from parcs.cdag.mapping_functions import EDGE_FUNCTIONS, FUNCTION_PARAMS
from parcs.cdag.utils import get_interactions_length, get_interactions_dict


def node_parser(line, parents):
    # preliminary: get order of interactions
    # remove spaces
    line = line.replace(' ', '')
    # First check: if dist = ?
    if line == 'free':
        return {
            'output_distribution': '?',
            'do_correction': True
        }

    # find the dist(p1=v1, ...) pattern
    output_params_pattern = re.compile(
        '({})\((.*)\)'.format('|'.join(DISTRIBUTION_PARAMS.keys()))
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
        values_ = ['?']*len(keys_)

    try:
        flag = 1
        assert set(keys_) == set(DISTRIBUTION_PARAMS[dist])
        flag = 2
        assert len(set(keys_)) == len(keys_)
    except AssertionError:
        if flag == 1: raise KeyError('params not valid')
        else: raise KeyError('duplicate params')

    params = {
        k: v for k, v in zip(keys_, values_)
    }
    # process params
    for p in params:
        # add leading + to negatives, to split with +
        eq = params[p].replace('-', '+-')
        if eq[0] == '+':
            eq = eq[1:]
        # resolve negative coefs -A, -2B, ... at the beginning
        for par in parents:
            coef = re.findall(r'-([0-9]*){}'.format(par), eq)
            if len(coef) == 0:
                # no instance
                continue
            else:
                if coef[0] == '':
                    # -C, -A, ...
                    eq = eq.replace('-{}'.format(par), '-1*{}'.format(par))
                else:
                    eq = eq.replace('-{}{}'.format(coef[0], par), '-{}*{}'.format(coef[0], par))
        terms = re.split(r'[+]', eq)
        if len(terms) == 1:
            if terms[0] == '?':
                # aim to randomize
                params[p] = {'bias': '?', 'linear': '?', 'interactions': '?'}
            else:
                # single number: must be bias
                params[p] = {
                    'bias': float(terms[0]),
                    'linear': np.zeros(shape=(len(parents,))),
                    'interactions': np.zeros(shape=get_interactions_length(len(parents)),)
                }
        else:
            # term is given
            # dict of interactions
            interactions_dict = get_interactions_dict(parents)
            # coefs placeholders
            bias_coef = 0
            linear_coef = np.zeros(shape=(len(parents),))
            interactions_coef = np.zeros(shape=get_interactions_length(len(parents)),)
            for term in terms:
                pa = []
                coef = 1
                for comp in term.split('*'):
                    if comp in parents:
                        pa.append(comp)
                    else:
                        coef = float(comp)

                if len(pa) == 0:
                    # bias term
                    bias_coef = coef
                elif len(pa) == 1:
                    # linear term
                    ind = parents.index(pa[0])
                    linear_coef[ind] = coef
                else:
                    # interaction term
                    ind = interactions_dict.index(set(pa))
                    interactions_coef[ind] = coef
            params[p] = {
                'bias': bias_coef,
                'linear': linear_coef,
                'interactions': interactions_coef
            }
    # do correction
    pattern = re.compile('correct\[(.*)]')
    res = pattern.search(line)
    try:
        corrs = res.group(1)
        correction_config = {
            conf.split('=')[0]: float(conf.split('=')[1])
            for conf in corrs.split(',')
        }
        return {
            'output_distribution': dist,
            'dist_params_coefs': params,
            'do_correction': True,
            'correction_config': correction_config
        }
    except AttributeError:
        return {
            'output_distribution': dist,
            'dist_params_coefs': params,
            'do_correction': False
        }

def edge_parser(line):
    line = line.replace(' ', '')
    # First check: if dist = ?
    if line == 'free':
        return {
            'function_name': '?',
            'do_correction': True
        }
    # find the func(p1=v1, ...) pattern
    output_params_pattern = re.compile(
        '({})\((.*)\)'.format('|'.join(EDGE_FUNCTIONS.keys()))
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
    correct = re.compile('correct\[]')
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
    obj = node_parser('?', ['A', 'B', 'C'])
    # obj = edge_parser('sigmoid(alpha=2.0, beta=1.8), correct[]')
    # graph_file_parser('../../graph_templates/causal_triangle.yml')