import re
import numpy as np
from pprint import pprint
from parcs.cdag.output_distributions import DISTRIBUTION_PARAMS
from parcs.cdag.mapping_functions import EDGE_FUNCTIONS
from parcs.cdag.utils import get_interactions_length, get_interactions_dict


def node_parser(line, parents):
    # preliminary: get order of interactions
    # remove spaces
    line = line.replace(' ', '')
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
    keys_ = [p.split('=')[0] for p in params.split(',')]
    values_ = [p.split('=')[1] for p in params.split(',')]
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
        # add coef 1 to -A, -B, ...
        for par in parents:
            eq = eq.replace('-{}'.format(par), '-1*{}'.format(par))

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
    func_params = {
        p.split('=')[0]: float(p.split('=')[1])
        for p in params.split(',')
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


if __name__ == '__main__':
    # obj = node_parser('gaussian(mu_=-B+1-A, sigma_=1), correct[hi=1]', ['A', 'B', 'C'])
    obj = edge_parser('sigmoid(alpha=2.0, beta=1.8), correct[]')
    pprint(obj)