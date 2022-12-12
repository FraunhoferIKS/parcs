import yaml
import numpy as np
from parcs.cdag.output_distributions import DISTRIBUTION_PARAMS


def config_parser(dir_):
    with open(dir_, 'r') as stream:
        conf = yaml.safe_load(stream)
    return conf


def config_dumper(dict_, dir_):
    with open(dir_, 'w') as outfile:
        yaml.dump(dict_, outfile, default_flow_style=False)


def dist_param_coefs_reader(obj, dist):
    given_params = obj.keys()
    assert set(given_params).issubset(set(DISTRIBUTION_PARAMS[dist]))
    return {
        param: {
            'bias': obj[param]['bias'],
            'linear': np.array(obj[param]['linear']),
            'interactions': np.array(obj[param]['interactions']),
        } for param in given_params
    }


def empty_dist_param_coefs(dist):
    return {
        param: {'bias': None, 'linear': None, 'interactions': None}
        for param in DISTRIBUTION_PARAMS[dist]
    }


def node_guideline_reader(obj):
    nodes = []
    for node_name in obj:
        node = {'name': node_name}
        if 'output_distribution' not in obj[node_name]:
            nodes.append(node)
            continue
        else:
            dist = obj[node_name]['output_distribution']
            node['output_distribution'] = dist
            if 'dist_param_coefs' in obj[node_name]:
                node['dist_param_coefs'] = dist_param_coefs_reader(
                    obj[node_name]['dist_param_coefs'], dist
                )
            else:
                node['dist_param_coefs'] = empty_dist_param_coefs(dist)
            nodes.append(node)
    return nodes


def info_md_parser():
    raise NotImplementedError


if __name__ == '__main__':
    example = {
        'x0': {
            'output_distribution': 'bernoulli',
            'dist_param_coefs': {
                'p_': {
                    'bias': 1,
                    'linear': [2, 1],
                    'interactions': [1]
                }
            }
        },
        'x1': {
            'output_distribution': 'bernoulli'
        },
    }
    print(node_guideline_reader(example))
