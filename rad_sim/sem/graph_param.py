import numpy as np
import pandas as pd
from itertools import product
from scipy.special import comb
from rad_sim.utils import read_yaml_config
from rad_sim.sem.utils import is_acyclic, mask_matrix
from rad_sim.sem.mapping_functions import get_output_function_options
from rad_sim.sem.graph_objects import (
    NODE_OUTPUT_TYPES, ALLOWED_EDGE_FUNCTIONS, ALLOWED_STATE_FUNCTIONS,
    ALLOWED_OUTPUT_FUNCTIONS
)
from pprint import pprint


def get_full_random_adj_matrix(num_nodes=None, connectivity_ratio=1):
    # create full UT matrix
    adj = np.triu(np.ones(shape=(num_nodes, num_nodes)), k=1)
    # create connectivity mask
    mask = np.random.choice([0, 1], p=[1-connectivity_ratio, connectivity_ratio], size=adj.shape)
    # mask
    adj = mask_matrix(matrix=adj, mask=mask)
    adj = adj.astype(int)
    # shuffle
    idx = np.random.permutation(num_nodes)
    adj = adj[:, idx]
    adj = adj[idx, :]
    # create names
    names = ['X{}'.format(i) for i in range(1, num_nodes+1)]
    return pd.DataFrame(adj, index=names, columns=names)


def pick_param(range_val=None):
    if isinstance(range_val, list):
        return np.random.uniform(range_val[0], range_val[1])
    elif isinstance(range_val, str):
        vals = [val.strip() for val in range_val.split(',')]
        try:
            vals = [int(val) for val in vals]
        except TypeError:
            pass
        return np.random.choice(vals)
    else:
        raise ValueError("invalid parameter range {}".format(range_val))


def get_random_state_coefs(state_function=None, num_parents=None, range_val=None):
    assert num_parents > 0
    if state_function == 'linear':
        num_coefs = num_parents
    elif state_function == 'poly1_interactions':
        num_coefs = 1 + num_parents + comb(num_parents, 2, exact=True)
    else:
        raise ValueError("invalid state function {}".format(state_function))
    return np.array([
        pick_param(range_val=range_val) for _ in range(num_coefs)
    ])


class GraphParam:
    def __init__(self):
        self.is_defined = {
            'adj_matrix': False,
            'node': {
                'output_types': False,
                'state_functions': False,
                'output_functions': False,
                'state_params': False,
                'output_params': False
            },
            'edge': {
                'functions': False,
                'params': False
            }
        }

        self.node_list = []
        self.adj_matrix = pd.DataFrame()
        self.edge_function_specs = {}

        # helper attributes
        self._node_names = []

    def _update_node_list(self, node_name=None, key=None, value=None):
        for node_id in range(len(self.node_list)):
            if self.node_list[node_id]['name'] == node_name:
                self.node_list[node_id][key] = value
                return self
        raise (NameError, "node name not found")

    def _read_from_node_list(self, node_name=None, key=None):
        for node_id in range(len(self.node_list)):
            if self.node_list[node_id]['name'] == node_name:
                return self.node_list[node_id][key]
        raise NameError("node name {} not found".format(node_name))

    def set_adj_matrix(self, set_type=None, **kwargs):
        if set_type == 'custom':
            assert 'adj_matrix' in kwargs, "custom set type requires adj_matrix arg"
            adj_matrix = kwargs['adj_matrix']
            assert adj_matrix.columns.to_list() == adj_matrix.index.to_list(), \
                "index and columns of adj_matrix must be identical"
            assert is_acyclic(adj_matrix=adj_matrix), "graph is not acyclic"
        elif set_type == 'full_random':
            assert 'num_nodes' in kwargs, "full random set type requires num_nodes arg"
            adj_matrix = get_full_random_adj_matrix(num_nodes=kwargs['num_nodes'])
        else:
            raise (ValueError, "set_type undefined")
        self._node_names = adj_matrix.columns.to_list()
        self.node_list = [
            {
                'name': node_name,
                'parents': [i for i in adj_matrix if adj_matrix.loc[i, node_name] == 1],
                'output_type': '',
                'state_function': '',
                'output_function': '',
                'state_params': {},
                'output_params': {}
            } for node_name in self._node_names
        ]
        self.edge_function_specs = {
            '{} -> {}'.format(i, j): {'function_name': '', 'params': {}}
            for i, j in product(self._node_names, self._node_names)
            if adj_matrix.loc[i, j] == 1
        }
        self.adj_matrix = adj_matrix
        self.is_defined['adj_matrix'] = True
        return self

    def set_node_output_types(self, set_type=None, **kwargs):
        if set_type == 'custom':
            assert 'node_types_list' in kwargs, "custom set type requires node_types_list arg"
            types_list = kwargs['node_types_list']
        elif set_type == 'full_random':
            node_types = np.random.choice(NODE_OUTPUT_TYPES, size=len(self._node_names))
            types_list = {name: type_ for name, type_ in zip(self._node_names, node_types)}
        else:
            raise (ValueError, "set_type undefined")
        for name in types_list:
            self._update_node_list(
                node_name=name, key='output_type',
                value=types_list[name]
            )
        self.is_defined['node']['output_types'] = True
        return self

    def set_edge_functions(self, set_type=None, **kwargs):
        assert self.is_defined['node']['output_types'], "node_types is not defined yet"
        if set_type == 'custom':
            assert 'functions' in kwargs, "custom set type requires edge_functions arg"
            funcs = kwargs['functions']
        elif set_type == 'full_random':
            funcs = {
                '{} -> {}'.format(i, j): np.random.choice(
                    ALLOWED_EDGE_FUNCTIONS[
                        self._read_from_node_list(node_name=i, key='output_type')
                    ]
                ) for i, j in product(self._node_names, self._node_names) if self.adj_matrix.loc[i, j] == 1
            }
        else:
            raise (ValueError, "set_type undefined")
        for edge in funcs:
            self.edge_function_specs[edge]['function_name'] = funcs[edge]
        self.is_defined['edge']['functions'] = True
        return self

    def set_edge_function_params(self, set_type=None, **kwargs):
        assert self.is_defined['edge']['functions'], "edge functions are not defined yet"
        if set_type == 'custom':
            assert 'params' in kwargs, "custom set type requires params arg"
            params = kwargs['params']
        elif set_type == 'full_random':
            assert 'config_dir' in kwargs, "specify the YAML config file for param ranges"
            ranges = read_yaml_config(config_dir=kwargs['config_dir'])['edge']
            params = {}
            for edge in self.edge_function_specs:
                func = self.edge_function_specs[edge]['function_name']
                try:
                    params[edge] = {
                        item: pick_param(range_val=ranges[func][item])
                        for item in ranges[func]
                    }
                except KeyError:
                    continue
        else:
            raise (ValueError, "set_type undefined")
        for edge in params:
            self.edge_function_specs[edge]['params'] = params[edge]
        self.is_defined['edge']['params'] = True
        return self

    def set_state_functions(self, set_type=None, **kwargs):
        assert self.is_defined['adj_matrix'], "adj_matrix is not defined yet"
        if set_type == 'custom':
            assert 'functions' in kwargs, "custom set type requires functions arg"
            funcs = kwargs['functions']
        elif set_type == 'full_random':
            funcs = {
                i: np.random.choice(ALLOWED_STATE_FUNCTIONS)
                for i in self._node_names
            }
        else:
            raise (ValueError, "set_type undefined")
        for name in funcs:
            self._update_node_list(
                node_name=name, key='state_function',
                value=funcs[name]
            )
        self.is_defined['node']['state_functions'] = True
        return self

    def set_output_functions(self, set_type=None, **kwargs):
        assert self.is_defined['adj_matrix'], "adj_matrix is not defined yet"
        if set_type == 'custom':
            assert 'functions' in kwargs, "custom set type requires functions arg"
            funcs = kwargs['functions']
        elif set_type == 'full_random':
            funcs = {
                i: np.random.choice(
                    ALLOWED_OUTPUT_FUNCTIONS[
                        self._read_from_node_list(node_name=i, key='output_type')
                    ]
                ) for i in self._node_names
            }
        else:
            raise (ValueError, "set_type undefined")
        for name in funcs:
            func = funcs[name]
            o_type = self._read_from_node_list(node_name=name, key='output_type')
            assert func in get_output_function_options(
                output_type=o_type
            ), "node {}: function {} doesn't match the output type {}".format(name, func, o_type)
            self._update_node_list(
                node_name=name, key='output_function',
                value=func
            )
        self.is_defined['node']['output_functions'] = True
        return self

    def set_state_params(self, set_type=None, **kwargs):
        assert self.is_defined['node']['state_functions'], "state functions are not defined yet"
        if set_type == 'custom':
            assert 'params' in kwargs, "custom set type requires params arg"
            params = kwargs['params']
        elif set_type == 'full_random':
            assert 'config_dir' in kwargs, "specify the YAML config file for param ranges"
            params = {}
            for node in self._node_names:
                state_func = self._read_from_node_list(node_name=node, key='state_function')
                num_parents = len(self._read_from_node_list(node_name=node, key='parents'))
                range_val = read_yaml_config(config_dir=kwargs['config_dir'])['node']['state'][state_func]['coefs']
                try:
                    coefs = get_random_state_coefs(
                        state_function=state_func,
                        num_parents=num_parents,
                        range_val=range_val
                    )
                except AssertionError:
                    continue
                params[node] = {'coefs': coefs}
        else:
            raise (ValueError, "set_type undefined")
        for name in params:
            self._update_node_list(
                node_name=name, key='state_params',
                value=params[name]
            )
        self.is_defined['node']['state_params'] = True
        return self

    def set_output_params(self, set_type=None, **kwargs):
        assert self.is_defined['node']['output_functions'], "output functions are not defined yet"
        if set_type == 'custom':
            assert 'params' in kwargs, "custom set type requires params arg"
            params = kwargs['params']
        elif set_type == 'full_random':
            assert 'config_dir' in kwargs, "specify the YAML config file for param ranges"
            ranges = read_yaml_config(config_dir=kwargs['config_dir'])['node']['output']
            params = {}
            for node in self.node_list:
                func = node['output_function']
                try:
                    params[node['name']] = {
                        item: pick_param(range_val=ranges[func][item])
                        for item in ranges[func]
                    }
                except KeyError:
                    continue
        else:
            raise (ValueError, "set_type undefined")
        for name in params:
            self._update_node_list(
                node_name=name, key='output_params',
                value=params[name]
            )
        self.is_defined['node']['output_params'] = True
        return self

    def get_node_list(self):
        return self.node_list

    def get_adj_matrix(self):
        return self.adj_matrix

    def get_edge_function_specs(self):
        return self.edge_function_specs


if __name__ == '__main__':
    param = GraphParam()
    param.set_adj_matrix(
        set_type='full_random', num_nodes=4
    ).set_node_output_types(
        set_type='full_random'
    ).set_edge_functions(
        set_type='full_random'
    ).set_edge_function_params(
        set_type='full_random', config_dir='../../configs/params/default.yml'
    ).set_state_functions(
        set_type='full_random'
    ).set_state_params(
        set_type='full_random', config_dir='../../configs/params/default.yml'
    ).set_output_functions(
        set_type='full_random'
    ).set_output_params(
        set_type='full_random', config_dir='../../configs/params/default.yml'
    )

    # adj = pd.DataFrame(
    #     [
    #         [0, 0, 1],
    #         [0, 0, 1],
    #         [0, 0, 0]
    #     ],
    #     columns=['A1', 'A2', 'B1'],
    #     index=['A1', 'A2', 'B1']
    # )
    # param.set_adj_matrix(
    #     set_type='custom', adj_matrix=adj
    # ).set_node_output_types(
    #     set_type='custom', node_types_list={'A1': 'binary', 'A2': 'binary', 'B1': 'binary'}
    # ).set_edge_functions(
    #     set_type='custom', functions={'A1 -> B1': 'sigmoid', 'A2 -> B1': 'sigmoid'}
    # ).set_edge_function_params(
    #     set_type='custom', params={'A2 -> B1': {'a': 2}}
    # ).set_state_functions(
    #     set_type='custom', functions={'A1': 'linear', 'A2': 'linear', 'B1': 'linear'}
    # ).set_output_functions(
    #     set_type='custom', functions={'A1': 'bernoulli', 'A2': 'bernoulli', 'B1': 'bernoulli'}
    # ).set_state_params(
    #     set_type='custom', params={'A1': {'a': 2}, 'A2': {}, 'B1': [1, 2]}
    # ).set_output_params(
    #     set_type='custom', params={'A1': {'a': 6}, 'A2': {}, 'B1': [1, 2]}
    # )
    from pprint import pprint
    # pprint(param.is_defined)
    pprint(param.node_list)
    # pprint(param.edge_function_specs)
