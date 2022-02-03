import pandas as pd
from modules.sem.utils import is_acyclic


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
                'function_params': False
            }
        }

        self.node_list = []
        self.info_matrix = pd.DataFrame()

        # helper attributes
        self._node_names = []

    def _update_node_list(self, node_name=None, key=None, value=None):
        for node_id in range(len(self.node_list)):
            if self.node_list[node_id]['name'] == node_name:
                self.node_list[node_id][key] = value
                return self
        raise (NameError, "node name not found")

    def set_adj_matrix(self, set_type=None, **kwargs):
        if set_type == 'custom':
            assert 'adj_matrix' in kwargs, "custom set type requires adj_matrix arg"
            adj_matrix = kwargs['adj_matrix']
            assert adj_matrix.columns.to_list() == adj_matrix.index.to_list(), \
                "index and columns of adj_matrix must be equal"
            assert is_acyclic(adj_matrix=adj_matrix), "graph is not acyclic"
            self._node_names = adj_matrix.columns.to_list()
            self.info_matrix = adj_matrix
            self.node_list = [
                {
                    'name': node_name,
                    'parents': [],
                    'output_type': '',
                    'state_function_name': '',
                    'output_function_name': '',
                    'state_params': {},
                    'output_params': {}
                } for node_name in self._node_names
            ]
        else:
            raise (ValueError, "set_type undefined")
        self.is_defined['adj_matrix'] = True
        return self

    def set_node_output_types(self, set_type=None, **kwargs):
        if set_type == 'custom':
            assert 'node_types_list' in kwargs, "custom set type requires node_types_list arg"
            for name in kwargs['node_types_list']:
                self._update_node_list(
                    node_name=name, key='output_type',
                    value=kwargs['node_types_list'][name]
                )
        else:
            raise (ValueError, "set_type undefined")
        self.is_defined['node']['output_types'] = True
        return self

    def set_edge_functions(self, set_type=None, **kwargs):
        # assert self.is_defined['node']['output_types'], "node_types is not defined yet"
        # if set_type == 'custom':
        #     assert 'edge_functions' in kwargs, "custom set type requires edge_functions arg"
        #     for edge in kwargs['edge_functions']:
        #         parent, child = (n.strip() for n in edge.split('->'))
        #         print(self.info_matrix.loc[parent, child])
        #         self.info_matrix.loc[parent, child] = {
        #             'function_name': 2#kwargs['edge_functions'][edge]
        #         }
        # else:
        #     raise (ValueError, "set_type undefined")
        # self.is_defined['edge']['functions'] = True
        return self


if __name__ == '__main__':
    import pandas as pd
    param = GraphParam()
    adj = pd.DataFrame(
        [
            [0, 0, 1],
            [0, 0, 1],
            [0, 0, 0]
        ],
        columns=['A1', 'A2', 'B1'],
        index=['A1', 'A2', 'B1']
    )
    param.set_adj_matrix(
        set_type='custom', adj_matrix=adj
    ).set_node_output_types(
        set_type='custom', node_types_list={'A1': 'binary', 'A2': 'binary', 'B1': 'binary'}
    ).set_edge_functions(
        set_type='custom', edge_functions={'A1 -> B1': 'sigmoid', 'A2 -> B1': 'sigmoid'}
    )
