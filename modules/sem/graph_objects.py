from modules.sem.utils import exp_prob, mask_matrix, is_acyclic, topological_sort
from modules.sem import mapping_functions
import numpy as np
import pandas as pd
from scipy.special import comb
from itertools import product


class Edge:
    def __init__(self,
                 parent=None,
                 child=None,
                 complexity=None):
        self.parent = parent
        self.child = child
        if complexity is None:
            self.complexity = 0
        else:
            self.complexity = complexity

        self.edge_function = {
            'name': None,
            'function': None,
            'params': {}
        }
        self.is_random = {
            'function': False,
            'params': False
        }

        # options list attributes
        self.function_list = {}
        self.make_function_list()

        # output values
        self.value = np.array([])

    def get_configs(self):
        return {
            'parent': self.parent,
            'child': self.child,
            'complexity': self.complexity,
            'edge_function': self.edge_function,
            'is_random': self.is_random
        }

    def get_function_options(self):
        return []

    def get_param_options(self, function=None):
        return {}

    def make_function_list(self):
        pass

    def get_function_probs(self):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_function_options())
        )

    def set_random_function(self):
        function_name = np.random.choice(
            self.get_function_options(),
            p=self.get_function_probs()
        )
        self.is_random['function'] = True
        self.edge_function['name'] = function_name
        self.edge_function['function'] = self.function_list[function_name]
        return self

    def set_random_function_params(self):
        param_options = self.get_param_options(
            function=self.edge_function['name']
        )
        for param in param_options:
            options = param_options[param]
            if isinstance(options, list):
                param_value = np.random.uniform(low=options[0], high=options[1])
            elif isinstance(options, set):
                param_value = np.random.choice(list(options))
            else:
                raise TypeError
            self.edge_function['params'][param] = param_value

        self.is_random['params'] = True
        return self

    def set_function(self, function_name=None):
        self.edge_function = {
            'name': function_name,
            'function': self.function_list[function_name]
        }
        self.is_random['function'] = False
        return self

    def set_function_params(self, params=None):
        self.edge_function['params'] = params
        self.is_random['params'] = False
        return self

    def random_initiate(self):
        self.set_random_function()
        self.set_random_function_params()
        return self

    def map(self, array=None):
        self.value = self.edge_function['function'](
            array=array,
            **self.edge_function['params']
        )
        return self.value


class BinaryInputEdge(Edge):
    def __init__(self,
                 parent=None,
                 child=None,
                 complexity=None):
        super().__init__(
            parent=parent,
            child=child,
            complexity=complexity
        )

    def get_function_options(self):
        return (
            'identity',
            'beta_noise'
        )

    def get_param_options(self, function=None):
        options = {
            'identity': {},
            'beta_noise': {
                'rho': [0.05, 0.3]
            }
        }
        return options[function]

    def make_function_list(self):
        self.function_list = {
            'identity': mapping_functions.edge_binary_identity,
            'beta_noise': mapping_functions.edge_binary_beta
        }


class ContinuousInputEdge(Edge):
    def __init__(self,
                 parent=None,
                 child=None,
                 complexity=None):
        super().__init__(
            parent=parent,
            child=child,
            complexity=complexity
        )

    def get_function_options(self):
        return (
            'sigmoid',
            'gaussian_rbf'
        )

    def get_param_options(self, function=None):
        options = {
            'sigmoid': {
                'alpha': [1, 6],
                'beta': [-0.8, 0.8],
                'gamma': {0, 1},
                'tau': {1, 3, 5},
                'rho': [0.05, 0.2]
            },
            'gaussian_rbf': {
                'alpha': [1, 6],
                'beta': [-0.8, 0.8],
                'gamma': {0, 1},
                'tau': {2, 4, 6},
                'rho': [0.05, 0.2]
            }
        }
        return options[function]

    def make_function_list(self):
        self.function_list = {
            'sigmoid': mapping_functions.edge_sigmoid,
            'gaussian_rbf': mapping_functions.edge_gaussian_rbf
        }


class Node:
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None):
        # basic attributes
        self.name = name
        self.parents = parents
        self.complexity = complexity

        # selected activation functions attributes
        self.state_function = {
            'name': None,
            'function': mapping_functions.state_empty,
            'params': {}
        }
        self.output_function = {
            'name': None,
            'function': mapping_functions.output_empty,
            'params': {}
        }
        # helper
        self.functions = {'state': self.state_function, 'output': self.output_function}
        self.is_random = {
            'state_function': False,
            'state_params': False,
            'output_function': False,
            'output_params': False
        }

        # value attributes
        self.value = {
            'state': np.array([]),
            'output': np.array([])
        }

        # options lists
        self.function_list = {
            'state': {},
            'output': {}
        }
        self.make_function_list()

    def get_configs(self):
        return {
            'name': self.name,
            'complexity': self.complexity,
            'state_function': self.state_function,
            'output_function': self.output_function,
            'is_random': self.is_random
        }

    def get_function_options(self, function_type=None):
        return []

    def get_param_options(self, function_type=None, function_name=None):
        return {}

    def make_function_list(self):
        pass

    def get_function_probs(self, function_type=None):
        return exp_prob(
            complexity=self.complexity,
            num_categories=len(self.get_function_options(function_type=function_type))
        )

    def _set_random_function(self, function_type=None):
        name = np.random.choice(
         self.get_function_options(function_type=function_type),
         p=self.get_function_probs(function_type=function_type)
        )
        self.functions[function_type]['name'] = name
        self.functions[function_type]['function'] = self.function_list[function_type][name]

        self.is_random['{}_function'.format(function_type)] = True
        return self

    def set_random_state_function(self):
        return self._set_random_function(function_type='state')

    def set_random_output_function(self):
        return self._set_random_function(function_type='output')

    @staticmethod
    def _get_param_size(num_parents=None, function_name=None, function_type=None):
        if function_type == 'output':
            return None
        else:
            if function_name == 'linear':
                return num_parents
            elif function_name == 'poly1_interactions':
                # interactions: True, Bias: True
                return 1 + num_parents + comb(num_parents, 2, exact=True)
            else:
                raise ValueError('Function {} not implemented'.format(function_name))

    def _set_random_params(self, function_type=None):
        param_options = self.get_param_options(
            function_type=function_type,
            function_name=self.functions[function_type]['name']
        )
        # for state functions, we need an array of random values
        size_ = self._get_param_size(
            num_parents=len(self.parents),
            function_name=self.functions[function_type]['name'],
            function_type=function_type
        )
        for param in param_options:
            options = param_options[param]
            if isinstance(options, list):
                param_value = np.random.uniform(
                    low=options[0],
                    high=options[1],
                    size=size_
                )
            elif isinstance(options, set):
                param_value = np.random.choice(
                    list(options),
                    size=size_
                )
            else:
                raise TypeError
            self.functions[function_type]['params'][param] = param_value

        self.is_random['{}_params'.format(function_type)] = True
        return self

    def set_random_state_params(self):
        return self._set_random_params(function_type='state')

    def set_random_output_params(self):
        return self._set_random_params(function_type='output')

    def _set_function(self, function_type=None, function_name=None):
        self.functions[function_type]['name'] = function_name
        self.functions[function_type]['function'] = \
            self.function_list[function_type][function_name]
        self.is_random['{}_function'] = False
        return self

    def set_state_function(self, function_name=None):
        return self._set_function(function_type='state', function_name=function_name)

    def set_output_function(self, function_name=None):
        return self._set_function(function_type='output', function_name=function_name)

    def _set_function_params(self, function_type=None, params=None):
        self.functions[function_type]['params'] = params
        self.is_random['{}_params'] = False
        return self

    def set_state_params(self, params=None):
        return self._set_function_params(function_type='state', params=params)

    def set_output_params(self, params=None):
        return self._set_function_params(function_type='output', params=params)

    def random_initiate(self):
        self.set_random_state_function()
        self.set_random_output_function()
        self.set_random_state_params()
        self.set_random_output_params()
        return self

    def calc_state(self, inputs=None, size=None):
        try:
            assert len(inputs) != 0
            result = self.state_function['function'](
                inputs=inputs,
                parents_order=self.parents,
                **self.state_function['params']
            )
            self.value['state'] = result
        except AssertionError:
            self.value['state'] = np.random.normal(size=size)
        return self.value['state']

    def calc_output(self):
        self.value['output'] = self.output_function['function'](
            array=self.value['state'],
            **self.output_function['params']
        )
        return self.value['output']


class ContinuousNode(Node):
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None,
                 **kwargs):
        super().__init__(
            name=name,
            parents=parents,
            complexity=complexity
        )

    def get_function_options(self, function_type=None):
        options = {
            'state': ('linear', 'poly1_interactions'),
            'output': ('gaussian_noise', 'gamma_noise')
        }
        return options[function_type]

    def get_param_options(self, function_type=None, function_name=None):
        options = {
            'state': {
                'linear': {
                    'coefs': [0, 3]
                },
                'poly1_interactions': {
                    'coefs': [0, 3]
                }
            },
            'output': {
                'gaussian_noise': {
                    'rho': [0.01, 0.5]
                },
                'gamma_noise': {
                    'rho': [0.01, 0.5]
                }
            }
        }
        return options[function_type][function_name]

    def make_function_list(self):
        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'gaussian_noise': mapping_functions.output_gaussian_noise,
                'gamma_noise': mapping_functions.output_gamma_noise
            }
        }
        return None


class BinaryNode(Node):
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None,
                 **kwargs):
        super().__init__(
            name=name,
            parents=parents,
            complexity=complexity
        )

    def get_function_options(self, function_type=None):
        options = {
            'state': ('linear', 'poly1_interactions'),
            'output': ('bernoulli', )
        }
        return options[function_type]

    def get_param_options(self, function_type=None, function_name=None):
        options = {
            'state': {
                'linear': {
                    'coefs': [0, 3]
                },
                'poly1_interactions': {
                    'coefs': [0, 3]
                }
            },
            'output': {
                'bernoulli': {
                    'rho': [0.01, 0.07],
                    'gamma': {0, 1},
                    'mean_': [0.1, 0.9]
                }
            }
        }
        return options[function_type][function_name]

    def make_function_list(self):
        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'bernoulli': mapping_functions.output_binary
            }
        }
        return None


class CategoricalNode(Node):
    def __init__(self,
                 name=None,
                 parents=None,
                 complexity=None,
                 **kwargs):
        super().__init__(
            name=name,
            parents=parents,
            complexity=complexity
        )

    def get_function_options(self, function_type=None):
        options = {
            'state': ('linear', 'poly1_interactions'),
            'output': ('multinomial', )
        }
        return options[function_type]

    def get_param_options(self, function_type=None, function_name=None):
        options = {
            'state': {
                'linear': {
                    'coefs': [0, 3]
                },
                'poly1_interactions': {
                    'coefs': [0, 3]
                }
            },
            'output': {
                'multinomial': {
                    'rho': [0.01, 0.07],
                    'gamma': {0, 1},
                    'centers': {tuple(np.random.uniform(size=np.random.randint(low=3, high=9))) for _ in range(10)}
                }
            }
        }
        return options[function_type][function_name]

    def make_function_list(self):
        self.function_list = {
            'state': {
                'linear': mapping_functions.state_linear,
                'poly1_interactions': mapping_functions.state_poly1_interactions
            },
            'output': {
                'multinomial': mapping_functions.output_multinomial
            }
        }
        return None


class Structure:
    def __init__(self,
                 adj_matrix: pd.DataFrame = None,
                 node_types=None,
                 complexity=None):
        if complexity is None:
            complexity = 0
        self.nodes = {}
        self.edges = {}
        assert adj_matrix.columns.tolist() == adj_matrix.index.tolist()
        self.adj_matrix = adj_matrix
        self.node_dict = {
            'continuous': ContinuousNode,
            'binary': BinaryNode,
            'categorical': CategoricalNode
        }
        self.edge_dict = {
            'continuous': ContinuousInputEdge,
            'binary': BinaryInputEdge
        }
        self.complexity = complexity

        self.data = pd.DataFrame([], columns=adj_matrix.columns)

        self._set_nodes(node_types=node_types)._set_edges(node_types=node_types)._random_initiate()

    def _set_nodes(self, node_types=None):
        adjm = self.adj_matrix

        self.nodes = {
            c: self.node_dict[node_types[c]](
                name=c,
                parents=adjm[adjm[c] == 1].index.tolist(),
                complexity=self.complexity
            ) for c in adjm.columns
        }
        return self

    def _set_edges(self, node_types=None):
        adjm = self.adj_matrix
        self.edges = {
            '{}->{}'.format(c[0], c[1]): self.edge_dict[node_types[c[0]]](
                parent=c[0],
                child=c[1],
                complexity=self.complexity
            ) for c in product(adjm.index, adjm.columns)
            if adjm.loc[c[0], c[1]] == 1
        }
        return self

    def _random_initiate(self):
        for n in self.nodes:
            self.nodes[n].random_initiate()
        for e in self.edges:
            self.edges[e].random_initiate()
        return self

    def sample(self, size=None):
        assert size is not None, 'Specify size for sample'
        for node in topological_sort(adj_matrix=self.adj_matrix):
            v = self.nodes[node]
            inputs = pd.DataFrame({
                p: self.edges['{}->{}'.format(p, v.name)].map(
                    array=self.nodes[p].value['output']
                ) for p in v.parents
            })
            v.calc_state(inputs=inputs, size=size)
            v.calc_output()
        self.data = pd.DataFrame({v: self.nodes[v].value['output'] for v in self.nodes})
        return self.data
    # TODO: what is this down here?!
    # def determinize_node(self, ):


class AutoEncoderSimulator(Structure):
    def __init__(self,
                 num_latent_vars=None,
                 latent_nodes_adjm=None,
                 num_nodes_in_hidden_layers=None,
                 output_layer_dtype_list=None,
                 complexity=0):
        latent_nodes = ['latent_{}'.format(i) for i in range(num_latent_vars)]
        hidden_layer_nodes = [
            ['hidden{}_{}'.format(num_hidden+1, i+1)
            for i in range(num_nodes_in_hidden_layers[num_hidden])]
            for num_hidden in range(len(num_nodes_in_hidden_layers))
        ]
        output_nodes = ['x{}'.format(i+1) for i in range(len(output_layer_dtype_list))]
        total_nodes = latent_nodes + [item for sublist in hidden_layer_nodes for item in sublist] + output_nodes
        num_nodes = len(total_nodes)

        adj_matrix = pd.DataFrame(np.zeros(shape=(num_nodes, num_nodes)), columns=total_nodes, index=total_nodes)

        if latent_nodes_adjm:
            assert is_acyclic(adj_matrix=pd.DataFrame(latent_nodes_adjm, columns=latent_nodes, index=latent_nodes))
            adj_matrix.loc[latent_nodes, latent_nodes] = latent_nodes_adjm

        # connect hidden to first layer
        adj_matrix.loc[latent_nodes, hidden_layer_nodes[0]] = 1
        # interconnect hidden layers
        for i in range(len(hidden_layer_nodes)-1):
            adj_matrix.loc[hidden_layer_nodes[i], hidden_layer_nodes[i+1]] = 1
        # last hidden layer to output
        adj_matrix.loc[hidden_layer_nodes[-1], output_nodes] = 1
        dtypes = ['continuous']*(num_nodes-len(output_nodes)) + output_layer_dtype_list
        node_types = {name:type_ for name, type_ in zip(total_nodes, dtypes)}

        super().__init__(
            adj_matrix=adj_matrix,
            node_types=node_types,
            complexity=complexity
        )

    def get_latent_vars(self):
        cols = [c for c in self.data.columns if c[:6]=='latent']
        return self.data[cols]

    def get_output_vars(self):
        cols = [c for c in self.data.columns if c[0]=='x']
        return self.data[cols]

    def get_full_vars(self):
        return self.data


# if __name__ == '__main__':
#     a = AutoEncoderSimulator(
#         num_latent_vars=2,
#         num_nodes_in_hidden_layers=[2, 5, 10],
#         output_layer_dtype_list=['continuous']*20,
#         complexity=0
#     )
#     a.sample(size=1000)
#     plt.scatter(a.data.x1, a.data.x2, c=a.data.x3)
#     plt.show()


if __name__ == '__main__':
    e = Edge(complexity=0.8)
    print(e.complexity)

# if __name__ == '__main__':
#     from pprint import pprint
#     node_names = ['x{}'.format(i) for i in range(6)]
#     adjm = pd.DataFrame(
#         [
#             [0, 1, 0, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1],
#             [0, 0, 0, 1, 1, 1],
#             [0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 0]
#
#         ],
#         index=node_names,
#         columns=node_names
#     )
#     node_types = {'x0': 'continuous', 'x1': 'continuous', 'x2': 'continuous', 'x3': 'continuous', 'x4': 'continuous', 'x5': 'categorical'}
#     s = Structure(adj_matrix=adjm, node_types=node_types)
#     for n in s.nodes:
#         print(n, '========')
#         pprint(s.nodes[n].get_configs())
#     for e in s.edges:
#         print(e, '========')
#         pprint(s.edges[e].get_configs())
#
#     s.sample(size=1000)
#     print(s.nodes['x5'].get_configs())
#     # plt.scatter(s.data.x3, s.data.x4, c=s.data.x5)
#     plt.hist(s.data.x5)
#     plt.show()