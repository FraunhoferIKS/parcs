from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
from pprint import pprint
import numpy as np
np.random.seed(2022)

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)

pprint(g.get_info())
# {'edges': {'A->Y': {'correction': {'offset': -0.9682150226680958,
#                                    'scale': 2.229350605378819},
#                     'edge_function': 'identity',
#                     'function_params': {}},
#            'C->A': {'edge_function': 'identity', 'function_params': {}},
#            'C->Y': {'edge_function': 'identity', 'function_params': {}}},
#  'nodes': {'A': {'dist_params_coefs': {'mu_': {'bias': -1.0,
#                                                'interactions': array([], dtype=float64),
#                                                'linear': array([2.])},
#                                        'sigma_': {'bias': 1.0,
#                                                   'interactions': array([], dtype=float64),
#                                                   'linear': array([0.1])}},
#                  'node_type': 'stochastic',
#                  'output_distribution': 'gaussian'},
#            'C': {'dist_params_coefs': {'mu_': {'bias': 0.0,
#                                                'interactions': array([], dtype=float64),
#                                                'linear': array([], dtype=float64)},
#                                        'sigma_': {'bias': 1.0,
#                                                   'interactions': array([], dtype=float64),
#                                                   'linear': array([], dtype=float64)}},
#                  'node_type': 'stochastic',
#                  'output_distribution': 'gaussian'},
#            'Y': {'dist_params_coefs': {'mu_': {'bias': 0,
#                                                'interactions': array([-0.3]),
#                                                'linear': array([1., 1.])},
#                                        'sigma_': {'bias': 2.0,
#                                                   'interactions': array([0.]),
#                                                   'linear': array([0., 0.])}},
#                  'node_type': 'stochastic',
#                  'output_distribution': 'gaussian'}}}

