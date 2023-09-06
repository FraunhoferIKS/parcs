from pyparcs import Description

# or the YAML file
description = Description('./outline.yml')


from pprint import pprint

pprint(description.nodes)
# {'A': {'correction_config': {},
#        'dist_params_coefs': {'mu_': {'bias': 0.0,
#                                      'interactions': [],
#                                      'linear': []},
#                              'sigma_': {'bias': 1.0,
#                                         'interactions': [],
#                                         'linear': []}},
#        'do_correction': False,
#        'output_distribution': 'normal'},
#  'B': {'correction_config': {},
#        'dist_params_coefs': {'p_': {'bias': 0.4,
#                                     'interactions': [],
#                                     'linear': []}},
#        'do_correction': False,
#        'output_distribution': 'bernoulli'},
#  'C': {'correction_config': {},
#        'dist_params_coefs': {'mu_': {'bias': -1.0,
#                                      'interactions': [0, 0, 1],
#                                      'linear': [2.0, 0]},
#                              'sigma_': {'bias': 1.0,
#                                         'interactions': [0, 0, 0],
#                                         'linear': [0, 0]}},
#        'do_correction': False,
#        'output_distribution': 'normal'}}

pprint(description.edges)
# {'A->C': {'do_correction': False,
#           'function_name': 'identity',
#           'function_params': {}},
#  'B->C': {'do_correction': False,
#           'function_name': 'identity',
#           'function_params': {}}}

print(description.parents_list)
# {'A': [], 'B': [], 'C': ['A', 'B']}
