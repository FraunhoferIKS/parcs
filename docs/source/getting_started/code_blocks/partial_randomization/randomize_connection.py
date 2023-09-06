from pyparcs import Description, Guideline
import numpy
numpy.random.seed(42)

guideline = Guideline(
    {'nodes': {'bernoulli': {'p_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]}},
     'edges': {'identity': None},
     'graph': {'density': 1}}
)

outline_L = {'L_1': 'normal(mu_=0, sigma_=1)', 'L_2': 'normal(mu_=L_1^2+L_1, sigma_=1)'}
outline_Z = {'Z_1': 'bernoulli(p_=0.3)', 'Z_2': 'bernoulli(p_=0.5)'}

description = Description(outline_L, infer_edges=True)
description.randomize_connection_to(outline_Z, guideline, infer_edges=True)

print(description.nodes.keys())
# dict_keys(['L_1', 'L_2', 'Z_1', 'Z_2'])

print(description.edges.keys())
# dict_keys(['L_1->L_2', 'L_1->Z_1', 'L_2->Z_1', 'L_1->Z_2', 'L_2->Z_2'])

print(description.nodes['Z_1']['dist_params_coefs']['p_'])
# {'bias': 0.3, 'linear': [-4.11, -4.88], 'interactions': [0, 0, 0]}
