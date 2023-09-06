from pyparcs import Description, Guideline
import numpy
numpy.random.seed(42)

guideline_free_nodes = Guideline(
    {'nodes': {'normal': {'mu_': [['f-range', -2, 2], 0, 0],
                          'sigma_': [['f-range', 1, 4], 0, 0]},
               'exponential': {'lambda_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]},
               'poisson': {'lambda_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]}}
     }
)
guideline_bernoulli = Guideline(
    {'nodes': {'bernoulli': {'p_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]}}}
)
description = Description({'A_1': 'random, tags[P2]',
                           'A_2': 'random, tags[P2]',
                           'Y': 'bernoulli(p_=?), correction[target_mean=0.5], tags[P1]',
                           'A_1->Y': 'identity()', 'A_2->Y': 'identity()'})

description.randomize_parameters(guideline_bernoulli, 'P1')
print(description.is_partial)  # gives True: the outline is still partially specified
description.randomize_parameters(guideline_free_nodes, 'P2')
print(description.is_partial)  # gives False: All the undefined parameters are specified
