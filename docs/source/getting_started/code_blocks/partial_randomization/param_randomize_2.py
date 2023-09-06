from pyparcs import Description, Guideline
import numpy
numpy.random.seed(42)

guideline = Guideline(
    {'nodes': {'normal': {'mu_': [['f-range', -2, 2], 0, 0],
                          'sigma_': [['f-range', 1, 4], 0, 0]},
               'bernoulli': {'p_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]},
               'exponential': {'lambda_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]},
               'poisson': {'lambda_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]}}
     }
)

for i in range(4):
    description = Description({'A_1': 'random',
                               'A_2': 'random',
                               'Y': 'bernoulli(p_=?), correction[target_mean=0.5]',
                               'A_1->Y': 'identity()', 'A_2->Y': 'identity()'})
    description.randomize_parameters(guideline)
    print(f'=== round {i} ===')
    print(description.nodes['A_1']['output_distribution'],
          description.nodes['A_2']['output_distribution'])
# === round 0 ===
# exponential exponential
# === round 1 ===
# poisson poisson
# === round 2 ===
# poisson normal
# === round 3 ===
# poisson exponential
