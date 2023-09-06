from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'C': 'normal(mu_=0, sigma_=1)',
                           'A': 'normal(mu_=2C-1, sigma_=1)',
                           'Y': 'normal(mu_=C+0.6A, sigma_=1)'},
                          infer_edges=True)
graph = Graph(description)
samples, _ = graph.do(size=3, interventions={'A': 2.5})
print(samples)
#           C    A         Y
# 0  1.651819  2.5  3.770674
# 1 -1.010956  2.5 -0.522014
# 2  1.108496  2.5  2.864730

# Average Treatment Effect:
do_1, _ = graph.do(size=1000, interventions={'A': 1})
do_0, _ = graph.do(size=1000, interventions={'A': 0})
ate = (do_1['Y']-do_0['Y']).mean()
print('ATE is:', np.round(ate, 2))
# ATE is: 0.61

# intervening on two variables
samples, _ = graph.do(size=1000, interventions={'A': 1/0.6, 'C': -1})
y_fixed = samples['Y']
print(f'mean: {np.round(y_fixed.mean(), 2)}, variance: {np.round(y_fixed.var(), 2)}')
# mean: -0.08, variance: 1.03
