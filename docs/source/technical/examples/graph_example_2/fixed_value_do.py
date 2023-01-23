from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np

# # nodes
# C: gaussian(mu_=0, sigma_=1)
# A: gaussian(mu_=2C-1, sigma_=1)
# Y: gaussian(mu_=C+0.6A, sigma_=1)
# # edges
# C->A: identity()
# C->Y: identity()
# A->Y: identity()

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)

samples = g.do(size=1000, interventions={'A': 2.5})
print(samples.head(3))
#           C    A         Y
# 0 -1.047174  2.5  0.902704
# 1  0.099876  2.5  1.282226
# 2 -1.145309  2.5  3.391779

# Average Treatment Effect:
do_1 = g.do(size=1000, interventions={'A': 1})
do_0 = g.do(size=1000, interventions={'A': 0})
ate = (do_1['Y']-do_0['Y']).mean()
print('ATE is:', np.round(ate, 2))
# ATE is: 0.59

# intervening on two variables
y_fixed = g.do(size=1000, interventions={'A': 1/0.6, 'C':-1})['Y']
print('mean: {}, variance: {}'.format(
    np.round(y_fixed.mean(), 2),
    np.round(y_fixed.var()), 2)
)
# mean: 0.0, variance: 1.0