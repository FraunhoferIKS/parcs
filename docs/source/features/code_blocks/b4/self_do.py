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

samples, errors = g.sample(
    size=3,
    return_errors=True
)
print(samples)
#           C         A         Y
# 0  0.496772  1.332864  0.409261
# 1  0.222669 -0.029024  0.884398
# 2 -0.205022 -2.051459 -2.569398

intrv_samples = g.do_self(
    func=lambda a: a+1, intervene_on='A',
    use_sampled_errors=True, sampled_errors=errors
)
print(intrv_samples)
#           C         A         Y
# 0  0.496772  0.332864 -0.190739
# 1  0.222669 -1.029024  0.284398
# 2 -0.205022 -3.051459 -3.169398