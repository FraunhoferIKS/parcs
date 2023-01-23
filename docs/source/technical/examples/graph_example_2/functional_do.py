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

samples = g.do_functional(
    size=10,
    intervene_on='Y', inputs=['A', 'C'],
    func=lambda a,c: (a+c)*10
)
print(samples.head(3))
#           C         A          Y
# 0 -0.585768 -3.240235 -38.260031
# 1 -0.713663 -1.262177 -19.758394
# 2  1.925642  0.791920  27.175618