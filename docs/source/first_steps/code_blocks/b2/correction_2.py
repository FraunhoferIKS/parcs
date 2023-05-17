from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser

nodes, edges = graph_file_parser('graph_description_correction.yml')
g = Graph(nodes=nodes, edges=edges)

samples = g.sample(size=3)

print(samples)
#           C         A    Y
# 0  0.446559 -1.114598  1.0
# 1 -1.871864 -4.844675  0.0
# 2  1.095366  0.805899  0.0