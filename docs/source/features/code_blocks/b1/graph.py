from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy
numpy.random.seed(2022)

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=5)
print(samples)
#           C         A         Y
# 0 -1.607407 -6.612316 -8.219723
# 1 -0.018125  0.398616  0.380491
# 2 -0.783934 -3.035199 -3.819132
# 3 -0.273290 -1.014330 -1.287620
# 4 -0.670573 -1.914762 -2.585335
print(samples.Y.values == (samples.C + samples.A).values)
# [ True  True  True  True  True]