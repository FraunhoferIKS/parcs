from parcs.cdag.graph_objects import Graph
from parcs.graph_builder.parsers import graph_file_parser
import numpy
numpy.random.seed(2022)

nodes, edges = graph_file_parser('graph_description_const.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=3)
print(samples)
#           C         A         Y
# 0 -1.607407 -6.612316 -8.219723
# 1 -0.018125  0.398616  0.380491
# 2 -0.783934 -3.035199 -3.819132