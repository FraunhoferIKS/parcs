from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy
numpy.random.seed(2022)

nodes, edges = graph_file_parser('graph_description_const.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=3)
print(samples)
#      C         A         Y
# 0  3.5  5.392593  8.619303
# 1  3.5  6.981875  9.811302
# 2  3.5  6.216066  7.318564