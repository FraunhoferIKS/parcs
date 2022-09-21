from parcs.cdag.graph_objects import Graph
from parcs.graph_builder.parsers import graph_file_parser
import numpy
numpy.random.seed(2022)

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=5)
print(samples)