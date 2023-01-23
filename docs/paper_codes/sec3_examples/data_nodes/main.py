from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np
np.random.seed(2022)


nodes, edges = graph_file_parser('data.yml')
g = Graph(nodes, edges)
samples = g.sample(6)
print(samples.round(2))