from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np

np.random.seed(2022)

nodes, edges = graph_file_parser('g_desc.yml')
g = Graph(nodes=nodes, edges=edges)

samples = g.sample(size=2)
print(samples.round(2))