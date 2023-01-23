from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np

np.random.seed(2022)

nodes, edges = graph_file_parser('g_desc.yml')
g = Graph(nodes=nodes, edges=edges)

samples, errors = g.sample(size=2, return_errors=True)
print(samples.round(2))
print(errors.round(2))

reproduced_samples = g.sample(use_sampled_errors=True, sampled_errors=errors)
print(samples.round(2))