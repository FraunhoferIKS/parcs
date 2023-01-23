from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)

samples = g.sample(size=3)
# AssertionError: Bern(p) probabilities are out of [0, 1] range