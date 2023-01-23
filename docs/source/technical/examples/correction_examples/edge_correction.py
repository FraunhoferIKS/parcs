from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser

nodes, edges = graph_file_parser('graph_description_edge.yml')
nodes, edges = graph_file_parser('graph_description_edge_2.yml')
g = Graph(nodes=nodes, edges=edges)

samples = g.sample(size=1000)
print(samples['Y'].mean())
# 1.0
# 0.477