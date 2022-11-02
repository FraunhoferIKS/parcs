from parcs.cdag.graph_objects import Graph
from parcs.graph_builder.parsers import graph_file_parser

# det
# nodes, edges = graph_file_parser('det.yml')
# g = Graph(nodes, edges)
# print(g.sample(3))

# data
nodes, edges = graph_file_parser('data.yml')
g = Graph(nodes, edges)
print(g.sample(4))