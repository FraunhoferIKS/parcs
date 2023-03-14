from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np
np.random.seed(2022)

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)

g.visualize(shape_coded=False)
