from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np

np.random.seed(2022)

nodes, edges = graph_file_parser('g_desc.yml')
g = Graph(nodes=nodes, edges=edges)

samples_1 = g.do(size=3, interventions={'Z_2': 2.5})
print(samples_1.round(2))

samples_2 = g.do_functional(size=3, intervene_on='Z_3',
                            inputs=['Z_1', 'Z_2'], func=lambda z1, z2: (z1+z2)*10)
print(samples_2.round(2))

samples_3 = g.do_self(size=3, func=lambda z2: z2+1, intervene_on='Z_2')
print(samples_3.round(2))