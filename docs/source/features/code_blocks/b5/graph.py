from parcs.cdag.graph_objects import Graph, m_graph_convert
from parcs.graph_builder.parsers import graph_file_parser
import numpy as np
np.random.seed(2022)

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=5)
print(samples)
#           C         A  R_A
# 0  1.500622  3.341016  1.0
# 1  0.774417  2.003075  1.0
# 2 -1.140551 -1.970714  0.0
# 3  0.590632  1.487290  0.0
# 4 -0.652315 -2.673828  0.0
print(m_graph_convert(samples, missingness_prefix='R_'))
#           A         C
# 0  3.341016  1.500622
# 1  2.003075  0.774417
# 2       NaN -1.140551
# 3       NaN  0.590632
# 4       NaN -0.652315