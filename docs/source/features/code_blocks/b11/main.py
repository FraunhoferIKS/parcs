from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.temporal_parsers import temporal_graph_file_parser
import numpy as np
np.random.seed(2022)

nodes, edges = temporal_graph_file_parser('gdf.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=5)
print(samples[['A', 'B_1', 'B_2', 'B_3', 'C_1', 'C_2', 'C_3']])
#           A       B_1       B_2       B_3       C_1       C_2        C_3
# 0  0.018717  0.791272 -0.853834 -0.370959  1.018717  3.037434   3.847424
# 1  1.662707  2.968408  4.685779  4.348742  2.662707  6.325414  10.956528
# 2  1.488412  1.471786  2.154080  0.071915  2.488412  5.976825   8.937023
# 3  1.129345  3.975753  2.199190  3.020516  2.129345  5.258690  10.363788
# 4  0.162772  1.466362  1.117786  1.093356  1.162772  3.325543   4.954677

print((samples['C_1']).equals(samples['A'] + samples['B_neg1'] + samples['C_0']))
print((samples['C_2']).equals(samples['A'] + samples['B_0'] + samples['C_1']))
print((samples['C_3']).equals(samples['A'] + samples['B_1'] + samples['C_2']))
# True
# True
# True
