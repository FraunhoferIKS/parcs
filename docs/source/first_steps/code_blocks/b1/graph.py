from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np
np.random.seed(2022)

nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=5)
print(samples)
#           C         A         Y
# 0  1.500622  3.542066  3.928658
# 1  0.774417  2.115694  3.251244
# 2 -1.140551 -2.120171 -3.445699
# 3  0.590632  1.564428  0.109688
# 4 -0.652315 -2.649744 -6.378569

samples, errors = g.sample(size=3, return_errors=True)
print(errors)
print(samples)
#           C         A         Y
# 0  0.083551  0.393740  0.074251
# 1  0.806554  0.201740  0.278956
# 2  0.488542  0.248906  0.166903
#           C         A         Y
# 0 -1.381576 -3.995493 -9.922780
# 1  0.865266 -0.177176 -0.437808
# 2 -0.028724 -1.733438 -3.710056

reproduced_samples = g.sample(
    use_sampled_errors=True,
    sampled_errors=errors
)
print(reproduced_samples)
#           C         A         Y
# 0 -1.381576 -3.995493 -9.922780
# 1  0.865266 -0.177176 -0.437808
# 2 -0.028724 -1.733438 -3.710056