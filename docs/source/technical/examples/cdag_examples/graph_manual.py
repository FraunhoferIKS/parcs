from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import numpy as np


nodes, edges = graph_file_parser('graph_description.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=5)
print(samples)
#           C         A         Y
# 0 -0.063525  1.748499  2.231440
# 1  0.724484 -0.499045 -4.305049
# 2  1.740207  2.135624  1.453687
# 3  0.071935 -1.684686  0.533123
# 4  0.398912 -0.249330  0.645679

samples, errors = g.sample(size=3, return_errors=True)
print(errors)
print(samples)
#           C         A         Y
# 0  0.083193  0.680838  0.853406
# 1  0.306862  0.170525  0.273802
# 2  0.506727  0.865491  0.490109
#           C         A         Y
# 0 -1.383912 -3.362831 -4.040595
# 1 -0.504763 -2.913560 -5.062230
# 2  0.016864  0.140918  0.107478

reproduced_samples = g.sample(
    use_sampled_errors=True,
    sampled_errors=errors
)
print(reproduced_samples)
#           C         A         Y
# 0 -1.383912 -3.362831 -4.040595
# 1 -0.504763 -2.913560 -5.062230
# 2  0.016864  0.140918  0.107478