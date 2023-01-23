from pyparcs.cdag.graph_objects import Edge
import numpy as np

params = {'gamma': 0, 'alpha': 1, 'beta': 0, 'tau': 1}
edge_no_correction = Edge(
    function_name='sigmoid',
    function_params=params
)
edge_correction = Edge(
    function_name='sigmoid',
    function_params=params,
    do_correction=True
)

x = np.array([0, 5, 10, 20])
mapped_x_no_correction = edge_no_correction.map(x)
mapped_x_correction = edge_correction.map(x)

print(np.round(mapped_x_no_correction, 2))
# [0.5 1.  1.  1. ]
print(np.round(mapped_x_correction, 2))
# [0.08 0.5  0.92 1.  ]
