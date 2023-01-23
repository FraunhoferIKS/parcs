from pyparcs.cdag.graph_objects import Edge
import numpy as np

edge = Edge(
    function_name='gaussian_rbf',
    function_params={
        'gamma': 0,
        'alpha': 1,
        'beta': 0,
        'tau': 2
    }
)

x = np.array([-10, -1, 0, 1, 10])
mapped_x = edge.map(x)
print(np.round(mapped_x, 2))
# [0.   0.37 1.   0.37 0.  ]