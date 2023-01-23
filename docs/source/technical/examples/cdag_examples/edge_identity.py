from pyparcs.cdag.graph_objects import Edge
import numpy as np
np.random.seed(1)

# replace 'identity' with other functions
edge = Edge(function_name='identity', function_params={})

x = np.random.normal(0, 1, size=5)
print(x)
# [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763]

mapped_x = edge.map(x)
print(mapped_x)
# [ 1.62434536 -0.61175641 -0.52817175 -1.07296862  0.86540763]




# declaring the name
edge = Edge(
    parent='Z_1',
    child='Z_2',
    function_name='identity',
    function_params={}
)