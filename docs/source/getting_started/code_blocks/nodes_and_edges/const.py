from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'A': 'constant(2)',
                           'B': 'constant(3)'})
graph = Graph(description)
samples, _ = graph.sample(3)
print(samples)
#      A    B
# 0  2.0  3.0
# 1  2.0  3.0
# 2  2.0  3.0
