from pyparcs import RandomDescription, Guideline, Graph
import numpy
numpy.random.seed(42)

guideline = Guideline(
    {'nodes': {'bernoulli': {'p_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]},
               'poisson': {'lambda_': [['f-range', 2, 4], ['f-range', 3, 5], 0]}},
     'edges': {'identity': None},
     'graph': {'density': ['f-range', 0.4, 1], 'num_nodes': ['i-range', 3, 10]}}
)

description = RandomDescription(guideline, node_prefix='X')
graph = Graph(description)

samples, _ = graph.sample(4)
print(samples)
#    X_5  X_0  X_3  X_6  X_2  X_1  X_8  X_4  X_7
# 0  0.0  1.0  3.0  2.0  0.0  1.0  1.0  0.0  2.0
# 1  0.0  1.0  1.0  2.0  1.0  0.0  0.0  0.0  0.0
# 2  1.0  0.0  1.0  0.0  1.0  0.0  0.0  1.0  0.0
# 3  1.0  0.0  2.0  1.0  3.0  4.0  0.0  0.0  3.0
