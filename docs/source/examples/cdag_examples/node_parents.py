import numpy as np
import pandas as pd
from parcs.cdag.graph_objects import Node
np.random.seed(1)

data = pd.DataFrame(np.random.uniform(0, 1, size=(100, 2)), columns=('Z_1', 'Z_2'))
node = Node(
    name='Z_3',
    parents=['Z_1', 'Z_2'],
    output_distribution='bernoulli',
    dist_params_coefs={
        'p_': {
            'bias': 0, 'linear': np.array([0, 0]), 'interactions': np.array([1])
        }
    }
)

samples = node.sample(data=data, size=500)
print(samples[:5])
# [1 0 0 0 0]
print(samples.mean())
# 0.23



data = pd.DataFrame(np.random.normal(0, 3, size=(100, 2)), columns=('X_1', 'X_2'))
node = Node(
    name='X_3',
    parents=['X_1', 'X_2'],
    output_distribution='bernoulli',
    dist_params_coefs={
        'p_': {
            'bias': 0, 'linear': np.array([0, 0]), 'interactions': np.array([1])
        }
    }
)

samples = node.sample(data=data, size=500)
# ValueError: probabilities are not non-negative
