import numpy as np
import pandas as pd
from pyparcs.cdag.graph_objects import Node
np.random.seed(1)

size_ = 100
data = pd.DataFrame(np.random.normal(0, 3, size=(size_, 2)), columns=('Z_1', 'Z_2'))
node = Node(
    name='Z_3',
    parents=['Z_1', 'Z_2'],
    output_distribution='bernoulli',
    dist_params_coefs={
        'p_': {
            'bias': 0, 'linear': np.array([0, 0]), 'interactions': np.array([1])
        }
    },
    do_correction=True,
    dist_config={
        'correction_config': {
            'p_': {'lower': 0, 'upper': 1}
        }
    }
)

errors = np.random.uniform(0, 1, size=size_)
samples = node.calculate(data, errors)
print(np.round(samples.mean(), 2))
# 0.62