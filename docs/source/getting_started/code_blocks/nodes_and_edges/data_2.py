from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'A': 'data(./dataset.csv, A)',
                           'B': 'data(./dataset.csv, B)',
                           'C': 'normal(mu_=A+B, sigma_=1)',
                           'A->C': 'identity()',
                           'B->C': 'identity()'})
graph = Graph(description)
samples, errors = graph.sample(full_data=True)

print(samples)
#     A    B           C
# 0  10  100  110.618855
# 1  11  101  110.988943
# 2  12  102  114.256234

print(errors)
#      A    B         C
# 0  0.0  0.0  0.731994
# 1  0.5  0.5  0.155995
# 2  1.0  1.0  0.601115
