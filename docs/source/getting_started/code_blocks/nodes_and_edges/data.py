from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'A': 'data(./dataset.csv, A)',
                           'B': 'data(./dataset.csv, B)',
                           'C': 'normal(mu_=A+B, sigma_=1)',
                           'A->C': 'identity()',
                           'B->C': 'identity()'})
graph = Graph(description)
samples, errors = graph.sample(5)

print(samples)
#     A    B           C
# 0  11  101  112.618855
# 1  11  101  110.988943
# 2  10  100  110.256234
# 3  12  102  115.879470
# 4  12  102  113.091568

print(errors)
#           A         B         C
# 0  0.374540  0.374540  0.731994
# 1  0.598658  0.598658  0.155995
# 2  0.058084  0.058084  0.601115
# 3  0.708073  0.708073  0.969910
# 4  0.832443  0.832443  0.181825
