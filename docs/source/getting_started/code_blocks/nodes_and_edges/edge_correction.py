from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'A': 'normal(mu_=100, sigma_=10)',
                           'B': 'normal(mu_=A, sigma_=1)',
                           'A->B': 'identity()'})
graph = Graph(description)
samples, errors = graph.sample(3)
#             A           B
# 0   96.801476   98.453296
# 1  106.188546  106.438423
# 2   89.890436   88.879378

description = Description({'A': 'normal(mu_=100, sigma_=10)',
                           'B': 'normal(mu_=A, sigma_=1)',
                           'A->B': 'identity(), correction[]'})
graph = Graph(description)
samples, _ = graph.sample(use_sampled_errors=True,
                          sampled_errors=errors)
#             A         B
# 0   96.801476  1.445580
# 1  106.188546  0.967103
# 2   89.890436 -1.897181
