from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'A': 'bernoulli(p_=0.3)',
                           'B': 'normal(mu_=0, sigma_=1)',
                           'C': 'deterministic(./custom_functions.py, log_sin)',
                           'A->C': 'identity()', 'B->C': 'identity()'})
graph = Graph(description)
samples, errors = graph.sample(2)
print(samples)
#      A         B         C
# 0  0.0  1.651819  2.204312
# 1  0.0 -1.010956  3.559511

print(errors)
#           A         B         C
# 0  0.374540  0.950714  0.731994
# 1  0.598658  0.156019  0.155995
