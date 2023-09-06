from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'A': 'bernoulli(p_=0.3)',
                           'B': 'normal(mu_=0, sigma_=1)',
                           'C_dummy': 'deterministic(./custom_functions.py, log_sin)',
                           'C': 'normal(mu_=C_dummy, sigma_=1)',
                           'A->C_dummy': 'identity()', 'B->C_dummy': 'identity()',
                           'C_dummy->C': 'identity()'})
graph = Graph(description)
samples, _ = graph.sample(2)
print(samples)
#      A         B   C_dummy         C
# 0  0.0  1.651819  2.204312  2.823166
# 1  0.0 -1.011057  3.559830  1.988763
