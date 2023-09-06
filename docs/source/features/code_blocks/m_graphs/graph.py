from pyparcs.helpers.missing_data import m_graph_convert
from pyparcs import Description, Graph
import numpy as np
np.random.seed(2022)

description = Description({'C': 'normal(mu_=0, sigma_=1)',
                           'A': 'normal(mu_=2C-1, sigma_=1)',
                           'R_A': 'bernoulli(p_=C+A-0.3AC), correction[target_mean=0.3]'},
                          infer_edges=True)

graph = Graph(description)
samples, _ = graph.sample(5)
print(samples)
#           C         A  R_A
# 0  0.774417  2.049457  0.0
# 1 -0.652315 -1.713998  0.0
# 2  1.310389  3.075017  1.0
# 3  0.240281 -0.888637  0.0
# 4 -0.884086 -2.497936  0.0
print(m_graph_convert(samples, missingness_prefix='R_', shared_subscript=False))
#           C         A
# 0  0.774417       NaN
# 1 -0.652315       NaN
# 2  1.310389  3.075017
# 3  0.240281       NaN
# 4 -0.884086       NaN
