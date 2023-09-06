from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'C': 'normal(mu_=0, sigma_=1)',
                           'A': 'normal(mu_=2C-1, sigma_=1)',
                           'Y': 'normal(mu_=C+0.6A, sigma_=1)'},
                          infer_edges=True)
graph = Graph(description)

samples, errors = graph.sample(3)
print(samples)
#           C         A         Y
# 0  1.651819  1.983786  3.460946
# 1 -1.010956 -2.772037 -3.685236
# 2  1.108496 -0.354075  1.152285

intrv_samples, _ = graph.do_self(
    func=lambda a: a+1, intervene_on='A',
    use_sampled_errors=True, sampled_errors=errors
)
print(intrv_samples)
#           C         A         Y
# 0  1.651819  2.983786  4.060946
# 1 -1.010956 -1.772037 -3.085236
# 2  1.108496  0.645925  1.752285
