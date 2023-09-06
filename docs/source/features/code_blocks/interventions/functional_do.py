from pyparcs import Description, Graph
import numpy as np
np.random.seed(42)

description = Description({'C': 'normal(mu_=0, sigma_=1)',
                           'A': 'normal(mu_=2C-1, sigma_=1)',
                           'Y': 'normal(mu_=C+0.6A, sigma_=1)'},
                          infer_edges=True)
graph = Graph(description)

samples, _ = graph.do_functional(
    size=3,
    intervene_on='Y', inputs=['A', 'C'],
    func=lambda a, c: (a+c)*10
)
print(samples)
#           C         A          Y
# 0  1.651819  1.983786  36.356056
# 1 -1.010956 -2.772037 -37.829930
# 2  1.108496 -0.354075   7.544213
