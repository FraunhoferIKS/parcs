from pyparcs import Description, Graph
import numpy
numpy.random.seed(42)

description = Description({'A': 'normal(mu_=0, sigma_=1)',
                           'B': 'exponential(lambda_=A^2+1)'},
                          infer_edges=True)
graph = Graph(description)
samples, errors = graph.sample(1)
print(samples)
#           A         B
# 0 -0.319852  4.112427
print(errors)
#          A         B
# 0  0.37454  0.950714
new_description = Description({'A': 'normal(mu_=0, sigma_=1)',
                               'B': 'exponential(lambda_=1)'})
new_graph = Graph(new_description)
new_samples, new_errors = new_graph.sample(use_sampled_errors=True,
                                           sampled_errors=errors)
print(new_samples)
#           A         B
# 0 -0.319852  4.010121
print(new_errors)
#          A         B
# 0  0.37454  0.950714