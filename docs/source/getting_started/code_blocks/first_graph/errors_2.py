from pyparcs import Description, Graph
import numpy
numpy.random.seed(42)

description = Description('./outline.yml')
graph = Graph(description)
samples, errors = graph.sample(2)
print(samples)
#           A    B         C
# 0 -0.319852  1.0 -0.020850
# 1  0.249876  0.0 -1.511305

new_samples, _ = graph.sample(use_sampled_errors=True, sampled_errors=errors)
print(new_samples)
#           A    B         C
# 0 -0.319852  1.0 -0.020850
# 1  0.249876  0.0 -1.511305