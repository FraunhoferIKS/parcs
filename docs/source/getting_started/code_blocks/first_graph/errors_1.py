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

print(errors)
#           A         B         C
# 0  0.374540  0.950714  0.731994
# 1  0.598658  0.156019  0.155995
