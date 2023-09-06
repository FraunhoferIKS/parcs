from pyparcs import Description, Graph
import numpy
numpy.random.seed(42)

outline = {'A': 'normal(mu_=0, sigma_=1)',
           'B': 'bernoulli(p_=0.4)',
           'C': 'normal(mu_=2A+B^2-1, sigma_=1)',
           'A->C': 'identity()',
           'B->C': 'identity()'}

description = Description(outline, infer_edges=False)
graph = Graph(description)
samples, _ = graph.sample(5)

# samples:
#           A    B         C
# 0 -0.319852  1.0 -0.020850
# 1  0.249876  0.0 -1.511305
# 2 -1.571066  1.0 -2.885899
# 3  0.547763  0.0  1.974996
# 4  0.963863  0.0  0.019293
