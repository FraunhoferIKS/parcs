from pyparcs import Description, Graph

description = Description({'A': 'normal(mu_=0, sigma_=1)',
                           'B': 'bernoulli(p_=2A), correction[]',
                           'A->B': 'identity()'})
graph = Graph(description)
samples, _ = graph.sample(5)
print(samples)
#           A    B
# 0 -0.135696  0.0
# 1  0.085163  0.0
# 2  0.428591  1.0
# 3 -0.586112  0.0
# 4  1.222686  0.0
