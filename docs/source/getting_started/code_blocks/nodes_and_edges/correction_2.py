from pyparcs import Description, Graph

description = Description({'A': 'normal(mu_=0, sigma_=1)',
                           'B': 'bernoulli(p_=2A)',
                           'A->B': 'identity()'})
graph = Graph(description)
samples, _ = graph.sample(5)
print(samples)
# pyparcs.core.exceptions.DistributionError: Bern(p) probabilities are out of [0, 1] range