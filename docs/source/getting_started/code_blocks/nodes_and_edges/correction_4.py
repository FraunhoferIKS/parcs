from pyparcs import Description, Graph

description = Description({'A': 'normal(mu_=0, sigma_=1)',
                           'B': 'bernoulli(p_=2A), correction[lower=0, upper=1, target_mean=0.3]',
                           'A->B': 'identity()'})
graph = Graph(description)
samples, _ = graph.sample(1000)

print(samples['B'].mean())
# 0.319
