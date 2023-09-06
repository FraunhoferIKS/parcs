from pyparcs import Description, Graph

description = Description({'A': 'normal(mu_=0, sigma_=1)',
                           'B': 'bernoulli(p_=0.3)'})
graph = Graph(description)
samples, _ = graph.sample(5)
print(samples)
#           A    B
# 0 -1.527236  0.0
# 1  0.385980  1.0
# 2 -0.217019  0.0
# 3 -0.137873  0.0
# 4  0.462656  1.0