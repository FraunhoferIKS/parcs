from pyparcs import Description, Graph

description = Description({'A_dummy': 'normal(mu_=0, sigma_=1), tags[D]',
                           'A': 'poisson(lambda_=A_dummy^2+1)',
                           'B': 'bernoulli(p_=A), correction[]',
                           'A_dummy->A': 'identity()',
                           'A->B': 'identity()'})
graph = Graph(description)
samples, errors = graph.sample(3)
print(samples)
#      A    B
# 0  1.0  0.0
# 1  2.0  1.0
# 2  5.0  1.0
print(errors)
#           A   A_dummy         B
# 0  0.539704  0.723826  0.236487
# 1  0.829672  0.600388  0.479554
# 2  0.476101  0.977471  0.635977
