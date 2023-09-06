from pyparcs import Description, Graph

description = Description({'A': 'normal(mu_=0, sigma_=1)',
                           'B': 'exponential(lambda_=A^2)',
                           'A->B': 'identity()'})
graph = Graph(description)
samples, _ = graph.sample(5)

print(samples)
#           A         B
# 0  0.285523  0.104599
# 1  0.703491  2.052430
# 2 -1.074570  1.325737
# 3  0.306434  0.275022
# 4  1.354107  2.695944
