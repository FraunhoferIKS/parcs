from pyparcs.graph_builder.parsers import graph_description_parser
import numpy as np
np.random.seed(2022)

description = {
    'C': 'gaussian(mu_=0, sigma_=1)',
    'A': 'gaussian(mu_=2C-1, sigma_=1)',
    'Y': 'gaussian(mu_=C+A-0.3AC, sigma_=2)',
    'A->Y': 'gaussian_rbf(alpha=1, beta=0, gamma=0, tau=2)'
}

nodes, edges = graph_description_parser(description, infer_edges=True)

print([e['name'] for e in edges])
# ['A->Y', 'C->A', 'C->Y']

print([e['function_name'] for e in edges])
# ['gaussian_rbf', 'identity', 'identity']