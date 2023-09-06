from pyparcs import Description, Graph
import numpy
numpy.random.seed(42)

outline = {'A': 'normal(mu_=0, sigma_=1)',
           'B': 'bernoulli(p_=0.4)',
           'C': 'normal(mu_=2A+B^2-1, sigma_=1)'}

description = Description(outline, infer_edges=False)

# pyparcs.core.exceptions.DescriptionError: The term 2A in the description file is invalid.
#     Another possibility is that a node name exists in another node's parameter,
#     while it is not marked as a parent by the described edges; in this case, check for the
#     nodes/edges consistency.
