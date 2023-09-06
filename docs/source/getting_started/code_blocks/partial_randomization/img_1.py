from pyparcs import Description, Graph
from matplotlib import pyplot as plt
import seaborn as sns
import numpy
numpy.random.seed(42)

description = Description({'A_1': 'normal(mu_=2, sigma_=1)',
                           'A_2': 'normal(mu_=3, sigma_=1)',
                           'Y': 'bernoulli(p_=A_1+2A_2), correction[target_mean=0.5]'},
                          infer_edges=True)
graph = Graph(description)
samples, _ = graph.sample(500)

sns.scatterplot(samples, x='A_1', y='A_2', hue='Y')
plt.show()
