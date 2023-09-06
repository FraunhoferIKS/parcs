import numpy
import seaborn as sns
from matplotlib import pyplot as plt
from pyparcs import Graph, RandomDescription, Guideline
numpy.random.seed(2022)
sns.set_theme()

guideline = Guideline({
    'graph': {'num_nodes': 3, 'density': 1},
    'nodes': {'normal': {'mu_': [['f-range', -5, -2, 2, 5],
                                 ['f-range', -1, 1],
                                 ['f-range', -1, 1]],
                         'sigma_': [['f-range', 1, 2], 0, 0]}},
    'edges': {'identity': None,
              'arctan': {'alpha': ['f-range', 1, 3],
                         'beta': ['f-range', -0.5, -0.5],
                         'gamma': ['choice', 0, 1]}}
})

for _ in range(3):
    desc = RandomDescription(guideline, node_prefix='Z')
    g = Graph(desc)
    samples, _ = g.sample(500)

    sns.jointplot(data=samples, x="Z_0", y="Z_1")
    plt.show()
