from pyparcs import Description, Graph, Guideline
from matplotlib import pyplot as plt
import seaborn as sns
import numpy
numpy.random.seed(42)

guideline = Guideline(
    {'nodes': {'normal': {'mu_': [['f-range', -2, 2], 0, 0],
                          'sigma_': [['f-range', 1, 4], 0, 0]},
               'bernoulli': {'p_': [['f-range', -1, 1], ['f-range', -5, -3, 3, 5], 0]}}}
)

f, axes = plt.subplots(1, 3, sharey=True)
for i in range(3):
    description = Description({'A_1': 'normal(mu_=?, sigma_=?)',
                               'A_2': 'normal(mu_=?, sigma_=?)',
                               'Y': 'bernoulli(p_=?), correction[target_mean=0.5]',
                               'A_1->Y': 'identity()', 'A_2->Y': 'identity()'})
    description.randomize_parameters(guideline)
    graph = Graph(description)
    samples, _ = graph.sample(400)
    sns.scatterplot(samples, x='A_1', y='A_2', hue='Y', ax=axes[i])

plt.show()
