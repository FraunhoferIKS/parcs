from pyparcs import Graph, Description, Guideline
from time import time
import seaborn as sns
from pandas import DataFrame
from matplotlib import pyplot as plt
import numpy
sns.set_theme()
numpy.random.seed(2023)

n_data = [100, 1000, 5000, 10_000, 20_000, 50_000, 100_000, 200_000, 500_000, 1_000_000, 2_000_000]
n_labels = ['100', '1K', '5K', '10K', '20K', '50K', '100K', '200K', '500K', '1M', '2M']
elapsed = []

for n in n_data:
    start = time()
    guide_y_1 = Guideline('./setup/guideline_y.yml')
    guide_y_2 = Guideline('./setup/guideline_y_edge.yml')

    # define the subgraphs
    desc_y = Description('./setup/description_y.yml')
    desc_y.randomize_parameters(guide_y_1)
    desc_y.randomize_parameters(guide_y_2, tag='P1')

    # instantiate the graph
    graph = Graph(desc_y)
    samples, _ = graph.sample(n)
    elapsed.append(time()-start)

to_plot = DataFrame({'x': n_labels, 'y': elapsed})
to_plot.to_csv('./parcs_y_only.csv')
# sns.lineplot(to_plot, x='x', y='y')
# plt.show()