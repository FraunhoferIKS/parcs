import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy

sns.set_theme()
numpy.random.seed(2023)

parcs = pd.read_csv('./parcs_y_only.csv')
vanilla = pd.read_csv('./vanilla.csv')
results = pd.DataFrame({'# samples': parcs.x, 'PARCS': parcs.y, 'plain': vanilla.y})


to_plot = pd.melt(results, ['# samples'])
to_plot.rename({'value': 'Elapsed time (Sec)', 'variable': 'Implementation'},
               inplace=True,
               axis=1)

sns.lineplot(to_plot, x='# samples', y='Elapsed time (Sec)', hue='Implementation')
plt.legend(fontsize=20)
plt.xlabel('# Samples', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Elapsed time (Sec)', fontsize=18)
plt.yticks(fontsize=18)
plt.show()
