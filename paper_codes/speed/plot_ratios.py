import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import numpy

sns.set_theme()
numpy.random.seed(2023)

parcs = pd.read_csv('./parcs_y_only.csv')
vanilla = pd.read_csv('./vanilla.csv')
results = pd.DataFrame({'# samples': parcs.x, 'Ratio': parcs.y/vanilla.y})

plot = sns.lineplot(results, x='# samples', y='Ratio', color='#38803d', label='PARCS/plain ratio')
plot.axhline(1.00, color='#7c7e82', linestyle='--', label='ratio = 1')
plt.legend(fontsize=20)
plt.xlabel('# Samples', fontsize=18)
plt.xticks(fontsize=18)
plt.ylabel('Elapsed time ratio', fontsize=18)
plt.yticks(fontsize=18)
plt.show()

print(results[results['# samples'] == '2M'])