import numpy as np
import pandas as pd
from pprint import pprint
from pyparcs import Description, Graph, Guideline
from pyparcs.helpers.missing_data import indicator_outline, sc_mask, m_graph_convert
np.random.seed(42)


outline_R = indicator_outline(adj_matrix=np.zeros(shape=(4, 4)),
                              node_names=[f'Z_{i}' for i in range(1, 5)],
                              miss_ratio=0.5,
                              prefix='R',
                              subscript_only=True)
pprint(outline_R)
# {'R_1': 'bernoulli(p_=?), correction[target_mean=0.5]',
#  'R_2': 'bernoulli(p_=?), correction[target_mean=0.5]',
#  'R_3': 'bernoulli(p_=?), correction[target_mean=0.5]',
#  'R_4': 'bernoulli(p_=?), correction[target_mean=0.5]'}

mask = pd.DataFrame(sc_mask(size=4),
                    index=[f'Z_{i}' for i in range(1, 5)],
                    columns=[f'R_{i}' for i in range(1, 5)])
guideline = Guideline('simple_guideline.yml')

description = Description('outline_Z.yml')
description.randomize_connection_to(outline_R, guideline,
                                    mask=mask)

graph = Graph(description)
samples, _ = graph.sample(1000)
masked_samples = m_graph_convert(samples, shared_subscript=True)

print(masked_samples.sample(4))
#           Z_1       Z_2  Z_4       Z_3
# 795  0.570035 -1.417940  1.0       NaN
# 354       NaN       NaN  NaN  0.724395
# 538       NaN       NaN  NaN       NaN
# 516       NaN -0.022907  NaN -1.825223
print(masked_samples.notna().mean())
# Z_3    0.503
# Z_4    0.493
# Z_1    0.473
# Z_2    0.482
