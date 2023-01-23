from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.randomizer import ConnectRandomizer
import pandas as pd
from matplotlib import pyplot as plt
from pyparcs.helpers.missing_data import sc_mask, m_graph_convert
import numpy as np
np.random.seed(2022)

mask = pd.DataFrame(sc_mask(size=4),
                    index=['Z_{}'.format(i) for i in range(1, 5)],
                    columns=['R_{}'.format(i) for i in range(1, 5)])

rndz = ConnectRandomizer(
    parent_graph_dir='gdf_Z.yml',
    child_graph_dir='gdf_R.yml',
    guideline_dir='simple_guideline.yml',
    delete_temp_graph_description=False,
    adj_matrix_mask=mask
)
nodes, edges = rndz.get_graph_params()
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(1000)
masked_samples = m_graph_convert(samples, shared_subscript=True)

print(masked_samples.sample(4))
#           Z_1       Z_2       Z_3  Z_4
# 430  0.328718       NaN       NaN  NaN
# 322       NaN  1.291225  4.473324  1.0
# 742  1.393201       NaN       NaN  1.0
# 271       NaN  0.031668       NaN  NaN
print(masked_samples['Z_4'].notna().mean())
# 0.229
