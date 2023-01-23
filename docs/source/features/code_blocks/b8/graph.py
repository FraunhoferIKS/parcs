from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.randomizer import ConnectRandomizer
import pandas as pd
import numpy as np
np.random.seed(2022)


rndz = ConnectRandomizer(
    parent_graph_dir='graph_description_L.yml',
    child_graph_dir='graph_description_Z.yml',
    guideline_dir='simple_guideline.yml',
    adj_matrix_mask=pd.DataFrame(np.ones(shape=(3, 3)),
                                 index=('L_1', 'L_2', 'L_3'),
                                 columns=('Z_1', 'Z_2', 'Z_3'))
)
nodes, edges = rndz.get_graph_params()
g = Graph(nodes=nodes, edges=edges)
print(g.sample(4))
#         L_1       Z_1       L_2       L_3        Z_2        Z_3
# 0  0.192125 -0.523060 -1.064670 -1.345416  -4.565572  -5.873356
# 1 -1.383525  0.079110 -3.010643 -5.204452 -11.578450 -12.299672
# 2  1.675641 -1.298673  2.850752  4.498497   7.355618   5.460474
# 3 -0.546389  0.540718 -0.436297 -1.625131  -2.717709  -1.557761