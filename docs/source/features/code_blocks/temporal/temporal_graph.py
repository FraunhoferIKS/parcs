from pyparcs.temporal import TemporalDescription
from pyparcs import Graph
import numpy as np
np.random.seed(42)

description = TemporalDescription('temporal_graph_description.yml', n_timesteps=3)
graph = Graph(description)

samples, _ = graph.sample(size=5)
print(samples)
#          Age       BP_0  Drug_0  Drug_neg1  ...  Drug_1       BP_3  Drug_2  Drug_3
# 0   4.023505   9.160127     0.0        0.0  ...     1.0   4.571410     1.0     1.0
# 1  15.981643  10.016958     0.0        0.0  ...     1.0  10.498443     1.0     1.0
# 2   0.496344  10.670488     0.0        0.0  ...     1.0  12.019636     1.0     1.0
# 3  12.139462   9.377000     0.0        0.0  ...     1.0   9.942749     1.0     1.0
# 4  11.713087   9.016991     0.0        0.0  ...     1.0   7.224285     1.0     1.0
#
# [5 rows x 13 columns]
