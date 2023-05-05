from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import temporal_graph_file_parser
import numpy as np
np.random.seed(2022)

nodes, edges = temporal_graph_file_parser('temporal_graph_description.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=5)
print(samples)

#   Age         BP_0	    Drug_0	Drug_neg1	noise_1	    noise_2	    noise_3	    BP_1	    BP_2	    Drug_1  BP_3	    Drug_2  Drug_3
# 0 14.338103   9.713334    0.0     0.0	        -1.716690	0.087874	0.902874	9.010290	8.139407	1.0	    9.972034	1.0	    1.0
# 1 5.995419	9.523038	0.0	    0.0	        0.032830	-0.769881	-0.053133	10.221624	8.987490	1.0	    5.414739	1.0 	1.0
# 2 1.446751	10.957976	0.0	    0.0	        -1.878440	0.975674	-0.111324	9.844053	11.332916	1.0	    11.903063	1.0	    1.0
# 3 19.496837	10.644655	0.0	    0.0	        1.053952	0.051638	1.727255	11.855923	13.186312	1.0	    13.789486	1.0	    1.0
# 4	19.052279	10.545336	0.0	    0.0	        0.641653	0.553784	0.880378	10.535430	10.936217	1.0	    11.323597	1.0	    1.0