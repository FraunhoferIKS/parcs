from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.randomizer import FreeRandomizer, guideline_iterator

for dir_, epoch, value in guideline_iterator(guideline_dir='simple_guideline.yml',
                                             to_iterate='graph/num_nodes',
                                             repeat=2):
    print('num_nodes:', value)
    print('\t EPOCH:', epoch)

    rndz = FreeRandomizer(
        guideline_dir=dir_
    )
    nodes, edges = rndz.get_graph_params()
    g = Graph(nodes=nodes, edges=edges)
    samples = g.sample(size=1)
    print(samples)
    print('=====')

# num_nodes: 1
# 	 EPOCH: 0
#         H_0
# 0  2.097445
# =====
# num_nodes: 1
# 	 EPOCH: 1
#         H_0
# 0  1.964081
# =====
# num_nodes: 2
# 	 EPOCH: 0
#    H_0       H_1
# 0  1.0 -1.779816
# =====
# num_nodes: 2
# 	 EPOCH: 1
#    H_0  H_1
# 0  1.0  1.0
# =====
