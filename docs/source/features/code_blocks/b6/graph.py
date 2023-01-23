from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.randomizer import FreeRandomizer

rndz = FreeRandomizer(
    guideline_dir='simple_guideline.yml'
)
nodes, edges = rndz.get_graph_params()

g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=3)
print(samples)
#         H_0  H_1       H_2       H_3
# 0  0.641136  1.0 -2.300603  0.343558
# 1 -2.007259  1.0  3.145062  1.988338
# 2  0.874383  1.0 -3.203189  1.714673