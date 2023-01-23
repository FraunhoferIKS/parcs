from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
from pyparcs.graph_builder.randomizer import ParamRandomizer

rndz = ParamRandomizer(
    graph_dir='graph_description_1.yml',
    guideline_dir='simple_guideline.yml'
)
nodes, edges = rndz.get_graph_params()

g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=3)
print(samples)
#           C         A    Y
# 0  1.660388  0.410814  1.0
# 1  1.253973 -2.983480  0.0
# 2  1.088486 -0.167692  1.0