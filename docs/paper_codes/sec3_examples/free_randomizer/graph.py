from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.randomizer import FreeRandomizer
import numpy as np
np.random.seed(2022)

rndz = FreeRandomizer(guideline_dir='guideline.yml')
nodes, edges = rndz.get_graph_params()

g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=2)
print(samples.round(2))