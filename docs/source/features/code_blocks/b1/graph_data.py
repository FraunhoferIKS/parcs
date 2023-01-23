from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import pandas as pd
import numpy
numpy.random.seed(2022)

# create dummy data
df = pd.DataFrame({'C': [i for i in range(10, 13)], 'B': numpy.random.normal(size=(3,))})
df.to_csv('dummy_data.csv')

# main script
nodes, edges = graph_file_parser('graph_description_data.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=3)
print(samples)
#     C          A          Y
# 0  11  22.144651  35.270247
# 1  12  24.062452  34.656983
# 2  10  18.592657  28.653529