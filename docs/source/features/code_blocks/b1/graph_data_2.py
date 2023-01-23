from pyparcs.cdag.graph_objects import Graph
from pyparcs.graph_builder.parsers import graph_file_parser
import pandas as pd
import numpy
numpy.random.seed(2022)

# create dummy data
df = pd.DataFrame({'C': [i for i in range(10, 13)], 'A': [i for i in range(100, 103)]})
df.to_csv('dummy_data.csv')

# main script
nodes, edges = graph_file_parser('graph_description_data_2.yml')
g = Graph(nodes=nodes, edges=edges)
samples = g.sample(size=3)
print(samples)
#      A   C           Y
# 0  102  12  112.859449
# 1  100  10  111.339772
# 2  102  12  115.306027