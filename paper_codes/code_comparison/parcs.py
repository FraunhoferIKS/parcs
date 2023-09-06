from pyparcs import Graph, Description, RandomDescription, Guideline

# Setup guidelines
guide_c = Guideline('./setup/guideline_c.yml')
guide_t = Guideline('./setup/guideline_t.yml')
guide_y_1 = Guideline('./setup/guideline_y.yml')
guide_y_2 = Guideline('./setup/guideline_y_edge.yml')
guide_connection = Guideline('./setup/connection_guideline.yml')

# define the subgraphs
desc_y = Description('./setup/description_y.yml')
desc_t = RandomDescription(guide_t, node_prefix='T')
desc_c = RandomDescription(guide_c, node_prefix='C')

# randomize Y except the Y_1->Y_2 edge
desc_y.randomize_parameters(guide_y_1)
desc_y.randomize_parameters(guide_y_2, tag='P1')

# make subgraph connections
graph_description = desc_c
graph_description.randomize_connection_to(desc_t.outline, guide_connection)
graph_description.randomize_connection_to(desc_y.outline, guide_connection)

# instantiate the graph
graph = Graph(graph_description)
samples, _ = graph.sample(100000)
