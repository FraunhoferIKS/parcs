========
M-Graphs
========

M-graphs are causal DAG models for missing data, where the binary `indicator` node `R_x` defines if node `X` is missing or not. The `missingness mechanism` (i.e. what are the causes of missing values) are then determined by incoming edges to the indicator node. In PARCS, a simple function called ``m_graph_convert`` returns the corresponding m-graph for a graph object. It takes the indicator variables and mask the data based on the indicator realizations

.. literalinclude:: code_blocks/b5/graph_description.yml
    :linenos:
    :emphasize-lines: 5

.. literalinclude:: code_blocks/b5/graph.py
    :linenos:
    :emphasize-lines: 16

In order to use the function, you need to define the indicators of variables with a specific `prefix`, such as ``R_``.

.. warning::
    Since the function ``m_graph_convert`` doesn't have access to the description file, and only reads the sample data, it is up to the user to comply with the M-graph assumptions and restrictions, e.g. not having an edge from indicator nodes to main variables.