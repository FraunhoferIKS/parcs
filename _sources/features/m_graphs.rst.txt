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

Using M-graph templates
=======================

Above, we showed how you can create an m-graph by determining a DAG as before. Another option is to allow PARCS to randomly set up a missingness mechanism for the dataset and induce the missingness. Many literature have proposed unique m-graph structures for missingness mechanisms. These structures determine which edges are allowed among `R`s and from `Z` to `R`s. By having two (sub) graphs of `Z` and `R`, we can use the :ref:`ConnectRandomizer <connect_randomizer>` to create outgoing edges from `Z` to `R`, while masking the edges according to one of the known m-graph structures. This way, `R->R` edges must be induced when creating `R`, and `Z->R` will be induced by the randomizer. An example is provided below:

Ground truth graph
------------------

We start from a `Z` graph. It can be a graph of data nodes, or simulated data. In this example, we use the following graph description file:

.. literalinclude:: code_blocks/b9/gdf_Z.yml
    :linenos:

Missingness indicators subgraph
-------------------------------

Nest, we determine the `R` subgraph. As mentioned, `R->R` edges must be defined here. We can again use a graph description file. But we can also use a helper which sets up an `R` subgraph for a given graph:

.. literalinclude:: code_blocks/b9/R_creator.py
    :linenos:

The parameters:

- ``adj_matrix`` determines the edges among R nodes. In this example, there is no edge among Rs
- ``node_name`` is the list of names for Z nodes
- ``prefix`` and ``subscript_only`` tells the function how to make names for R nodes. If `subscript_only=True` then the node names are `R_` and the subscript of Z nodes. If `False`, then the names will be `R_<Z node name>`
- ``file_dir`` is the directory of the created graph description file.

The resulting file is shown below:

.. literalinclude:: code_blocks/b9/gdf_R.yml
    :linenos:

As for the ``adj_matrix`` parameter, PARCS again provides helper functions to induce different missingness mechanisms:

- ``R_adj_matrix(size, shuffle, density)`` freely randomize edges among Rs to induce an acyclic structure.
- ``R_attrition_adj_matrix(size, step, density)`` gives an adjacency matrix which induces `attrition` missingness, i.e. edges from early Rs to later Rs. ``step`` determines how many Rs can an indicator affect. e.g. if ``step=2`` then `R_1` will have an edge to `R_2` and `R_3` but not anymore.

.. note::
    In order to make post-hoc changes, simply edit the resulting graph description file. for example you can set ``target_mean`` parameter to the ``correction`` of edges in order to control the ratio of missingness.

Creating `Z->R` edges
---------------------

**Next** step is to use the `ConnectRandomizer` to create the `Z->R` edges. Here we apply masks in order to determine the missingness mechanism. These masks can be made manually, or read from the `helper` module as well:

- ``fully_observed_mar`` creates an `p x q` mask matrix where `p` and `q` are length of Z and R node vectors respectively, and `p - q` determine the number of fully observed Z nodes (they have no R). with the parameter ``fully_observed_indices`` we specify the index of fully observed nodes.
- ``nsc_mask`` allows for no-self censoring mechanism, i.e. all the `Z->R` edges are allowed except for `Z_i->R_i` edges (diagonal is zero)
- ``sc_mask`` is an identity matrix, allowing only for self-censoring edges
- ``block_conditional_mask`` is an upper triangular mask where each Z can have edge to only later Rs (not previous ones). This mask assumes an order in Z nodes (e.g. a chronological order).

Finally, we sample from the constructed graph as follows:

.. literalinclude:: code_blocks/b9/graph.py
    :linenos: