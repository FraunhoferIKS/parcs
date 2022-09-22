==================
More Graph Objects
==================

As mentioned in :ref:` the first steps <create_the_first_graph>`, the main objects are nodes which are defined by standard probability distributions and edges that are defined by PARCS edge functions. In this section we introduce other types of nodes and edges that you can use in your PARCS simulation


Deterministic nodes
===================

A node can be defined to output a deterministic function of its parents. In this case, a deterministic function replaces the output distribution. To define a deterministic node:

1. create a `.py` file in the same directory as your main Python script
2. Define your custom function in the file. The input must be considered the dataset pandas data frame. (see example below)
3. define the node in the `description file` by ``deterministic(<.py file name>, <function name>)``

.. literalinclude:: code_blocks/b1/customs.py
    :caption: :code:`customs.py`
    :linenos:

.. literalinclude:: code_blocks/b1/graph_description.yml
    :caption: :code:`graph_description.yml`
    :linenos:

.. literalinclude:: code_blocks/b1/graph.py
    :caption: :code:`graph.py`
    :linenos:

While defining the function, we consider the input to be the simulated data of the graph, and call the columns by the same node names as in graph description file. Note that you can write your function using Pandas functionalities, e.g. column operations, or for loops using pandas `iterrows <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.iterrows.html>`_. The output must be a pandas ``Series``.

.. note::
    The data passed to the custom function only includes data of the parent nodes (inferred from the description file). In order to do interventions and calculate the node using other parents, see :ref:`interventions docs <functional_intervention>`

.. warning::
    Deterministic nodes are designed for simulating variables that are deterministic given the parent variables. Even though it is technically possible to perform stochastic sampling inside the function (e.g. ``return np.random.normal(data['A']+data['C'], 1)``) it is not recommended to, as the stochasticity cannot be controlled by PARCS.


Constant nodes
==============

This class of nodes provide a simple way to add a constant node to the graph. The syntax in the description file as simple as creating a node by ``constant(<value>)``

.. literalinclude:: code_blocks/b1/graph_description_const.yml
    :caption: :code:`graph_description_const.yml`
    :linenos:

.. literalinclude:: code_blocks/b1/graph_const.py
    :caption: :code:`graph_const.py`
    :linenos:

.. note::
    We can simulate deterministic and constant nodes by `hacking` stochastic ``gaussian`` node with ``mean_`` being a desired constant or determinist term, and ``sigma_`` being 0. Using the deterministic and constant nodes, however, adds more to the clarity of the simulation and the description file.


Data nodes
==============

This class of nodes allows us to read an external CSV file for the samples. The file should be given to PARCS by its directory. To provide an example, we assume there is already a CSV file (created in the first lines of `graph.py` file):

.. literalinclude:: code_blocks/b1/graph_description_data.yml
    :caption: :code:`graph_description_data.yml`
    :linenos:

.. literalinclude:: code_blocks/b1/graph_data.py
    :caption: :code:`graph.py`
    :linenos:

Two important notes:

- Data nodes must have no incoming edges in the graph description file
- If the graph contains at least one data node, then the boolean `with_replacement` parameter is needed, in order to determine the sampling behaviour for the data nodes. If `size` parameter is larger than the size of the data node and `with_replacement=False`, sampling raises an error.

sampling from interventional distributions using `.do()`, ... methods are similar to `.sample()`.