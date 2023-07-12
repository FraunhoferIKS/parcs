==================
More Graph Objects
==================

As mentioned in :ref:` the first steps <create_the_first_graph>`, the main objects are nodes which are defined by standard probability distributions and edges that are defined by PARCS edge functions. In this section we introduce other types of nodes and edges that you can use in your PARCS simulation


.. _deterministic nodes:

Deterministic nodes
===================

A node can be defined to output a deterministic function of its parents. In this case, a deterministic function replaces the output distribution. To define a deterministic node:

1. create a `.py` file in a directory which you will point to using relative/main paths. **NOTE** that under the hood, PARCS does `path.append()` to find the Python file, so be careful not to use an existing file name which is reachable in your current Python path.
2. Define your custom function in the file. The input must be considered the dataset pandas data frame. (see example below)
3. define the node in the `description file` by ``deterministic(<path/to/module>.py, <function name>)``

.. literalinclude:: code_blocks/b1/python_functions/customs.py
    :caption: :code:`./python_functions/customs.py`
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

.. seealso::
    Declaring a deterministic node in temporal graphs is a bit different. Please see


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

.. note::
    Data nodes must have no incoming edges in the graph description file

The sampling procedure based on the errors repeats for the data nodes again. Therefore we might have duplicate samples as the a row can be selected more than once (sotchastically). If more than one columns are read from the csv file, then the graph samples *jointly*. It does so by using a shared error vector for all the data nodes, leading to selection of the same rows simultaneously.

.. literalinclude:: code_blocks/b1/graph_description_data_2.yml
    :caption: :code:`graph_description_data_2.yml`
    :linenos:

.. literalinclude:: code_blocks/b1/graph_data_2.py
    :caption: :code:`graph.py`
    :linenos:

sampling from interventional distributions using `.do()`, ... methods are similar to `.sample()`.