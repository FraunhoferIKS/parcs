.. _create_the_first_graph:

Create the First Graph
======================

Based on the description above, we can define a causal DAG in PARCS by specifying the output distributions and parameter coefficient vectors of the nodes, along with edges and their edge functions. This can be done using the :func:`~parcs.cdag.graph_objects.Graph` class, and a ``.yml`` description file.

.. literalinclude:: examples/graph_example_1/graph_description.yml
    :caption: :code:`graph_description.yml`
    :linenos:

.. literalinclude:: examples/graph_example_1/graph.py
    :caption: :code:`graph.py`
    :linenos:
    :lines: 1-13

The :code:`graph_file_parser` function, outputs *node* and *edge* objects to instantiate a :code:`Graph` class. These objects are lists of Python dicts, including all information given by the description file. See "here" for a detailed explanation. The output of the graph object is a :code:`Pandas.DataFrame` object.

.. seealso::
    1. Read more on :ref:`graph file parser conventions <conventions_graph_description_file>`
    2. List of :ref:`available output distributions <available_output_distributions>`
    3. List of :ref:`available edge functions <available_edge_functions>`

.. _get_started_sampling_error_terms:

Sampling error terms
--------------------

The math library in PARCS that supports sampling from the distributions is Scipy. Instead of using the :code:`.rvs()` methods of the distributions, however, PARCS samples *error* terms from a :math:`\text{Unif}(0, 1)` per each data row, and passes the realization to the :code:`.ppf()` method to obtain the corresponding sample from the target distribution. (read more about :ref:`internal mechanics of nodes <node_doc>`). With this procedure, we follow two main goals:

* Reproducibility is explicitly handled by returning and re-using the error terms.
* These error terms enable us to run simulate for counterfactual analysis. We explain this goal in the section below.

Considering the error terms, the :code:`.sample()` method receives the following options:

.. literalinclude:: examples/graph_example_1/graph.py
    :linenos:
    :lines: 15-35

Node correction
---------------
The library provides a method to apply a standardizing *correction* to the distribution parameters. This option is available for all parameters that have a non-real numbers support, such as success probability of the Bernoulli distribution.

.. literalinclude:: examples/correction_examples/graph_description.yml
    :linenos:
    :caption: graph_description.yml

.. literalinclude:: examples/correction_examples/correction_1.py
    :linenos:
    :caption: graph.py

To activate the correction, we can add an extra :code:`correction[...]` term to the line of the node.

.. literalinclude:: examples/correction_examples/graph_description_correction.yml
    :linenos:
    :caption: graph_description_correction.yml
    :emphasize-lines: 4

.. literalinclude:: examples/correction_examples/correction_2.py
    :linenos:
    :caption: graph.py

In this example, the correction transformation for Bernoulli distribution, is a Sigmoid function, mapping the real values to the [lower, upper] range. As a result, the success probability of the example has the form :math:`\sigma(10A+10C)`. see HERE for more details about blah blah.