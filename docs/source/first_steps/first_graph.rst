.. _create_the_first_graph:

Create the First Graph
======================

We define a causal DAG in PARCS by specifying the output distributions and parameter coefficient vectors of the nodes, along with edges and their edge functions. This can be done using the :func:`~parcs.cdag.graph_objects.Graph` class, and a ``.yml`` description file.

.. literalinclude:: code_blocks/b1/graph_description.yml
    :caption: :code:`graph_description.yml`
    :linenos:

In the description file, names like ``...->...`` define edges, and the remaining lines are considered as nodes. What comes afterward is the a distribution for nodes, and an edge function for edges.

.. seealso::
    1. Read more about :ref:`graph file parser conventions <conventions_graph_description_file>`
    2. List of :ref:`available output distributions and edge functions <function_list>`

The description file then is used to instantiate a graph object.

.. literalinclude:: code_blocks/b1/graph.py
    :caption: :code:`graph.py`
    :linenos:
    :lines: 1-15

The ``graph_file_parser`` function converts the description lines to standard *node* and *edge* objects needed by the ``Graph`` class. These objects are lists of Python dictionaries. Finally, the ``.sample()`` method return a ``Pandas.DataFrame`` object.

.. _sampling_error_terms:

Sampling error terms
--------------------

The math library in PARCS that supports sampling from the distributions is Scipy. Instead of using the :code:`.rvs()` methods of the distributions, however, PARCS samples *error* terms from a :math:`\text{Unif}(0, 1)` per each data row, and passes the realization to the :code:`.ppf()` method to obtain the corresponding sample from the target distribution. (read more about :ref:`internal mechanics of nodes <node_doc>`). With this procedure, we follow two main goals:

* Reproducibility is explicitly handled by returning and re-using the error terms.
* These error terms enable us to run simulate for counterfactual analysis. We explain this goal in the section below.

To reuse the errors, we set ``return_errors=True``, and pass the returned dataframe to ``sampled_errors`` in the next run:

.. literalinclude:: code_blocks/b1/graph.py
    :linenos:
    :lines: 17-37

.. note::
    We still can control the random number generation in the main script by |numpy_seed|_. In this case, the sampled errors will also be affected by the fixed seed, while reproducibility of simulation regarding the data rows is still controlled by the error terms.
.. |numpy_seed| replace:: ``numpy.random.seed()``
.. _numpy_seed: https://numpy.org/doc/stable/reference/random/generated/numpy.random.seed.html

.. _node_correction:

Node correction
---------------
PARCS provides a method to apply a standardizing *correction* to the distribution parameters. This option is available for all parameters that have a non-real numbers support, such as success probability of the Bernoulli distribution. Here is an example which runs into a problem, as the passed values to ``p_`` of the Bernoulli distribution are not in the range of :math:`[0, 1]`.

.. literalinclude:: code_blocks/b2/graph_description.yml
    :linenos:
    :caption: graph_description.yml

.. literalinclude:: code_blocks/b2/correction_1.py
    :linenos:
    :caption: graph.py

To activate the correction, we add an extra :code:`correction[...]` term to the line of the node.

.. literalinclude:: code_blocks/b2/graph_description_correction.yml
    :linenos:
    :caption: graph_description_correction.yml
    :emphasize-lines: 4

.. literalinclude:: code_blocks/b2/correction_2.py
    :linenos:
    :caption: graph.py

In this example, the correction transformation for Bernoulli distribution, is a Sigmoid function, mapping the real values to the [lower, upper] range. As a result, the success probability of the example has the form :math:`\sigma(10A+10C)`. Read more about sigmoid correction at :func:`~parcs.cdag.utils.SigmoidCorrection`.