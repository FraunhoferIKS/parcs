=============
Graph objects
=============

In Getting started document we explained how to make a graph using a graph description file, how to sample from observational and interventional distributions, and how to randomize it partially. This is the most efficient way to utilize PARCS. Nevertheless, it might be beneficial if we get to know the nuts and bolts of the graph object.

In PARCS two main objects make DAGs: *nodes*, *edges*

.. _edge_doc:

Edges
=====

An ``Edge`` object is primarily defined by a pair of parent-child nodes :math:`(Z_i, Z_j)` and an *edge function* :math:`e_{ij}(.)` along with its parameters. The edge function maps the input values from the parent node and create a transformed array, to be then passed to the child node as the actual input.

.. math::
    \begin{align}
    z^*_i = e_{ij}(z_i)
    \end{align}
    :label: edge-1

An edge object is declared via the following code:

.. literalinclude:: examples/cdag_examples/edge_identity.py
    :lines: 1-14

where ``function_name`` argument determines the selected edge function. Other examples are provided below.

.. seealso::
    List of :ref:`available edge functions <available_edge_functions>`

.. literalinclude:: examples/cdag_examples/edge_sigmoid.py
    :lines: 4-17

.. literalinclude:: examples/cdag_examples/edge_gaussian_rbf.py
    :lines: 4-17


Input normalization
-------------------
During randomization of simulation in PARCS, we need to normalize the inputs before applying the edge function. by setting ``do_correction=True``, the edge object calculates the empirical mean and standard deviation of the input data, and normalizes it as

.. math::
    \begin{align}
        \tilde{z}_i = \frac{z_i - \mu_q}{\sigma_q}
    \end{align}
    :label: edge_correction

where the subscript :math:`q` means that the statistics are calculated for the truncated input, excluding first and last *q* quantiles. This is done to prevent improper normalization in the presence of outliers. By default, the quantile is set to :math:`0.05`

.. literalinclude:: examples/cdag_examples/edge_correction.py
    :lines: 4-22

For the next data batch, edge uses the calculated statistics from the first feed. Therefore this normalization becomes the characteristics of the simulation model and stays fixed for all further sampling.

.. note::
    Since the statistics are calculated on the first batch, PARCS throws an instability warning if the first batch is ``<500``.

.. warning:: requires *see also* section which points to randomizer section

Parent-Child node names
-----------------------
We can also declare the parent and child name of the edge upon instantiation. These attributes will be consumed by the graph as the index of the edge.

.. literalinclude:: examples/cdag_examples/edge_identity.py
    :lines: 20-25

However, when an edge object is instantiated independently and outside a graph object, it is not necessary to give any node names

.. seealso:: :ref:`Edge API reference <edge_api>`

.. _node_doc:

Nodes
=====

A ``Node`` object is primarily defined by an *output distribution* :math:`\mathcal{P}_j(\theta_1, \dots, \theta_K)`, and *parameter coefficient vectors* :math:`W_j^{(\theta_k)}`. The sampling algorithm for a node is defined below:

1. Given a set of parents for the node :math:`Z_j` as :math:`(Z_1, \dots, Z_M)` with sampled values for 1 instance, an *input dictionary* :math:`S_j` is defined as concatenation of a bias element (:math:`1`), all parent values and :math:`m`-wise multiplications, where :math:`m = \{2, \dots, M\}`. As an example, for two parents :math:`\{Z_1, Z_2, Z_3\}` the input dictionary will be :math:`s_j = (z_1, z_2, z_3) \cup (z_1 z_2, z_1z_3, z_2z_3) \cup (z_1z_2z_3)`
2. The dot product of :math:`s_j.W_j^{(\theta_k)}` gives the calculated :math:`\theta_k` parameter for :math:`\mathcal{P}_j`. We calculate this dot product for all parameters.
3. To finish the sampling process, we sample from :math:`\mathcal{P}_j(\Theta)` once, and yield it as the sampled :math:`z_j` for that specific instance.

.. warning::
    requires equation \\
    requires schematic

Based on the data type (continuous, binary, etc.) and the desired underlying distribution for a node, we can choose an output distribution from provided PARCS distributions. Subsequently, we need to define the parameter coefficient vectors according to the number of parents and distributions.

Sampling a source node
----------------------
A source node in a graph, is a node with no parents. The code below, creates a source node with Bernoulli distribution and the success probability of :math:`0.7`.

.. literalinclude:: examples/cdag_examples/node_basic.py
    :lines: 1-21

Explanations:

* the ``dist_params_coefs`` is a dictionary of distributions parameters. In case of the Bernoulli distribution, the only parameter is the success probability i.e. ``p_``. Three distinct key-value pairs define the coefficient vectors for each parameter. The keys are ``bias, linear, interactions``. In this example, the source node needs only the bias, so other vectors are set to empty. In next examples about non-source nodes, we provide more details regarding the coefficients.
* the ``data`` input is required for the ``.sample()`` method. In this case, as the node has no parents, the data frame can be empty.

.. seealso::
    List of :ref:`available output distributions <available_output_distributions>`

Sampling a Node with Parents
----------------------------

We repeat the previous process, but now for a node assuming it has two parents.

.. literalinclude:: examples/cdag_examples/node_parents.py
    :lines: 1-22

In the example above, we assumed the data frame is already filled with parent values, where both parents are uniformly distributed r.v.s. and the edges are set to identity function (in the next steps, we also simulate the parents as PARCS nodes). The ``dist_params_coefs`` implements :math:`P(Z_3=1|Z_1, Z_2) = Z_1Z_2`

.. note:: To create the interaction input vector elements, the order of permutation over the parent values follows the standard of Python's ``itertools.combinations`` function. As an example, the order for the case of three ``(X,Y,Z)`` nodes is: ``XY, XZ, XYZ``

Correction of distribution parameters
-------------------------------------
It is obvious that not all of the distribution parameters can be chosen freely. As an example, the Bernoulli's success probability must lie in :math:`[0, 1]` range, or the Gaussian normal's standard deviation must be a positive value. This means that arbitrary parent data types potentially lead to invalid parameter values. An example is shown below

.. literalinclude:: examples/cdag_examples/node_parents.py
    :lines: 26-39

For this reason, either the data type and the support of the parents must follow the restrictions, or a correction must be done. Correction is possible via ``do_correction=True`` argument, which in return tells the node object to read the ``correction_config`` key out of a ``dist_config`` argument. The ``correction_config`` then specifies the correction arguments for each distribution parameter. In our example, correction of the success probability is a *Sigmoid correction*, meaning that the sampled and calculated raw parameter value is transformed by a sigmoid function.

.. literalinclude:: examples/cdag_examples/node_correction.py
    :lines: 6-26

Graphs
======
To create a graph for the example above, we can serialize and order the 3 nodes and 2 connection edges as follows:

.. literalinclude:: examples/cdag_examples/graph_manual.py
    :lines: 6-69

This simulation can also be done using the ``Graph`` object.