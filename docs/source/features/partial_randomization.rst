.. _partial_randomization:

=====================
Partial Randomization
=====================

In simulations, sometimes we need to explore the graphs with certain restrictions; we need to fix some (not all) parameters that models our restrictions, while randomizing the rest of the parameters. As an example, for testing an ML or causal inference model with Gaussian assumption on the nodes, we want to simulate a graph with all Gaussian output distributions, while allowing the mean values to be selected randomly (because perhaps we want to test our model on MVG distributions with arbitrary means and variances).

An important feature of PARCS, the *partial randomization*, allows us to simulate for such scenarios. In this section we explain how this can be done.

.. warning::
    to add: For more details regarding the motivation and use-cases please see "here".


Parameter Randomization
=======================

As before, simulation starts with a graph description file. However we can use two new values to activate randomization PARCS: :code:`?` for parameters, and :code:`random` for functions and distributions. Here is an example of the causal triangle where we randomize the mean of node A, and also the output distribution of Y:

.. literalinclude:: code_blocks/b2/graph_description_1.yml
    :linenos:
    :caption: graph_description.yml
    :emphasize-lines: 3, 4

.. literalinclude:: code_blocks/b2/graph_1.py
    :linenos:
    :caption: graph.py
    :emphasize-lines: 5-9

In :code:`graph.py` file, instead of :code:`graph_file_parser`, we import :code:`parcs.graph_builder.randomizer.ParamRandomizer`. ``ParamRandomizer`` takes the graph description directory, as well as a *guideline file*. This file tells the radomizer, what are the options for parameters and functions. Let's have a look at the guideline file:

.. literalinclude:: code_blocks/b2/simple_guideline.yml
    :linenos:
    :caption: simple_guideline.yml

1. There are two main keys: :code:`nodes` and :code:`edges`
2. the next-level keys determine the different possibilities for choosing the functions and distributions. For instance, in the example file we tell the randomizer to choose between :code:`bernoulli` and :code:`gaussian` for distributions, and between :code:`identity` and :code:`gaussian_rbf` for the edge functions.
3. Each function/distribution option contains the corresponding parameters as keys. The value of these keys is a list of the length 3 which corresponds to *bias, linear,* and *interactions* coefficients (in the same order). Each component is a directive for the respective coefficient, i.e. first component is used to randomize the bias value, second one for linear coefficient, and third for the interaction coefficient. The directives follow the following convention:

   a. a single value, means *fixed*, thus for all the randomization turns, randomizer picks the same value
   b. :code:`[f-range, X, Y]` tells the randomizer to pick a float value between X and Y uniformly, i.e. from `Unif(X, Y)`.
   c. :code:`[i-range, X, Y]` is similar to (a), except that the value is integer (with equal probabilities).
   d. :code:`[choice, X_0, X_1, ...]` tells the randomizer to pick a value from the given list of options in the directive with equal probabilities (element 2nd onward).

In our example, therefore, mean of node A is selected according to line 5 of the guideline, and the distribution of Y is chosen between Bernoulli and Gaussian distributions.

.. note::
    For ``f-range`` and ``i-range`` directives, we can specify more than one range. To do so, we simply write the next low/high pairs in the list. **Example:** The guideline :code:`[f-range, -1, -0.5, 0.5, 1]` samples randomly from the range of :math:`(-1, -0.5) \cup (0.5, 1)`.

.. seealso::
    :ref:`Randomization conventions <conventions_inducing_randomization>`

Free Randomizer
===============

Using free randomizer, we can randomly generate a graph based on guidelines for number of nodes and graph sparsity. In fact, this randomizer is wrapped around the parameter randomization, meaning that first, PARCS initializes a graph with some nodes and edges, and assigns `free` and `?` to all functions and parameters, then passes the process to parameter randomizer to fill the blanks.

The guideline is similar to that of parameter randomizer, except that we need to give the directives for ``graph`` as well. The keys are `num_nodes` and `graph_sparsity`.

.. literalinclude:: code_blocks/b6/simple_guideline.yml
    :linenos:
    :emphasize-lines: 1-3
    :caption: simple_guideline.yml

using the `FreeRandomizer` class, we generate a graph object.

.. literalinclude:: code_blocks/b6/graph.py
    :linenos:
    :caption: graph.py

Nodes are named `H_<num>` by default. You can change the name prefix by the key ``node_name_prefix`` in the guideline (under graph key). The value must be string.

.. _connect_randomizer:

Connect Randomizer
==================

This class randomizes the causal flow from a parent to a child graph. To clarify this functionality, suppose we want to simulate the following data model.

.. math::

    \begin{aligned}
        L &= AL \\
        Z &= BZ + CL
    \end{aligned}

where :math:`Z` and :math:`L` are random vectors, :math:`A` and :math:`B` are a lower triangular coefficient matrices, and :math:`C` is another arbitrary coefficient matrix. In this model, :math:`L` and :math:`Z` are normal simulated graphs, while :math:`Z` receives causal edges from :math:`L` via :math:`C` matrix.

This model can be simulated as the following:

.. literalinclude:: code_blocks/b8/graph_description_L.yml
    :linenos:
    :caption: `graph description for L`

This description is a normal PARCS graph description file for the parent graph.

.. literalinclude:: code_blocks/b8/graph_description_Z.yml
    :linenos:
    :emphasize-lines: 4, 5
    :caption: `graph description for Z`

The second graph description file has a minor special style. In line 4 and 5 you can see a symbol `*` behind `mu_` distribution parameters. This syntax tells PARCS to affect these parameters when external causal edges (from L) connect to the graph. Similarly, the fact that the node `Z_1` doesn't have an asterisk means that PARCS will not let any edge to go to `Z_1`.

Similar to other randomization classes, `ConnectRandomizer` also requires a (normal) guideline file.

Finally, the graph can be simulated as follows:

.. literalinclude:: code_blocks/b8/graph.py
    :linenos:
    :caption: `graph.py`

argument `adj_matrix_mask` is a further restriction by the user to limit the new edges. For example we can suppress any outgoing edge from `L_1` if in this mask, first row is all zero (rows and columns follow the same order of the nodes in the graph description file of the parent and child graphs respectively).

ConnectionRandomizer creates a temp graph description file as an intermediate step, which deletes it after the process if ``delete_temp_graph_description=False``. To check how the class works, we can have a look at this temp file.

.. literalinclude:: code_blocks/b8/combined_gdf.yml
    :linenos:
    :emphasize-lines: 10-12
    :caption: temp file: `combined_gdf.yml`

The highlighted lines show the traces of the new connection between graphs. Note that this description file is next passed to the `ParamRandomizer`, the parent class of `ConnectionRandomizer`, and that is why there are question marks in the description file.

.. note::
    The density of new edges follows the `graph_density` entry in the guideline file. However, based on asterisks and adjacency matrix mask, number of edges might be less than intended. E.g., `graph_density: 1` tells PARCS to fully connect the parent graph to child graph. However, if the child nodes doesn't have asterisks in parameters, PARCS will skip all nodes (as we have requested no parameter to be affected by new edges), hence no new edges will be added.

Edge correction
===============

When randomizing the graph parameters, we potentially face two undesired issues:

1. For edge functions such as Sigmoid and Gaussian RBF, if the parent output is too far away from the active region of the function, we will end up with low variability in the transformed values. For example, If for a Sigmoid edge function, the output distribution of the parent node is :math:`\mathcal{N}(10, 2)`, then the transformed values of a Sigmoid edge function for 1000 samples are in the range of :math:`(0.9912\dots, 0.9999\dots)`.
2. Take an example of a node with 3 parents, 2 of which are subjected to Sigmoid edge function and have a :math:`[0, 1]` support while calculating the child node's distribution parameters, while the remaining is subjected to the identity function and retain its original support. In this case, we have an imbalance when calculating the distribution coefficients, as we pick the distribution coefficients randomly from the same value range for all parents.

In general, both items indicate an issue similar to the issue of *sensitivity of calculations to unit of measurement*, and of course the remedy to that can be the same, which is normalization. In issue 1, normalization brings the values to the active region of the edge function, and in issue 2, it brings all the parents to the same scale.

Such normalization can be done via activating the `correction` option for the edges. PARCS automatically activates edge correction when randomizing the parameters. But in general, we can do edge correction in any simulation, by modifying the graph description file.

.. literalinclude:: code_blocks/b3/graph_description_edge.yml
    :linenos:
    :caption: graph_description.yml

.. literalinclude:: code_blocks/b3/edge_correction.py
    :linenos:
    :caption: graph.py
    :lines: 1-4, 6-10

In this example, the variable Y has the mean of `1.0` because of the mean of A and C. Now we activate correction for `A->Y` and `C->Y` edges:

.. literalinclude:: code_blocks/b3/graph_description_edge_2.yml
    :linenos:
    :caption: graph_description.yml
    :emphasize-lines: 8-9

.. literalinclude:: code_blocks/b3/edge_correction.py
    :linenos:
    :caption: graph.py
    :lines: 5-9, 11

.. note::
    Similar to :ref:`Node correction <node_correction_node>` **edge correction parameters (i.e. mean and standard deviation) is always initialized upon the first batch of data**. For this purpose, the graph object always `burns` the first 500 samples, to initialize the corrections. Read more about edge correction at :func:`~pyparcs.cdag.utils.EdgeCorrection`.