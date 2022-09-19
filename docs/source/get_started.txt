.. _get_started:

===============
Getting started
===============

We start by a brief introduction of the theoretical background of PARCS library. Next, we jump right into the code and introduce all the functionalities. Links to more details are provided throughout the document.

Theoretical background
======================

The underlying causal mechanism of a data distribution can be specified using causal Directed Acyclic Graphs (DAGs). A causal DAG includes nodes which represent variables, and directed edges which represent a causal flow from the parent to child nodes. A data distribution :math:`(X_1, \dots, X_D)` is factorized with respect to a causal graph :math:`\mathcal{G}` as

.. math::
    \begin{align}
    p(X_1, \dots, X_D) = \prod_{d=1}^{D} p(X_d|\text{PA}_{\mathcal{G}}(X_d)).
    \end{align}
    :label: dag_factorization

Eq. :eq:`dag_factorization` has two implications:

* Each node is a function of its parents in the associating causal DAG.
* In order to sample from the full data distribution, we can start with sampling the source nodes (nodes without parents), then sampling their child nodes, and continuing the process according to the topological ordering of the graph, until all nodes are sampled.

Simulation in PARCS follows this formulation, and in addition to that, 4 more design ideas:

1. **Each node is identified by a specific univariate distribution** :math:`\mathcal{P}` which is parameterized by :math:`\Theta = (\theta_1, \dots, \theta_K)`. For instance, :math:`\mathcal{P}` can be the Gaussian normal distribution :math:`\mathcal{N}(\Theta)`, parameterized by :math:`\Theta = (\theta_1=\mu, \theta_2=\sigma)`. Consequently, **to sample a node means to sample from its distribution.**
2. **Distribution parameters are functions of the parents of the node**, i.e. :math:`\Theta = \Theta\big(\text{PA(X_i)}\big)`. Continuing the above example, if a node :math:`X_3` with an associating Gaussian normal distribution has two parents :math:`X_1` and :math:`X_2`, then :math:`\mu = \mu(X_1, X_2)` and :math:`\sigma = \sigma(X_1, X_2)`.
3. In PARCS, the **functions of distribution parameters are limited to bias, linear and interaction terms**. Interactions are defined as multiplication of any subset of the parents. In our example, the mean of the Gaussian normal, can be :math:`X_1 -2.5X_2 + 0.7X_1X_2 + 1`, but not :math:`X_1^2 - 2X_1X_2^3`. Therefore, assuming a *variables dictionary* which includes bias, linear, and all interaction terms, **each parameter equation can be defined as the dot product of a *coefficient vector* and the variables dictionary**. Subsequently, each distribution parameter has a unique coefficient vector.
4. The causal flow of edges can be subjected to a deterministic transformation, meaning that **instead of the parent variables, functions of the parent variables are inputs of distribution parameters**. In our example, it is possible that :math:`\mu = X_1 + g(X_2)` where :math:`g(.)` is an arbitrary transformation (which we call *edge function*). **These functions are placed on edges, are applied to individual parents, and thus are parent-child specific**.

Figure ? depicts a schematic of how the 4 ideas form the internal structure of PARCS simulation. (more details)

.. _get_started_graph:

Starting with causal DAGs
=========================
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


Observations, Interventions, Counterfactuals
============================================

In addition to the :code:`.sample()` method, which allows us to sample from the observational distribution, the graph object provides methods to sample from interventional distributions as well. There are three types of interventions available:

Fixed-value intervention
------------------------

This method sets the value of one or more nodes to a fixed value. As this is the most basic type of intervention in causal inference theory, we name the method simply :code:`.do()`. Below is an example of a causal triangle graph, with fixed-value interventions applied.

.. literalinclude:: examples/graph_example_2/fixed_value_do.py
    :linenos:
    :emphasize-lines: 17, 25-26, 32

functional intervention
-----------------------

In this type of intervention which you do via :code:`.do_functional()`, you can set a node's value to a deterministic function of other nodes. The function is a Python function (defined by :code:`def` keyword or as a :code:`lambda` function). Using this method, you can induce the *soft (parametric) intervention* scenario, where the inputs of the intervention function are the parents of the node in the original DAG. Nevertheless, the inputs of the function can be any subset of the nodes, except for the descendants of the intervened node.

.. literalinclude:: examples/graph_example_2/functional_do.py
    :linenos:
    :lines: 17-26
    :emphasize-lines: 3-4

.. _get_started_self_intervention:

self intervention
-----------------

Imagine the following causal question: *what if for every patient, we administer 1 unit of drug less than what we normally administer*. The "do" term for this intervention would be :math:`\text{do}(A=f(A_\text{old}))`. To simulate data for this type of intervention, we use the method :code:`.do_self()`

.. literalinclude:: examples/graph_example_2/self_do.py
    :linenos:
    :lines: 17-35
    :emphasize-lines: 12

.. note:: In the example above, line 13 tells the method to reuse the samples. This decision can be made in previous types of intervention as well. Using this feature, we can simulate **counterfactual**  scenarios: *what would have happened had we changed our actions?* In this case, the intervention must be applied to the same sampled dataset, and this can be done by reusing the sampled errors.

Partial Randomization
=====================

In simulations, sometimes we need to explore the graphs with certain restrictions; we need to fix some (not all) parameters that models our restrictions, while randomizing the rest of the parameters. As an example, for testing an ML or causal inference model with Gaussian assumption on the nodes, we want to simulate a graph with all Gaussian output distributions, while allowing the mean values to be selected randomly (because perhaps we want to test our model on MVG distributions with arbitrary means and variances).

An important feature of PARCS, the *partial randomization*, allows us to simulate for such scenarios. In this section we explain how this can be done. For more details regarding the motivation and use-cases please see "here".

Parameter Randomization
-----------------------

As before, simulation starts with a graph description file. However we can use two new values to activate randomization PARCS: :code:`?` for parameters, and :code:`random` for functions and distributions. Here is an example of the causal triangle where we randomize the mean of node A, and also the output distribution of Y:

.. literalinclude:: examples/randomization_examples/graph_description_1.yml
    :linenos:
    :caption: graph_description.yml
    :emphasize-lines: 3, 4

.. literalinclude:: examples/randomization_examples/graph_1.py
    :linenos:
    :caption: graph.py
    :emphasize-lines: 5-9

In :code:`graph.py` file, instead of :code:`graph_file_parser`, we import :code:`parcs.graph_builder.randomizer.ParamRandomizer`. :code:`ParamRandomizer` takes the graph description directory, as well as a *guideline file*. This file tells the radomizer, what are the options for parameters and functions. Let's have a look at the guideline file:

.. literalinclude:: examples/randomization_examples/simple_guideline.yml
    :linenos:
    :caption: simple_guideline.yml

1. There are two main keys: :code:`nodes` and :code:`edges`
2. the next-level keys determine the different possibilities for choosing the functions and distributions. For instance, in the example file we tell the randomizer to choose between :code:`bernoulli` and :code:`gaussian` for distributions, and between :code:`identity` and :code:`gaussian_rbf` for the edge functions.
3. Each function/distribution option contains the corresponding parameters as keys. The value of these keys is a list of the length 3 which corresponds to *bias, linear,* and *interactions* coefficients (in the same order). Each component is a directive for the respective coefficient, i.e. first component is used to randomize the bias value, second one for linear coefficient, and third for the interaction coefficient. The directives follow the following convention:

   a. a single value, means *fixed*, thus for all the randomization turns, randomizer picks the same value
   b. :code:`[f-range, X, Y]` tells the randomizer to pick a float value between X and Y (uniformly).
   c. :code:`[i-range, X, Y]` is similar to (a), except that the value is integer.
   d. :code:`[choice, X_0, X_1, ...]` tells the randomizer to pick a value from the given list of options in the directive (element 2nd onward).

In our example, therefore, mean of node A is selected according to line 5 of the guideline, and the distribution of Y is chosen between Bernoulli and Gaussian distributions.

Edge correction
---------------

When randomizing the graph parameters, we potentially face two undesired issues:

1. For edge functions such as Sigmoid and Gaussian RBF, if the parent output is too far away from the active region of the function, we will end up with low variability in the transformed values. For example, If for a Sigmoid edge function, the output distribution of the parent node is :math:`\mathcal{N}(10, 2)`, then the transformed values of a Sigmoid edge function for 1000 samples are in the range of :math:`(0.9912\dots, 0.9999\dots)`.
2. Take an example of a node with 3 parents, 2 of which are subjected to Sigmoid edge function and have a :math:`[0, 1]` support while calculating the child node's distribution parameters, while the remaining is subjected to the identity function and retain its original support. In this case, we have an imbalance when calculating the distribution coefficients, as we pick the distribution coefficients randomly from the same value range for all parents.

In general, both items indicate an issue similar to the issue of *sensitivity of calculations to unit of measurement*, and of course the remedy to that can be the same, which is normalization. In issue 1, normalization brings the values to the active region of the edge function, and in issue 2, it brings all the parents to the same scale.

Such normalization can be done via activating the `correction` option for the edges (For more details blah blah). PARCS automatically activates edge correction when randomizing the parameters. But in general, we can do edge correction in any simulation, by modifying the graph description file.

.. literalinclude:: examples/correction_examples/graph_description_edge.yml
    :linenos:
    :caption: graph_description.yml

.. literalinclude:: examples/correction_examples/edge_correction.py
    :linenos:
    :caption: graph.py
    :lines: 1-4, 6-10

In this example, the variable Y has the mean of `1.0` because of the mean of A and C. Now we activate correction for `A->Y` and `C->Y` edges:

.. literalinclude:: examples/correction_examples/graph_description_edge_2.yml
    :linenos:
    :caption: graph_description.yml
    :emphasize-lines: 8-9

.. literalinclude:: examples/correction_examples/edge_correction.py
    :linenos:
    :caption: graph.py
    :lines: 5-9, 11