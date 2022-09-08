===========
Get Started
===========

We start by a brief introduction of the theoretical background of PARCS library. Next, we jump right into the code and introduce all the functionalities. Links to more details are provided along the document.

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

1. Each node is identified by a specific univariate distribution :math:`\mathcal{P}` which is parameterized by :math:`\Theta = (\theta_1, \dots, \theta_K)`. For instance, :math:`\mathcal{P}` can be the Gaussian normal distribution :math:`\mathcal{N}(\Theta)`, parameterized by :math:`\Theta = (\theta_1=\mu, \theta_2=\sigma)`. Consequently, to sample a node means to sample from its distribution. (see here for the list of available distributions)
2. The distribution parameters are functions of the parents of the node, i.e. :math:`\Theta = \Theta\big(\text{PA(X_i)}\big)`. Continuing the above example, if a node :math:`X_3` with an associating Gaussian normal distribution has two parents :math:`X_1` and :math:`X_2`, then :math:`\mu = \mu(X_1, X_2)` and :math:`\sigma = \sigma(X_1, X_2)`.
3. In PARCS, the functions of the distribution parameters are limited to bias, linear and interaction terms. Interactions are defined as multiplication of any subset of the parents. In our example, the mean of the Gaussian normal, can be :math:`X_1 -2.5X_2 + 0.7X_1X_2 + 1`, but not :math:`X_1^2 - 2X_1X_2^3`. Therefore, assuming a *variables dictionary* which includes bias, linear, and all interaction terms, each parameter equation can be defined as the dot product of a *coefficient vector* and the variables dictionary. Subsequently, each distribution parameter has a unique coefficient vector.
4. The causal flow of edges can be subjected to a deterministic transformation, meaning that instead of the parent variables, functions of the parent variables are inputs of distribution parameters. In our example, it is possible that :math:`\mu = X_1 + g(X_2)` where :math:`g(.)` is an arbitrary transformation (which we call *edge function*). These functions are placed on edges, are applied to individual parents, and thus are parent-child specific. (see here for a list of available transformations)

Figure ? depicts a schematic of how the 4 ideas form the internal structure of PARCS simulation. (more details)

Starting with causal DAGs
=========================
Based on the description above, we can define a causal DAG in PARCS by specifying the output distributions and parameter coefficient vectors of the nodes, along with edges and their edge functions. This can be done using the `parcs.cdag.graph_objects.Graph` class, and a `.yml` description file.

.. literalinclude:: examples/graph_example_1/graph_description.yml
    :caption: :code:`graph_description.yml`
    :linenos:

As self-explainatory as the description file is, please refer to "here" to read a detailed review of the description file conventions.

.. literalinclude:: examples/graph_example_1/graph.py
    :caption: :code:`graph.py`
    :linenos:
    :lines: 1-13

The :code:`graph_file_parser` function, outputs *node* and *edge* objects to instantiate a :code:`Graph` class. These objects are lists of Python dicts, including all information given by the description file. See "here" for a detailed explanation. The output of the graph object is a :code:`Pandas.DataFrame` object.

Sampling error terms
--------------------

The math library in PARCS that supports sampling from the distributions is Scipy. Instead of using the :code:`.rvs()` methods of the distributions, however, PARCS samples *error* terms from a :math:`\text{Unif}(0, 1)` per each data row, and passes the realization to the :code:`.ppf()` method to obtain the corresponding sample from the target distribution. (see here for more details about math stuff). With this procedure, we follow two main goals:

* Reproducibility is explicitly handled by returning and re-using the error terms.
* These error terms enable us to run simulate for counterfactual analysis. We explain this goal in the section below.

Considering the error terms, the :code:`.sample()` method receives the following options:

.. literalinclude:: examples/graph_example_1/graph.py
    :linenos:
    :lines: 15-35

Observations, Interventions, Counterfactuals
============================================

In addition to the :code:`.sample()` method, which allows us to sample from the observational distribution, the graph object provides methods to sample from interventional distributions as well. There are three types of interventions available:

Fixed-value intervention
------------------------

This method sets the value of one or more nodes to a fixed value. As this is the most basic type of intervention in causal inference theory, we name the method simply `.do()`. Below is an example of a causal triangle graph, with two instances of fixed-value intervention.

.. literalinclude:: examples/graph_example_2/fixed_value_do.py
    :linenos:
    :emphasize-lines: 17, 25-26, 32

