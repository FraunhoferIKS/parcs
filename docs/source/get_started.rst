===========
Get Started
===========

The underlying causal mechanism of a data distribution can be specified using causal Directed Acyclic Graphs (DAGs). A causal DAG includes nodes which represent variables, and directed edges which represent a causal flow from the parent to child nodes. A data distribution :math:`(X_1, \dots, X_D)` is factorized with respect to a causal graph :math:`\mathcal{G}` as

.. math::
    \begin{align}
    p(X_1, \dots, X_D) = \prod_{d=1}^{D} p(X_d|\text{PA}_{\mathcal{G}}(X_d)).
    \end{align}
    :label: dag_factorization

Eq. :eq:`dag_factorization` has two implications:

* Each node is a function of its parents in the associating causal DAG.
* In order to sample from the full data distribution, we can start with sampling the source nodes (nodes without parents), then sampling their child nodes, and continuing the process according to the topological ordering of the graph, until all nodes are sampled.

Simulation in PARCS follow the same concept, with a 4 more design ideas:

1. Each node is identified by a specific univariate distribution :math:`\mathcal{P}` which is parameterized by :math:`\Theta = (\theta_1, \dots, \theta_K)`. For instance, :math:`\mathcal{P}` can be the Gaussian normal distribution :math:`\mathcal{N}(\Theta)` parameterized by :math:`\Theta = (\theta_1=\mu, \theta_2=\sigma)`. Consequently, to sample a node means to sample from its distribution.
2. The distribution parameters are functions of the parents of the node, i.e. :math:`\Theta = \Theta\big(\text{PA(X_i)}\big)`. Continuing the above example, if a node :math:`X_3` with an associating Gaussian normal distribution has two parents :math:`X_1` and :math:`X_2`, then :math:`\mu = \mu(X_1, X_2)` and :math:`\sigma = \sigma(X_1, X_2)`
3. In PARCS, the functions of the distribution parameters are restricted to only include bias, linear and interaction terms. Interactions are defined as multiplication of any subset of the parents. In our example, the mean of the Gaussian normal, can be :math:`X_1 -2.5X_2 + 0.7X_1X_2 + 1`, but not :math:`X_1^2 - 2X_1X_2^3`
4. The causal flow of edges can be subjected to a deterministic transformation, meaning that instead of the parent variables, functions of the parent variables are inputs of distribution parameters. In our example, it is possible that :math:`\mu = X_1 + g(X_2)` where :math:`g(.)` is an arbitrary transformation. These functions are placed on edges, are applied to individual parents, and thus are parent-child specific.

Figure ? depicts a schematic of how these 4 ideas form the internal structure of PARCS simulation.

