Nodes and Edges
===============

This section explains the details of nodes and edges, their different types, and other
information that can be useful when simulating data using PARCS.

Node Types
----------

.. _stochastic_nodes:

Stochastic Nodes
~~~~~~~~~~~~~~~~

In the previous section we saw how a node is described by its sampling distribution. This type of
node is named *stochastic* node in PARCS. For a distribution with many parameters, the general
description of these nodes follows this pattern:

.. code-block::

    N: outputDistribution(param1_=..., param2_=..., ...)

As explained in the :ref:`theoretical background <theoretical_background>`, parameters accept
bias, linear, interactions, and quadratic terms of the parents. For example, if a node has 3
parents A, B and C, possible terms are:

.. code-block::

    A, B, C, A^2, AB, AC, B^2, BC, C^2

and hence, a possible combination looks something like ``A^2 + 2A + BC -1``.

Stochastic nodes can be marked with a feature called *Correction*. To explain this feature, look
at the graph below:

.. literalinclude:: code_blocks/nodes_and_edges/correction_1.py
    :linenos:
    :caption: a simple PARCS graph
    :lines: 1-13

This is a normal simulation graph. Now what happens if we describe B as a child of A?

.. literalinclude:: code_blocks/nodes_and_edges/correction_2.py
    :linenos:
    :caption: a graph that raises an error!

Of course the range of ``2A`` is bigger than ``[0, 1]``; therefore, the bernoulli distribution
receives invalid success probabilities for some of the samples. For distribution parameters with
a valid range smaller than real range, PARCS provide a simple sigmoid transformation, such that
it maps the real range to a custom range. For the bernoulli parameter, a standard sigmoid
transformation does the job, as it maps the values to ``[0, 1]``. For other parameters such as
the normal's standard deviation, or the exponential's rate, we can obtain wider ranges by customizing the
sigmoid function. This function is activated by putting a ``correction[]`` after the distribution:

.. literalinclude:: code_blocks/nodes_and_edges/correction_3.py
    :linenos:
    :caption: Using node correction
    :emphasize-lines: 2
    :lines: 3-14

the correction function has three parameters: ``lower, upper, target_mean``. As a default, lower
and upper parameters are 0 and 1. If we want to impose a certain mean on the corrected
parameter, we can set the ``target_mean`` parameter also. This parameter is useful, for example,
when we want to simulate a bernoulli node to have certain expected values.

.. literalinclude:: code_blocks/nodes_and_edges/correction_4.py
    :linenos:
    :caption: Using node correction
    :lines: 3-10

Note that by using correction, the given argument to the corrected parameter is wrapped with a
sigmoid function, and the target mean achieved by updating the bias value. So it is better to
read carefully the formula of correction in order to include it in your simulations. Overall,
this feature can be useful as researchers often use this trick to sample from the distributions
with restricted parameters.

.. _deterministic_nodes:

Deterministic Nodes
~~~~~~~~~~~~~~~~~~~

How to describe a simulation variable as :math:`Y = a_0 + a_1X_1 + a_2\sin{(X_2)} - a_3\log
(X_3^3)`? Clearly, we cannot use the stochastic nodes. Apart from the fact that only bias,
linear, interaction and quadratic terms are allowed, the described variable is not a random
variable, but a deterministic one.  You can use ``deterministic(_, _)`` nodes to implement such
nodes in the graph. First argument points to a python script which consists of custom python
functions, and second argument specifies the name of the function:

.. literalinclude:: code_blocks/nodes_and_edges/deterministic.py
    :linenos:
    :caption: Node C is deterministic

and the ``custom_functions.py`` file is:

.. literalinclude:: code_blocks/nodes_and_edges/custom_functions.py
    :linenos:
    :caption: ``custom_functions.py``

Some remarks:

1. The parents of the deterministic edge are only specified by the edges in the outline, and the columns that we read from the ``data`` object in the custom function.
2. Because of (1) the ``infer_edges`` option then does not work for deterministic nodes
3. The ``data`` object which the function receives, is a pandas DataFrame. Note that the edge functions are already applied on the data, before being passed to the node.
4. Even though not needed, the error vector is sampled for the deterministic node too. Apart from the consistency, this will help us later if we want to compare two graphs over a unique datapoint when a deterministic node is replaced by a stochastic one.

.. warning::
    PARCS identifies the custom function by appending the given path to the file, to the python
    directory. Therefore, be careful not to have a name conflict with other functions in your python
    path. A good cautionary measure is to always use a unique prefix for the functions such as
    ``parcs_custom_...``

What if we wanted to create a stochastic node (e.g. normally distributed variable) but with
custom equation for its parameters? Currently, this is not immediately possible; but with as a
quick workaround, you can first create a dummy deterministic node and pass it to the stochastic node:

.. literalinclude:: code_blocks/nodes_and_edges/dummy.py
    :linenos:
    :caption: A dummy node to simulate more complex stochastic nodes

Later we introduce a small feature to better represent and filter out the dummy nodes from the
last samples data frame.

Data Nodes
~~~~~~~~~~

In many simulation studies, it is desirable to augment a real dataset with some simulated nodes;
an example is simulating a missing data scenario where a *missingness mask* is simulated for a
given dataset. Data nodes allow us to read an external CSV file to sample a node. The argument of
data nodes are the CSV file path and name of the variable in the CSV file.


.. literalinclude:: code_blocks/nodes_and_edges/dataset.csv
    :caption: ``dataset.csv``
    :linenos:

.. literalinclude:: code_blocks/nodes_and_edges/data.py
    :linenos:
    :caption: main file

Some remarks:

1. The error vectors for A and B are identical. This is because we are sampling from the joint :math:`(A, B)` distribution. In other words, the *rows* of the CSV file are being sampled, rather than the single elements.
2. The length of the dataset is 3, but in the example, we sampled 5 data points. Because of that, we have duplicate samples for :math:`(A, B)`. The interpretation of this behavior is that we are sampling from a *distribution* which is determined by the dataset. Just like a parametric distribution such as Bernoulli, realizations may repeat.
3. We can sample the entire dataset once, using the ``full_data=True`` in the sample method. This way, the size of the sample is determined by the length of the dataset, and every data point is sampled once.

.. literalinclude:: code_blocks/nodes_and_edges/data_2.py
    :linenos:
    :caption: sampling using ``full_data``
    :lines: 14-22


.. note::
    Data nodes must have no incoming edges in the outline

Constant Nodes
~~~~~~~~~~~~~~

This class of nodes provides a simple way of adding a constant-value node to the graph. The syntax in the description file is as simple as creating a node by ``constant(<value>)``

.. literalinclude:: code_blocks/nodes_and_edges/const.py
    :linenos:
    :caption: sampling constant nodes
    :lines: 5-13

.. note::
    We can simulate deterministic and constant nodes by `hacking` stochastic ``normal`` node with ``mu_`` being a desired constant or determinist term, and ``sigma_`` being 0. Using the deterministic and constant nodes, however, adds more to the clarity of the simulation and the description file.

Edge Correction
---------------

Similar to the concept of distribution parameter correction for :ref:`stochastic nodes <stochastic_nodes>`, edges can also be marked for a correction. For edges, this is equal to normalization of the input values using the mean and standard deviation of the parent node distribution.

.. literalinclude:: code_blocks/nodes_and_edges/edge_correction.py
    :linenos:
    :caption: sampling constant nodes
    :lines: 5-24
    :emphasize-lines: 13

Edge correction for other edge function also has the same effect. The difference is, after normalization, the input values go through the transformation defined by the edge function.

Nodes and Edges Tags
--------------------

Last piece of information to complete the PARCS outlines syntax is the nodes and edges tags. There are a couple of occasions when we want to tag some nodes and edges to control the simulation behavior. This can be done by introducing a ``tags[]`` expression in the line.

.. code-block:: yaml

    A: poisson(lambda_=2), tags[T1, T2]
    B: normal(mu_=A^2, sigma_=A+1), correction[], tags[T3]
    A->B: sigmoid(alpha=1, beta=0, gamma=0, tau=1), tags[T4, T5]

First use of the tags that we introduce is to suppress returning the dummy nodes in the samples using the ``D`` tag:

.. literalinclude:: code_blocks/nodes_and_edges/dummy_tag.py
    :linenos:
    :caption: sampling constant nodes
    :lines: 3-19

As shown in the code, the error vector for the dummy node is still included in the returned errors, but the samples do only have ``A`` and ``B``.