Rules and Conventions
=====================

In this document, we present the PARCS conventions for descriptions, guidelines, etc.

.. _conventions_description_outline:

Description Outlines
--------------------

Outline is a Python dictionary (or a YAML file) to define a graph:

General
~~~~~~~

* It is not required to sort the nodes and edges in the YAML file or when writing the dictionary. But this would help better readability of the code. It is also advised to use comment lines to explain the YAML file lines.

Node Names
~~~~~~~~~~

* The best practice is to use clear node names, which only includes one underscore, followed by a number. Example: ``Z_1``.
* If the node name is longer than 1 letter, it is advised to capitalize the first letter. This will help especially with the clarity of parameters' equations. Example: ``mu_= 2AgeHeight`` instead of ``mu_=2ageheight``
* Special characters (other than underscore) are *not allowed*. Also, the node name must start with a letter.
* It is advised to use node names that are not an exact subset of other names. Example: having both ``A_1`` and ``A_11`` nodes. This naming does not violate any parsing rules, and PARCS is allowed to handle it; but it is the best practice if you can avoid it, e.g. by padding the indices when the nodes are numbered: ``A_01, A_02, ..., A_11``.

Edge Names
~~~~~~~~~~

* Edge names always are made of node names and a connectiong ``->`` symbol.
* Obviously, node names and edges must be consistent. In other words, if there is an `A->B` edge, there must be nodes `A` and `B` defined (unless using ``infer_edges`` parameter).

Values
~~~~~~

* The values for nodes and edges are distribution and edge functions respectively. The order of parameters does not matter.
* Spaces are allowed in the values. When parsing, PARCS strips the string of the spaces.
* The order of line extensions such as ``correction[]`` and ``tags[]`` does not matter.


.. _conventions_inducing_randomization:

Inducing Randomization
----------------------

To tell PARCS to randomize part of the graph description, follow the conventions below:

* **Node distributions and edge functions**: The ``random`` keyword tells PARCS to sample the distributions or functions from the line. example: ``A: random``, ``A->B: random``
* **All parameters**: the ``?`` symbol in the parenthesis of the function or distribution randomizes all the parameters; example: ``A: normal(?)``, ``A->B: sigmoid(?)``
* **Single parameters**: ``?`` in front of a parameter randomizes it only; example: ``A: normal(mu_=?, sigma_=1)``, ``A->B:arctan(alpha=3, beta=?, gamma=0)``
* **Single coefficient (nodes)**: ``?`` at the beginning of a term randomizes the coefficient for that term; example: ``A: normal(mu_=2X+?Y, sigma_=1)``

Limiting the Connection Randomization
-------------------------------------

We can suppress a distribution parameter in the child subgraph from receiving any incoming new edge (from the parent subgraph) with a ``!`` symbol behind the parameter name. Suppose the description below defines a child subgraph:

.. code-block:: yaml

    A: normal(mu_=2X+3Y, !sigma_=2)

When randomizing the connection of a parent subgraph to this child subgraph, the ``sigma_`` parameter won't receive any new edges.

.. _conventions_guideline:

Guideline Outlines
------------------

A guideline outline may consist of three main keys: nodes, edges, and graph:

``graph`` key may include ``num_nodes`` and ``density`` keys. Num nodes must be defined with ``i-range`` directive, and determines the number of nodes, used by the :ref:`random description class <random_description_doc>`. The Density must be defined maximum in the 0-1 range. It defines the graph sparsity (1-density). It is used by random description class and the :ref:`connection randomization method <connection_randomization_doc>`.


``nodes`` key includes the distribution names, which in return have parameter randomization directives. The main pattern is:

.. code-block:: yaml

    nodes:
      distribution1:
        parameter1: [[...], [...], [...]]
        parameter2: [[...], [...], [...]]
        parameter3: ...
      distribution2:
        ...

The three ``[...]`` lists for parameters correspond to bias, linear, and interactions coefficients. If a distribution name exists in the guideline, it will be a candidate for ``random`` nodes in the description.

``edges`` key includes the edge functions, which in return have parameter directives. The main pattern is:

.. code-block:: yaml

    nodes:
      function1:
        parameter1: [...]
        parameter2: [...]
        parameter3: ...
      function2:
        ...

Unlike the nodes, edge function parameters need only one directive.

.. note::
    The ``identity`` edge function needs ``null`` (in Yaml) or ``None`` (in Python) as the value.