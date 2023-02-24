===========
Conventions
===========

In this document, we present the PARCS conventions for description files, guideline files, etc.

.. _conventions_graph_description_file:

Graph description file
======================

The description file is a YAML file. Below is a list of conventions for the description file:

* The rows start by ``some_name:``, following are the *names* and values are functions/distributions.The parser recognizes all the keys including `arrow` sign  ``->`` as edges, and the remaining keys as nodes.
* Node names and edges must be consistent. In other words, if there is an `A->B` edge, there must be nodes `A` and `B` defined. We recommend to start the node names with letters, and (multiple) `_` and numbers, for more stability.
* the value of each row (after ``:``) is an edge function for edges, and an output distribution for nodes. Parameters of the functions must correspond to the function itself. See the full list of available edge functions and output distributions.
* For activating `correction`, the ``correction[...]`` keyword should be brought after the main function. It's not necessary, however recommended for readability, that the function and correction are separated by a comma.

.. _conventions_inducing_randomization:

Inducing randomization
----------------------

To tell PARCS to randomize part of the graph description, follow the conventions below:

* The question mark ``?`` in front of a distribution/function parameter tells PARCS to randomize that parameter.
* Note that for distribution parameters, the entire parameter (not a coefficient) can be randomized. E.g. ``mu_=?`` is possible, while ``mu_=2A+?C`` is an invalid syntax
* The question mark ``?`` as the argument of a function/distribution tells PARCS to randomize all parameters. E.g. ``gaussian(?)`` means both ``mu_`` and ``sigma_`` will be randomized
* The keyword ``random`` in front of node/edge names means the function/distribution will be randomly chosen. E.g. ``X_0: random``. Obviously the parameters of the selected function will be randomized as well.

The randomization is done based on a `guideline` file. For more details, see below.