.. _create_the_first_graph:

Creating the First Graph
========================

The following PARCS code example creates a simulation graph and samples from it:

.. literalinclude:: code_blocks/first_graph/main.py
    :caption: A PARCS code example
    :linenos:

Let's explore the code part by part.

Graph Outline
-------------

As PARCS simulates data based on a causal DAG, the first step in simulation is to describe a graph. For this, PARCS provides an easy and readable way.

.. literalinclude:: code_blocks/first_graph/main.py
    :caption: Outline
    :lines: 5-9

Line 5 of the code defines an ``outline`` dictionary which is the *structural equations* of the graph. The first three entries describe three nodes: :math:`A, B, C`. Node A follows a standard normal distribution and B follows a Bernoulli distribution with the success probability of :math:`p(B=1) = 0.4`. The outline describes node C using another normal distribution with unit standard deviation, whose mean is a function of A and B. In other words, we have :math:`C = B^2+2A-1 + \epsilon`, where :math:`\epsilon \sim \mathcal{N}(0, 1)`.

So, what are the rules and conventions of writing a graph outline?

1. Each key in the outline dictionary describes a node or an edge. The signature for the edge names is the arrow symbol ``->``. Indeed, if an edge ``X_i -> X_j`` is described, then nodes ``X_i`` and ``X_j`` must be described in the outline too. Node names follow a specific set of rules too. For instance, they must start with a letter, and the only allowed special character is the underscore.
2. Nodes are described by a sampling distribution. In this case, the nodes are known as *stochastic* in PARCS. There are other types of nodes available too, such as deterministic and data nodes. We will explore them in the later sections.
3. The equation describing the distribution parameters of a child node can consist of a bias term, and linear, interaction, and quadratic terms of the parent variables. Thus, valid terms to include for a node with 3 parents A, B and C are: ``A, B, C, A^2, AB, AC, B^2, BC, C^2, (bias)``.
4. Edges are described by a so-called *edge function*. These functions are transformations applied to the parent samples before being passed to the child node. In our example, the functions are identity, thus no transformation is applied. If a non-identity transformation :math:`E(X)` is applied, then the input to the child parameters are :math:`E(X)` rather than :math:`X`, even though in the outline, we still write ``X`` in the child node entry.

An outline can also be written in a YAML file. For instance, equivalent to the dictionary above, we can have the following file:

.. literalinclude:: code_blocks/first_graph/outline.yml
    :caption: :code:`outline.yml`

If the outline is written in a YAML file, the ``Description`` object in line 11 receives the path to
the file as ``Description('./path/to/outline.yml')``. Writing the outline separately in a YAML helps
us with better code separation and cleaner workspace.

.. seealso::
    * Read more about :ref:`rules and conventions for writing description outlines <conventions_description_outline>`.
    * Find the list of available distributions and edge functions :ref:`here <function_list>`.

.. _descriptions:

Descriptions
------------

In the next step, the outline is passed to the ``Description`` class.

.. literalinclude:: code_blocks/first_graph/main.py
    :caption: Description
    :lines: 11

As we will see in later sections, this intermediate step between the outline and the main PARCS graph, helps us do useful preprocessing. For now, let's study how this object works.

First, the ``infer_edges=False`` argument in line 11: You must have noticed that the outline holds sort of a redundancy with respect to describing the edges. As the node names A and B appear in the equation for C, PARCS could have infer the edges ``A->C`` and ``B->C``. But if you don't include the edges in the outline, you will get an error:

.. literalinclude:: code_blocks/first_graph/infer_edge_error.py
    :caption: PARCS error due to incomplete outline

So what is the rationale for this behavior? Note that edges are described by their edge function, so PARCS needs to know what function you have chosen for an edge in order to sample from the graph. However, since the ``identity()`` function is a basic function and will probably selected most of the times, you can set the ``infer_edges=True`` argument and skip the edges in the outline, and ``Description`` assumes an identity edge for all possibilities:

.. literalinclude:: code_blocks/first_graph/infer_edge_true.py
    :caption: implicit edges are inferred

If you want to set other edge functions, you must describe them explicitly in the outline.

Now, let's briefly check the main attributes of the description object. Firstly, the description object has two ``nodes`` and ``edges`` attributes which hold the lists of parsed nodes and edges for the graph:

.. literalinclude:: code_blocks/first_graph/desc.py
    :caption: instantiating a ``Description`` object
    :linenos:
    :lines: 7-42

Each entry in these two attributes describe a node or an edge. As a start, you check the ``output_distribution`` and ``function_name`` keys. Next, you can see how the equations in the nodes are reflected in the ``dist_params_coefs`` key. These coefficient vectors follows the orders of the parent list which is another attribute of the description object.

.. literalinclude:: code_blocks/first_graph/desc.py
    :caption: ``Description`` nodes and edges
    :linenos:
    :lines: 43-44

Good news is, as a general user of PARCS, you almost never need to dig into these attributes. The interplay between the objects handle the graph definition and data generation for you. This was only a sneak peek into behind the scenes.

.. seealso::
    * See :class:`pyparcs.Description` for the class API doc.
    * See the sequence diagram for :ref:`Description instantiation <uml_description_instantiation>`


Instantiate a Graph
-------------------

Finally, the description object is passed to the ``Graph`` class to instantiate a graph object, from which you can sample the generated data. More specifically, the ``.sample()`` method provides the samples.

.. literalinclude:: code_blocks/first_graph/main.py
    :caption: Description
    :lines: 11-13

You probably have noticed that the sampling method returns two values, one of which we ignored in the code example by the underscore symbol. PARCS samples the graph by sampling an *error vector* from a continuous uniform :math:`\text{unif}(0, 1)` for each node and pass it to the inverse CDF of the distribution. In this approach, all the stochasticity of a node is reflected in the error values, as the result of iCDF given the error value is fixed. This means that reproducibility is guaranteed by controlling the error vectors.

Going back to the sampling method, the second output is the sampled error vectors.

.. literalinclude:: code_blocks/first_graph/errors_1.py
    :caption: returning the error vectors
    :linenos:

As a quick sanity check, you can compare the error vector for B: ``[0.95, 0.15]`` with the samples for B: ``[1.0, 0.0]``. As we know, the errors give :math:`1` if bigger than :math:`p`, and :math:`0` otherwise.

The sampling method allows us to reproduce the results by re-using the sampled errors:

.. literalinclude:: code_blocks/first_graph/errors_2.py
    :caption: reusing sampled errors
    :linenos:

Note that when using the sampled errors, we do not set the ``size`` argument, as the graph object samples according to the length of the error vectors. This was a trivial use of sampled errors. The scenario becomes interesting if you want to compare the generated data of a graph before and after a change. In the example below, we cut an edge and compare one data point specifically:

.. literalinclude:: code_blocks/first_graph/errors_3.py
    :caption: reusing sampled errors after changing the graph
    :linenos:

.. seealso::
    Read the API doc for the :class:`pyparcs.Graph` class.
