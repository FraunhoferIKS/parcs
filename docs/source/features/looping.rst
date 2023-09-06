Looping over Randomized Parameters
==================================

In the :ref:`Partial randomization <partial_randomization>` section, we explained how to generate graphs from a graph parameter space. What might come after, is running an algorithm on many generated graphs and measuring the performance of the algorithm with respect to a graph parameter, e.g. the number of nodes. ``GuidelineIterator`` allows us to change a graph parameter in a loop, and do such analyses.

To use the iterator, the guideline outline doesn't need to change. The only point is that the randomization directive of the target parameter, becomes the looping directive. Let's say we want to iterate over number of nodes and run an algorithm:

.. literalinclude:: code_blocks/guideline_iterator/graph.py
    :linenos:
    :caption: graph.py
    :emphasize-lines: 5, 12

By passing the outline to the ``GuidelineIterator`` rather than the ``Guideline`` class, and calling the ``get_guidelines``, you get a generator that yields a guideline according to the outline, with only the exception of the directive which is specified by the ``route`` argument: That argument is fixed (based on the ``step`` argument for each iteration.

.. note::
    To iterate over the distribution parameters coefficients, you can use the keywords ``bias, linear, interaction``. Example: ``route = 'nodes.normal.mu_.linear'``