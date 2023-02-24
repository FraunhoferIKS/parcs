==================================
Looping over randomized parameters
==================================

In the :ref:`Partial randomization <partial_randomization>` section, we explained how to generate graphs from a graph parameter space. What might come after, is that we run an algorithm on many generated graphs, and measure the performance of the algorithm vs. a graph parameter, e.g. the number of nodes. The function ``guideline_iterator`` allows us to change a graph parameter in a loop, and do such analyses.

To use the iterator, the guideline file doesn't need to change. The only point is that the randomization directive of the target parameter, becomes the looping directive:

.. literalinclude:: code_blocks/b7/simple_guideline.yml
    :linenos:
    :caption: graph_description.yml

Let's say we want to iterate over number of nodes and run an algorithm. We want to estimate the variance of the results by bootstrapping, so we run the algorithm for each value 2 times:

.. literalinclude:: code_blocks/b7/graph.py
    :linenos:
    :caption: graph.py
    :emphasize-lines: 4-6

The function ``guideline_iterator`` receives the guideline file and the target parameter to be looped over, as well as number of repeats per each value. It then returns an iterable object which gives a tuple of `(guideline_path, epoch, value)` at each round. Next, we can pass the guideline path as before to the randomizer, and use the epoch and value to log the results.

The behavior of the function for different directives is as follows:

- if ``[i-range, a, b]`` then it enumerates integers of `[a, b]` (`b` included).
- if ``[f-range, a, b]`` then it gives ``steps=n`` equidistant values between `a` and `b`. the `steps` parameter is needed in this case.
- if ``[choice, a, b, ...]`` then it iterates over the options in the same order.
- if fixed, then it raises an error, saying that the directive is a fixed value.



