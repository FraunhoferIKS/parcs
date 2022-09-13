Interventions, Counterfactuals
==============================

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