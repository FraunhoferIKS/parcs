Interventions and Counterfactuals
=================================

In addition to the :code:`.sample()` method, which allows us to sample from the observational distribution, the graph object provides methods to sample from interventional distributions, defined in different three types.

Fixed-value Intervention
------------------------

This method sets the value of one or more nodes to a fixed value. As this is the most basic type of intervention in causal inference theory, we name the method simply :code:`.do()`. Below is an example of a causal triangle graph, with fixed-value interventions applied.

.. literalinclude:: code_blocks/interventions/fixed_value_do.py
    :linenos:
    :lines: 5-28
    :emphasize-lines: 6, 14, 15, 21

.. _functional_intervention:

Functional Intervention
-----------------------

In this type of intervention which you do via :code:`.do_functional()`, you can set a node's value to a deterministic function of other nodes. The function is a Python function (defined by :code:`def` keyword or as a :code:`lambda` function). Using this method, you can induce the *soft (parametric) intervention* scenario, where the inputs of the intervention function are the parents of the node in the original DAG. Nevertheless, the inputs of the function can be any subset of the nodes, except for the descendants of the intervened node.

.. literalinclude:: code_blocks/interventions/functional_do.py
    :linenos:
    :lines: 5-20
    :emphasize-lines: 7-11

.. _self_intervention:

Self Intervention
-----------------

Imagine the following causal question: *what if for every patient, we administer 1 unit of drug less than what we normally administer*. The "do" term for this intervention would be :math:`\text{do}(A=f(A_\text{old}))`. To simulate data for this type of intervention, we use the method :code:`.do_self()`

.. literalinclude:: code_blocks/interventions/self_do.py
    :linenos:
    :lines: 5-26
    :emphasize-lines: 14-17

.. note:: In the example above, line 16 tells the method to reuse the samples. This decision can be made in previous types of intervention as well. Using this feature, we can simulate **counterfactual**  scenarios: *what would have happened had we changed our actions?* In this case, the intervention must be applied to the same sampled dataset, and this can be done by reusing the sampled errors.