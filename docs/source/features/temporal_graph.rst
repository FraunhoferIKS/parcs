Temporal Graphs
===============

In PARCS, you can define a temporal graph where the nodes and relations are expanded along the
time axis. This is possible by only making a few small modifications to the current pipeline.
Let's imagine the following scenario to explain this feature:
The goal is to model the interaction between the blood pressure variable :code:`BP` and the drug
variable :code:`Drug`.
Initially, we let the blood pressure be dependent on gaussian noise :code:`Noise`, while the drug
depends on the age variable :code:`Age` of the patient as well as the blood pressure. This setup
can be modeled by the non-temporal version of PARCS.

.. literalinclude:: code_blocks/temporal/base_description.yml
    :caption: a static graph description file
    :linenos:
    
.. _configuration_file:

Temporal Syntax in the Graph Description
----------------------------------------

Now we discuss the temporal expansion: every temporal component is marked by a subscript where :code:`_{t}` indicates the current timestep.

.. literalinclude:: code_blocks/temporal/temporal_graph_description.yml
    :linenos:

Note, how the :code:`Age` node lacks the :code:`_{t}` subscript; it is considered
a baseline variable (BLV) that is created at the beginning and does not change. Additionally, we make a distinction between
temporal recursive and temporal non-recursive variables. Both :code:`BP` and :code:`Drug` belong to the former class because 
they depend on nodes from previous timesteps. On the other hand, :code:`Noise` is not conditioned on a previous inputs and
thus is considered temporal non-recursive.
In general, variables can exhibit long-range dependencies that go further back than just the previous timestep. 
Therefore, we add another temporal dependency from :code:`Drug` to :code:`BP`, where the effect is delayed by two timesteps.

For all temporal variables the first timestep of the simulation is indexed as :code:`1`.
All temporal recursive definitions require a set of initial values that have to be specified.
In the case of :code:`Drug`, for example, one has to specify the values for :code:`Drug_-1` and
:code:`Drug_0` because we reference :code:`Drug_-1` in the definition of :code:`BP_1`.

.. _parsing_calling:

Parsing and Calling Temporal Outlines
-------------------------------------

To actually parse your temporal outline, all you need to do is to instantiate the ``TemporalDescription`` class with a temporal outline and a ``n_timesteps`` argument.

.. literalinclude:: code_blocks/temporal/temporal_graph.py
    :linenos:
    :emphasize-lines: 5

.. _temporal_deterministic_nodes:

Deterministic Nodes in Temporal Graphs
--------------------------------------

As introduced in :ref:`this section <deterministic_nodes>`, deterministic nodes are declared by a custom user-defined python function. What if the deterministic node is temporal? e.g. instead of having a function like :math:`C = A^2 + B`, we have :math:`C_t = A_{t-2}^2 + B_{t}`...

Writing the custom function for a deterministic node is almost the same as writing a static function, with a small modification. Below is the temporal description file:

.. literalinclude:: code_blocks/temporal/det_outline.yml
    :linenos:


Firstly, the data columns in the custom function is just as we must expect it, e.g., :code:`data['A_{t-1}']` or :code:`data['B_{t}']`. Secondly, you must use the :code:`temporal` decorator for your custom function. An example is depicted below:

.. literalinclude:: code_blocks/temporal/customs.py
    :linenos:

In this code, the :code:`temporal_deterministic` decorator receives two inputs:

-  list of all temporal nodes **in the function**;
-  the earliest time index that appears **in the function**.

According to this code, the nodes :code:`C_1, C_2, C_3` will be calculated as

.. math::
    C_1 &= A+B_{-1}+C_{0}, \\
    C_2 &= A+B_{0}+C_{1}, \\
    C_3 &= A+B_{1}+C_{2}.

Finally, here is the PARCS main code:

.. literalinclude:: code_blocks/temporal/main.py
    :linenos: