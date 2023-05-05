Temporal Graph
==============================

In PARCS, you can define a temporal graph where the nodes and relations are expanded along the
time axis. This is possible by only making a few small modifications to the current pipeline.
Let's imagine the following scenario to explain this feature:
The goal is to model the interaction between the blood pressure variable :code:`BP` and the drug
variable :code:`Drug`.
Initially, we let the blood pressure be dependent on gaussian noise :code:`Noise`, while the drug
depends on the age variable :code:`Age` of the patient as well as the blood pressure. This setup
can be modeled by the non-temporal version of PARCS.

.. literalinclude:: code_blocks/b10/base_description.yml 
    :linenos:
    
.. _configuration_file:

Temporal syntax in Configuration File
-----------------------

Now we discuss the temporal expansion: every temporal component is marked by a subscript where :code:`_{t}` indicates the current timestep.
The total number of timesteps is denoted by :code:`n_timesteps`. Note, how the :code:`Age` node lacks the :code:`_{t}` subscript; it is considered
a baseline variable (BLV) that is created in the beginning and does not change. Additionally, we make a distinction between
temporal recursive and temporal non-recursive variables. Both :code:`BP` and :code:`Drug` belong to the former class because 
they depend on nodes from previous timesteps. An the other hand, :code:`Noise` is not conditioned on a previous inputs and
thus is considered temporal non-recursive.
In general, variables can exhibit long-range dependencies that go further back than just the previous timestep. 
Therefore, we add another temporal dependency from :code:`Drug` to :code:`BP`, where the effect is delayed by two timesteps. 

.. literalinclude:: code_blocks/b10/temporal_graph_description.yml 
    :linenos:
    :lines: 1-7,9,12-30

For all temporal variables the first timestep of the simulation is indicated by :code:`_1`.
All temporal recursive definitions require a set of initial values that have to be specified.
In the case of :code:`Drug` ,for example, one has to specify the values for :code:`Drug_-1` and
:code:`Drug_0` because we reference :code:`Drug_-1` in the definition of :code:`BP_1`.

.. literalinclude:: code_blocks/b10/temporal_graph_description.yml 
    :linenos:
    :lines: 8-12
    :emphasize-lines: 1,3,4

Putting everything together, we obtain the final configuration file. The emphasized lines show the edge dependencies 
that can be now modeled with the temporal version of PARCS.

.. literalinclude:: code_blocks/b10/temporal_graph_description.yml 
    :linenos:
    :emphasize-lines: 25,26,30

.. _parsing_calling:

Parsing and Calling Yaml File
-----------------------

To actually parse your temporal configuration file, all you need to do is calling the temporal parser
which creates all nodes and edges based on the given timestep.

.. literalinclude:: code_blocks/b10/temporal_graph.py
    :linenos:
    :emphasize-lines: 6

