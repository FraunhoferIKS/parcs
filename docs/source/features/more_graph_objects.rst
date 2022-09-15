==================
More Graph Objects
==================

As mentioned in :ref:` the first steps <create_the_first_graph>`, the main objects are nodes which are defined by standard probability distributions and edges that are defined by PARCS edge functions. In this section we introduce other types of nodes and edges that you can use in your PARCS simulation


Deterministic nodes
===================

A node can be defined to output a deterministic function of its parents. In this case, a deterministic function replaces the output distribution. To define a deterministic node:

1. create a `.py` file in the same directory as your main Python script
2. Define your custom function in the file
3. define the node in the `description file` by ``deterministic(<.py file name>, <function name>)``

.. literalinclude:: code_blocks/b1/customs.py
    :caption: :code:`customs.py`
    :linenos:

.. literalinclude:: code_blocks/b1/graph_description.yml
    :caption: :code:`graph_description.yml`
    :linenos:

.. literalinclude:: code_blocks/b1/graph.py
    :caption: :code:`graph.py`
    :linenos:


