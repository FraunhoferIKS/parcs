========
Diagrams
========

Main Classes and Functions
==========================

.. _uml_description_instantiation:

Description: Instantiation
--------------------------

.. image:: ./UMLs/img/description_instantiation.svg

.. seealso::
    * :ref:`description_parser() sequence diagram <uml_description_parser>`
    * :ref:`get_adj_matrix() flowchart <flowchart_get_adj_matrix>`
    * :ref:`topological_sort() flowchart <flowchart_topological_sort>`

Description: Parameter Randomization
------------------------------------

.. image:: ./UMLs/img/description_randomize_parameters.svg

.. _uml_description_randomize_connection:

Description: Randomize Connection
---------------------------------

.. image:: ./UMLs/img/description_randomize_connection.svg

.. seealso::
    * :ref:`augment_line() flowchart <flowchart_augment_line>`

Graph
-----

.. image:: ./UMLs/img/graph.svg

PARCS Backend
=============

.. _flowchart_get_adj_matrix:

Adjacency Matrix
----------------

Flowchart for :func:`pyparcs.api.utils.get_adj_matrix`

.. image:: ./UMLs/img/adj_matrix.svg

.. _flowchart_augment_line:

Augment Line
------------

This function adds extra parents to the parameter equations in the process of :func:`pyparcs.Description.randomize_connection_to`

.. image:: ./UMLs/img/augment_line.svg

.. seealso::
    * :ref:`Description.randomize_connection_to() sequence diagram <uml_description_randomize_connection>`

Description Parser
------------------

.. _uml_description_parser:

Main Function
~~~~~~~~~~~~~

.. image:: ./UMLs/img/description_parser.svg

Node Parser
~~~~~~~~~~~

.. image:: ./UMLs/img/node_parser.svg

To parse stochastic nodes, the distribution parameters are parsed according to the following:

.. image:: ./UMLs/img/equation_parser.svg

Edge Parser
~~~~~~~~~~~

.. image:: ./UMLs/img/edge_parser.svg

.. seealso::
    * :ref:`Node and Edge dictionaries schemas <graph_objects_schemas>`

.. _graph_objects_schemas:

Graph Objects Schemas
---------------------

.. image:: ./UMLs/img/graph_objects.svg

.. _flowchart_topological_sort:

Topological Sort
----------------

Flowchart for :func:`pyparcs.api.utils.topological_sort`

.. image:: ./UMLs/img/topological_sort.svg