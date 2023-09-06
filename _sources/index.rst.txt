
|

.. image:: ../../images/parcs_light.svg
  :width: 400
  :align: center
  :target: https://www.iks.fraunhofer.de/

.. container::

    .. container:: leftside

        |

    .. container:: rightside

        .. image:: ../../images/iks.png
          :width: 150
          :align: left
          :target: https://www.iks.fraunhofer.de/

|
|

PARCS Documentation
-------------------

**PA**-rtially **R**-andomized **C**-ausal **S**-imulator is a simulation tool for causal methods. This library is designed to facilitate simulation study design and serve as a standard benchmarking tool for causal inference and discovery methods. PARCS generates simulation mechanisms based on causal DAGs and a wide range of adjustable parameters. Once the simulation setup is described via legible instructions and rules, PARCS automatically probes the space of all complying mechanisms and synthesizes data from both observational and interventional distributions.

Install PARCS via the following command:

.. code-block:: console

    (.venv) $ pip install pyparcs

.. _citation:

Citation
--------

.. note::
   This project is under active development.

Citation information will be put here in the near future.

|

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   getting_started/theoretical_background
   getting_started/first_graph
   getting_started/function_list
   getting_started/nodes_and_edges
   getting_started/partial_randomization

.. toctree::
   :maxdepth: 1
   :caption: More Features

   features/interventions
   features/looping
   features/temporal_graph
   features/m_graphs

.. toctree::
   :maxdepth: 1
   :caption: Technical Details

   technical/conventions
   technical/api
   technical/diagrams