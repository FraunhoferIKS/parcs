.. PARCS documentation master file, created by
   sphinx-quickstart on Fri Aug 26 12:47:10 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to PARCS documentation!
===============================

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


Documentation
-------------

.. toctree::
   :maxdepth: 1
   :caption: First Steps

   first_steps/theoretical_background
   first_steps/first_graph
   first_steps/function_list

.. toctree::
   :maxdepth: 1
   :caption: More Features

   features/interventions
   features/partial_randomization
   features/looping
   features/more_graph_objects
   features/m_graphs

.. toctree::
   :maxdepth: 1
   :caption: Technical Details

   technical/graph_objects
   technical/api
   technical/conventions


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
