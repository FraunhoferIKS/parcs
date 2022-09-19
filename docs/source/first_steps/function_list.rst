.. _function_list:

=====================================
Available functions and distributions
=====================================

For all functions and distributions, you can find a link to the API doc on top of the parameters table, including more details.

.. _available_output_distributions:

Output distributions
====================

The distributions are provided by `Scipy's` `stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ module. Instead of calling the ``.rvs()`` method, however, we use the ``.ppf()`` (precent point function) method to obtain realizations given the error terms.

``bernoulli``
-------------

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Correction
   * - ``p_``
     - success probability.
     - :ref:`Sigmoid correction <sigmoid_correction>`

``gaussian``
------------

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Correction
   * - ``mu_``
     - mean of the distribution
     - N/A
   * - ``sigma_``
     - standard deviation of the distribution
     - :ref:`Sigmoid correction <sigmoid_correction>`

.. _available_edge_functions:

Edge functions
==============

``identity``
------------

.. list-table:: :func:`~parcs.cdag.mapping_functions.edge_identity`
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * -
     -
     -


``sigmoid``
-----------

.. list-table:: :func:`~parcs.cdag.mapping_functions.edge_sigmoid`
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - :math:`\alpha`: ``alpha``
     - | Scale of the Sigmoid. Takes positive values
       | The reasonable range is approx. :math:`[0.5, 6]`
     - ``1``
   * - :math:`\beta`: ``beta``
     - | offset of the Sigmoid. Takes real values.
       | The reasonable range is approx. :math:`[-0.8, 0.8]`
     - ``0``
   * - :math:`\gamma`: ``gamma``
     - mirroring on x-axis. Takes :math:`\{0, 1\}`
     - ``1``
   * - :math:`\tau`: ``tau``
     - | induces a dwelling region in the center of the function.
       | Takes odd values :math:`\{1, 3, \dots\}`
     - ``1``

``gaussian_rbf``
----------------

.. list-table:: :func:`~parcs.cdag.mapping_functions.gaussian_rbf`
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - :math:`\alpha`: ``alpha``
     - | Scale of the RBF. Takes positive values
       | The reasonable range is approx. :math:`[0.5, 6]`
     - ``1``
   * - :math:`\beta`: ``beta``
     - | offset of the RBF. Takes real values.
       | The reasonable range is approx. :math:`[-0.8, 0.8]`
     - ``0``
   * - :math:`\gamma`: ``gamma``
     - mirroring on y-axis. Takes :math:`\{0, 1\}`
     - ``0``
   * - :math:`\tau`: ``tau``
     - | induces a dwelling region in the center of the function.
       | Takes even values :math:`\{2, 4, \dots\}`
     - ``2``

Corrections
===========

.. _sigmoid_correction:

Sigmoid Correction
------------------

.. list-table:: :func:`~parcs.cdag.utils.SigmoidCorrection`
   :widths: 10 70 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - type
     - default (yml/py)
   * - ``lower, upper``
     - lower and upper range of the corrected values.
     - `float`
     - `0, 1`
   * - ``to_center``
     - | If the raw values must be centered (mean = 0)
       | before sigmoid correction.
     - `bool`
     - `false/False`
   * - ``target_mean``
     - | specify this option if you want to fix the mean
       | of corrected values. must be in range of (lower, upper)
     - *float*
     - `null/None`

Edge Correction
------------------

.. list-table:: :func:`~parcs.cdag.utils.EdgeCorrection`
   :widths: 10 70 10 10
   :header-rows: 1

   * - Parameter
     - Description
     - type
     - default (yml/py)
   * -
     -
     -
     -
