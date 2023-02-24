.. _function_list:

=====================================
Available functions and distributions
=====================================

.. _available_output_distributions:

Output distributions
====================

The distributions are provided by `Scipy's` `stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ module. Instead of calling the ``.rvs()`` method, however, we use the ``.ppf()`` (percent point function) method to obtain realizations given the error terms.

For the parameters with described *range*, we can specify correction.

.. list-table::
   :widths: 20 60 20
   :header-rows: 1

   * - Distribution
     - Parameters
     - Range
   * - ``bernoulli``
     - ``p_``: success probability
     - :math:`[0, 1]`
   * - ``uniform``
     - ``mu_``: center point: `(a+b)/2`
     -
   * -
     - ``diff_``: range: `b-a`
     -
   * - ``gaussian``
     - ``mu_``: mean
     -
   * -
     - ``sigma_``: standard deviation
     - :math:`[0, \infty)`
   * - ``lognormal``
     - ``mu_``: mean
     -
   * -
     - ``sigma_``: standard deviation
     - :math:`[0, \infty)`
   * - ``poisson``
     - ``lambda_``: rate
     - :math:`[0, \infty)`
   * - ``Exponential``
     - ``lambda_``: rate
     - :math:`[0, \infty)`

.. _available_edge_functions:

Edge functions
==============

Read more about each edge function by clicking the link in the table title.

``identity``
------------

.. list-table:: :func:`~pyparcs.cdag.mapping_functions.edge_identity`
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

.. list-table:: :func:`~pyparcs.cdag.mapping_functions.edge_sigmoid`
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - :math:`\alpha`: ``alpha``
     - | Scale of the Sigmoid. Takes positive values
       | The reasonable range is approx. :math:`[1, 3]`
     - ``2``
   * - :math:`\beta`: ``beta``
     - | offset of the Sigmoid. Takes real values.
       | The reasonable range is approx. :math:`[-2, 2]`
     - ``0``
   * - :math:`\gamma`: ``gamma``
     - mirroring on x-axis. Takes :math:`\{0, 1\}`
     - ``0``
   * - :math:`\tau`: ``tau``
     - | induces a dwelling region in the center of the function.
       | Takes odd values :math:`\{1, 3, \dots\}`
     - ``1``

``gaussian_rbf``
----------------

.. list-table:: :func:`~pyparcs.cdag.mapping_functions.edge_gaussian_rbf`
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - :math:`\alpha`: ``alpha``
     - | Scale of the RBF. Takes positive values
       | The reasonable range is approx. :math:`[1, 3]`
     - ``2``
   * - :math:`\beta`: ``beta``
     - | offset of the RBF. Takes real values.
       | The reasonable range is approx. :math:`[-2, 2]`
     - ``0``
   * - :math:`\gamma`: ``gamma``
     - mirroring on y-axis. Takes :math:`\{0, 1\}`
     - ``0``
   * - :math:`\tau`: ``tau``
     - | induces a dwelling region in the center of the function.
       | Takes even values :math:`\{2, 4, \dots\}`
     - ``2``

``arctan``
----------------

.. list-table:: :func:`~pyparcs.cdag.mapping_functions.edge_arctan`
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Default
   * - :math:`\alpha`: ``alpha``
     - | Scale of the RBF. Takes positive values
       | The reasonable range is approx. :math:`[1, 3]`
     - ``2``
   * - :math:`\beta`: ``beta``
     - | offset of the RBF. Takes real values.
       | The reasonable range is approx. :math:`[-2, 2]`
     - ``0``
   * - :math:`\gamma`: ``gamma``
     - mirroring on y-axis. Takes :math:`\{0, 1\}`
     - ``0``

.. _corrections:

Corrections
===========

.. _sigmoid_correction:

Sigmoid Correction
------------------

.. list-table:: :func:`~pyparcs.cdag.utils.SigmoidCorrection`
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

.. list-table:: :func:`~pyparcs.cdag.utils.EdgeCorrection`
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
