=====================================
Available Functions and Distributions
=====================================

.. _available_output_distributions:

Output distributions
====================

The distributions are provided by `Scipy's` `stats <https://docs.scipy.org/doc/scipy/reference/stats.html>`_ module. Instead of calling the ``.rvs()`` method, however, we use the ``.ppf()`` (precent point function) method to obtain realizations given the error terms.

``bernoulli``
-------------
`Bernoulli distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.bernoulli.html#scipy.stats.bernoulli>`_ defines a binary output variable.

.. list-table:: Bernoulli Distribution parameters
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Affected by Correction
   * - ``p_``
     - success probability.
     - Yes

Bernoulli correction is a :ref:`Sigmoid correction <sigmoid_correction>` which applies to ``p_`` parameter. For Bernoulli distribution, upper and lower must be 0, 1 (default values)

--------------

``gaussian``
------------
`Gaussian normal distribution <https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html#scipy.stats.norm>`_ defines a normally distributed variable.

.. list-table:: Bernoulli Distribution parameters
   :widths: 20 60 20
   :header-rows: 1

   * - Parameter
     - Description
     - Affected by Correction
   * - ``mu_``
     - mean of the distribution
     - No
   * - ``sigma_``
     - standard deviation of the distribution
     - Yes

Similar to Bernoulli, Gaussian correction is a Bernoulli correction is a :ref:`Sigmoid correction <sigmoid_correction>` which applies to ``sigma_`` parameter.

.. _available_edge_functions:

Edge functions
==============

``identity``
------------

As the name indicates, this function provides an identity mapping. However, it can be still affected by edge correction.

``sigmoid``
-----------

Sigmoid maps the raw values using a sigmoid function. The function is parameterized by a number of parameters presented in the table. The formula is:

.. math::
    \begin{align}
        z^*_i = \sigma\Big(
            (-1)^\gamma.  2  \alpha  (z_i - \beta)^\tau
        \Big).
    \end{align}
    :label: edge_sigmoid

.. list-table:: Sigmoid edge function parameters
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

Figures below the table illustrate how each parameter changes the sigmoid function.

``gaussian_rbf``
-----------

Gaussian RBF maps the raw values using the Gaussian radial basis function. The function is parameterized by a number of parameters presented in the table. The formula is:

.. math::
    \begin{align}
        z^*_i = \gamma + (-1)^\gamma .
        \exp\big(-\alpha \|z_i - \beta\|^\tau\big),
    \end{align}
    :label: edge_gaussian_rbf

.. list-table:: Gaussian RBF edge function parameters
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

Figures below the table illustrate how each parameter changes the RBF function.

Corrections
===========

.. _sigmoid_correction:

Sigmoid Correction
------------------
Sigmoid correction maps the real values to a ``[lower, upper]`` range, with a possibility to `center` the raw values, or set a `target mean` for the corrected values.

.. list-table:: Correction
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