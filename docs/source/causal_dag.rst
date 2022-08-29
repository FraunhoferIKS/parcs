====================
Causal DAGs in PARCS
====================

In PARCS, Directed Acyclic Graph (DAG) is used to model the ground-truth of the synthesized data. Here we introduce 3 main PARCS objects related to DAGs: *nodes*, *edges* and *graphs*:


Edges
=====

An ``Edge`` object is primarily defined by a pair of parent-child nodes :math:`(Z_i, Z_j)` and an *edge function* :math:`e_{ij}(.)` along with its parameters. The edge function maps the input values from the parent node and create a transformed array, to be then passed to the child node as the actual input.

.. math::
    \begin{align}
    z^*_i = e_{ij}(z_i)
    \end{align}
    :label: edge-1

An edge object is declared via the following code:

.. literalinclude:: examples/cdag_examples/edge_identity.py
    :lines: 1-14

where ``function_name`` argument determines the selected edge function. The simplest function , ``identity`` , follows the identity transform:

.. math::
    z_i^* = z_i.

Other available edge functions are introduced below.

Edge functions
--------------

Sigmoid function
~~~~~~~~~~~~~~~~
Sigmoid edge function is activated by ``function_name='sigmoid'`` and follows the transformation of

.. math::
    \begin{align}
        z^*_i = \sigma\Big(
            (-1)^\gamma.  2  \alpha  (z_i - \beta)^\tau
        \Big),
    \end{align}
    :label: edge_sigmoid_1

Parameters in Eq. :eq:`edge_sigmoid_1` are :math:`\gamma, \alpha, \beta` and :math:`\tau`, and they control the overall shape of the sigmoid function (Figure ?). Combined together, we can achieve different forms of sigmoid function using different parameter sets. The parameters are passed to the class as a dictionary:

.. literalinclude:: examples/cdag_examples/edge_sigmoid.py
    :lines: 4-17

Gaussian RBF function
~~~~~~~~~~~~~~~~~~~~~
Gaussian RBF function is activated by ``function_name='gaussian_rbf'`` and follows the transformation of

expon = -alpha * ((array - beta)**tau)
    return gamma + ((-1)**gamma) * np.exp(expon)

.. math::
    \begin{align}
        z^*_i = \gamma + (-1)^\gamma .
        \exp\big(-\alpha \|z_i - \beta\|^\tau\big),
    \end{align}
    :label: edge_gaussian_rbf_1

Parameters in Eq. :eq:`edge_gaussian_rbf_1`, similar to :eq:`edge_sigmoid_1` parameters, control the overall shape of the radial basis function (Figure ?):

.. literalinclude:: examples/cdag_examples/edge_gaussian_rbf.py
    :lines: 4-17


Input normalization
-------------------
During randomization of simulation in PARCS, we need to normalize the inputs before applying the edge function. by setting ``do_correction=True``, the edge object calculates the empirical mean and standard deviation of the input data, and normalizes it as

.. math::
    \begin{align}
        \tilde{z}_i = \frac{z_i - \mu_q}{\sigma_q}
    \end{align}
    :label: edge_correction

where the subscript :math:`q` means that the statistics are calculated for the truncated input, excluding first and last *q* quantiles. This is done to prevent improper normalization in the presence of outliers. By default, the quantile is set to :math:`0.05`

.. literalinclude:: examples/cdag_examples/edge_correction.py
    :lines: 4-22

For the next data batch, edge uses the calculated statistics from the first feed. Therefore this normalization becomes the characteristics of the simulation model and stays fixed for all further sampling.

.. note::
    Since the statistics are calculated on the first batch, PARCS throws an instability warning if the first batch is ``<500``.