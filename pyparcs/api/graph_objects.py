#  Copyright (c) 2022. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
#  acting on behalf of its Fraunhofer-Institut für Kognitive Systeme IKS. All rights reserved.
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, see <http://www.gnu.org/licenses/>.
#
#  https://www.gnu.de/documents/gpl-2.0.de.html
#
#  Contact: alireza.zamanian@iks.fraunhofer.de
from pathlib import Path
from typing import Optional, Union, List
from typeguard import typechecked
import numpy as np
import pandas as pd
from pyparcs.api.corrections import EdgeCorrection
from pyparcs.api.output_distributions import OUTPUT_DISTRIBUTIONS
from pyparcs.api.mapping_functions import EDGE_FUNCTIONS
from pyparcs.core.exceptions import (parcs_assert, validate_deterministic_function,
                                     validate_error_term, ExternalResourceError, DistributionError)


OUTPUT_DISTRIBUTIONS_KEYS = OUTPUT_DISTRIBUTIONS.keys()
EDGE_FUNCTIONS_KEYS = EDGE_FUNCTIONS.keys()
REPORT_TYPES = ['md', 'raw']


@typechecked
class Node:
    """ **Stochastic Node object in causal DAGs**

    This class represents the DAG's stochastic nodes, which is the basic form of the nodes.
    Node can be sampled by passing the data to the ``.calculate()`` method.

    Parameters
    ----------
    output_distribution : str
        Selected from the available distributions. call ``graph_objects.OUTPUT_DISTRIBUTIONS``
        to see the list.
    do_correction : bool, default=False
        perform correction over the result obtained from parent nodes
        in order to comply with distribution parameters restrictions.
        If ``do_correction=True``,then a ``correction_config`` must be included in the
        ``dist_config`` argument. The corresponding value is a dictionary of correction configs
        for each distribution parameter.
    correction_config : dict
        config dictionary for correction.
        If ``do_correction=True``, then it is read.
    dist_params_coefs : dict
        first-level keys are names of parameter distributions.
        for each parameter, three *bias*, *linear* and *interactions* key are given.
        The bias value is a float, while linear and interaction values are numpy arrays.
    """
    def __init__(self,
                 output_distribution: str,
                 dist_params_coefs: dict,
                 do_correction: bool = False,
                 correction_config=None):
        if correction_config is None:
            correction_config = {}
        parcs_assert(output_distribution in OUTPUT_DISTRIBUTIONS_KEYS, DistributionError,
                     f'output_distribution should be in {OUTPUT_DISTRIBUTIONS_KEYS}, '
                     f'got {output_distribution} instead')
        # basic attributes
        self.info = {
            'output_distribution': output_distribution,
            'dist_params_coefs': dist_params_coefs
        }
        self.output_distribution = OUTPUT_DISTRIBUTIONS[output_distribution](
            correction_config=correction_config,
            do_correction=do_correction,
            coefs=dist_params_coefs
        )
        self.do_correction = do_correction

    def get_info(self):
        if self.do_correction:
            self.info['correction'] = self.output_distribution.sigma_correction.get_params()
        return self.info

    def calculate(self,
                  data: pd.DataFrame,
                  parents: List[str],
                  errors: pd.Series,
                  **kwargs) -> np.ndarray:
        return self.output_distribution.calculate(
            data[parents].values,
            errors
        )


@typechecked
class DetNode:
    """ **Deterministic Node object in causal DAGs**

    Use this class to create a node, independent of any graphs.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph`
    class instead.
    Deterministic node yields the output by calling the `.calculate()` method.

    Parameters
    ----------
    function : callable
        the user-defined function that represents the node.

    Examples
    --------
    >>> from pyparcs.api.graph_objects import DetNode
    >>> import pandas
    >>> n_0 = DetNode(function=lambda d: d['N_1']+d['N_2'])
    >>> data = pandas.DataFrame([[1, 2], [2, 2], [3, 3]], columns=('N_1', 'N_2'))
    >>> n_0.calculate(data, ['N_1', 'N_2'])
    array([3, 4, 6])
    """

    def __init__(self, function):
        self.info = {'node_type': 'deterministic'}
        self.function = function
        validate_deterministic_function(self.function)

    def get_info(self):
        return self.info

    @typechecked
    def calculate(self, data: pd.DataFrame, **kwargs):
        try:
            return self.function(data).values
        except KeyError as exc:
            raise ExternalResourceError("assumed parent names for node") from exc


@typechecked
class ConstNode:
    """ **Constant Node object in causal DAGs**

    Use this class to create a node with constant value.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph`
    class instead.

    Parameters
    ----------
    value : float
        constant value of the node

    Examples
    --------
    >>> from pyparcs.api.graph_objects import ConstNode
    >>> import pandas
    >>> n_0 = ConstNode(value=2)
    >>> data = pandas.DataFrame([[1, 2], [2, 2], [3, 3]], columns=('N_1', 'N_2'))
    >>> n_0.calculate(data)
    array([3, 4, 6])
    """

    def __init__(self,
                 value: Union[np.number, np.int, float]):
        self.info = {'node_type': 'constant', 'value': value}

    def get_info(self):
        return self.info

    def calculate(self, size_=Optional[int], **kwargs):
        return np.ones(shape=(size_,)) * self.info['value']


@typechecked
class DataNode:
    """ **Data Node object in causal DAGs**

    Use this class to create a node with the samples being read from an external data file.
    If you want to construct a causal DAG, please use the :func:`~pyparcs.cdag.graph_objects.Graph`
    class instead.

    Parameters
    ----------
    csv_dir : str
        CSV file directory
    """

    def __init__(self,
                 csv_dir: Union[str, Path],
                 col: str):
        self.col = col
        self.samples = pd.read_csv(csv_dir)[self.col]
        self.info = {'node_type': 'data', 'size': len(self.samples)}

    def get_info(self):
        return self.info

    @typechecked
    def calculate(self, errors: pd.Series, **kwargs) -> np.ndarray:
        validate_error_term(errors, self.col)
        ind = np.floor(errors * len(self.samples))
        ind.replace(self.info['size'], self.info['size']-1, inplace=True)
        return self.samples.iloc[ind].values


@typechecked
class Edge:
    """ **Edge object in causal DAGs**

    Use this class to create an edge, independent of any graphs.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph`
    class instead. Edge object receives an array, and maps it based on the edge function.
    If ``do_correction = True``, batch normalization parameters are set upon the next data batch,
    and be used in further transformations. :ref:`(read more) <edge_doc>`

    Parameters
    ----------
    name : str, optional
        a name in the form of 'X->Y' where X and Y are parent and chile dones respectively.
    do_correction : bool, default=True
        if ``True``, then batch normalization is done. Parameters of normalization are stored
        upon first run. if ``True``, then the length of the first batch must be ``>1000``.
    function_name : str
        Selected from the available edge functions. call ``graph_objects.EDGE_FUNCTIONS``
        to see the list.
    function_params : dict
        a dictionary of function parameters, must fit the function name.
        in case of empty dict ``{}`` it uses the default parameters of function.

    Attributes
    ----------
    edge_function : dict
        A dictionary of the following values: `name`, `function` and `params` of the edge function

    Examples
    --------
    >>> import numpy
    >>> from pyparcs.api.graph_objects import Edge
    >>> # a standard Sigmoid activation
    >>> edge = Edge(
    ...     function_name='sigmoid',
    ...     function_params={},
    ...     do_correction=False
    ... )
    >>> x = numpy.array([-10, -1, 0, 1, 10])
    >>> x_mapped = numpy.round( edge.map(x), 2)
    >>> x_mapped
    array([0.  , 0.12, 0.5 , 0.88, 1.  ])
    """
    def __init__(self,
                 function_name: str,
                 function_params: dict = {},
                 do_correction=False,
                 name: Optional[str] = None):
        parcs_assert(function_name in EDGE_FUNCTIONS_KEYS, DistributionError,
                     f'function_name should be in f{EDGE_FUNCTIONS_KEYS},'
                     f'got {function_name} instead')
        self.name = name
        self.do_correction = do_correction
        if self.do_correction:
            self.corrector = EdgeCorrection()

        self.edge_function = {
            'name': function_name,
            'function': EDGE_FUNCTIONS[function_name],
            'params': function_params
        }
        self.info = {
            'edge_function': function_name,
            'function_params': function_params
        }

    def get_info(self):
        if self.do_correction:
            self.info['correction'] = self.corrector.get_params()
        return self.info

    def map(self, array: np.ndarray):
        """ **maps an input array**

        This method maps a given input array based on set `edge_function` and
        `do_correction` values.

        Parameters
        ----------
        array : 1D numpy array
            input array

        Returns
        -------
        1D numpy array
            the transformed array

        """
        if self.do_correction:
            array = self.corrector.transform(array)
        return self.edge_function['function'](
            array=array,
            **self.edge_function['params']
        )


NODE_OBJECT_DICT = {
    'stochastic': Node,
    'deterministic': DetNode,
    'data': DataNode,
    'constant': ConstNode
}
