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

import numpy as np
import pandas as pd
from pyparcs.cdag.utils import topological_sort, EdgeCorrection
from pyparcs.cdag.output_distributions import OUTPUT_DISTRIBUTIONS
from pyparcs.cdag.mapping_functions import EDGE_FUNCTIONS
from pyparcs.graph_builder.utils import info_md_parser
from pyparcs.exceptions import *
from typeguard import typechecked
from typing import Optional, Union, List, Callable
from pathlib import Path

OUTPUT_DISTRIBUTIONS_KEYS = OUTPUT_DISTRIBUTIONS.keys()
EDGE_FUNCTIONS_KEYS = EDGE_FUNCTIONS.keys()
REPORT_TYPES = ['md', 'raw']


@typechecked
class Node:
    """ **Node object in causal DAGs**

    Use this class to create a node, independent of any graphs.
    If you want to construct a causal DAG, please use the :func:`~pyparcs.cdag.graph_objects.Graph` class instead.
    Node can be sampled by passing the data and error terms (``pd.DataFrame``) to the ``.calculate()`` method.

    Parameters
    ----------
    output_distribution : str
        Selected from the available distributions. call ``graph_objects.OUTPUT_DISTRIBUTIONS`` to see the list.
    do_correction : bool, default=False
        perform correction over the result obtained from parent nodes
        in order to comply with distribution parameters restrictions.
        If ``do_correction=True``,then a ``correction_config`` must be included in the ``dist_config`` argument.
        The corresponding value is a dictionary of correction configs for each distribution parameter.
    correction_config : dict
        config dictionary for correction.
        If ``do_correction=True``, then it is read.
    dist_params_coefs : dict
        first-level keys are names of parameter distributions.
        for each parameter, three *bias*, *linear* and *interactions* key are given.
        The bias value is a float, while linear and interaction values are numpy arrays.
    """

    def __init__(self,
                 name: Optional[str],
                 output_distribution: str,
                 dist_params_coefs: dict,
                 do_correction: bool = False,
                 correction_config=None):
        if correction_config is None:
            correction_config = {}
        parcs_assert(output_distribution in OUTPUT_DISTRIBUTIONS_KEYS, DistributionError,
                     f'output_distribution should be in {OUTPUT_DISTRIBUTIONS_KEYS}, got {output_distribution} instead')
        # basic attributes
        self.name = name
        self.info = {
            'node_type': 'stochastic',
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

    def calculate(self, data: pd.DataFrame, parents: List[str], errors: pd.Series) -> np.ndarray:
        """ **calculates node's output the node**

        calculates the output of the noise based on the sampled errors and the data of parent nodes.

        Parameters
        ----------
        data : pd.DataFrame
            even if the node is source, data must be an empty data frame
        parents : list(str)
            list of parents, the order of which corresponds to the given parameter coefficient vector
        errors : numpy array
            sampled errors

        Returns
        -------
        samples : np.array
            calculated outputs based on parent samples and sampled errors

        """
        return self.output_distribution.calculate(
            data[parents].values,
            errors
        )


@typechecked
class DetNode:
    """ **Deterministic Node object in causal DAGs**

    Use this class to create a node, independent of any graphs.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph` class instead.
    Deterministic node yields the output by calling the `.calculate()` method.

    Parameters
    ----------
    function : callable
        the user-defined function that represents the node.

    Examples
    --------
    >>> from pyparcs.cdag.graph_objects import DetNode
    >>> import pandas
    >>> n_0 = DetNode(name='N_0', function=lambda d: d['N_1']+d['N_2'])
    >>> data = pandas.DataFrame([[1, 2], [2, 2], [3, 3]], columns=('N_1', 'N_2'))
    >>> n_0.calculate(data, ['N_1', 'N_2'])
    array([3, 4, 6])
    """

    def __init__(self,
                 function,
                 name: Optional[str] = None):
        self.name = name
        self.info = {'node_type': 'deterministic'}
        self.function = function
        validate_deterministic_function(self.function, self.name)

    def get_info(self):
        return self.info

    @typechecked
    def calculate(self, data: pd.DataFrame):
        try:
            return self.function(data).values
        except KeyError:
            raise ExternalResourceError("assumed parent names for node")


@typechecked
class ConstNode:
    """ **Constant Node object in causal DAGs**

    Use this class to create a node with constant value.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph` class instead.

    Parameters
    ----------
    name : str, optional
        name of the node. Optional unless the node is used in a graph
    value : float
        constant value of the node

    Examples
    --------
    >>> from pyparcs.cdag.graph_objects import ConstNode
    >>> import pandas
    >>> n_0 = ConstNode(name='N_0', value=2)
    >>> data = pandas.DataFrame([[1, 2], [2, 2], [3, 3]], columns=('N_1', 'N_2'))
    >>> n_0.calculate(data)
    array([3, 4, 6])
    """

    def __init__(self,
                 value: Union[np.number, np.int, float],
                 name: Optional[str] = None):
        self.name = name
        self.info = {'node_type': 'constant', 'value': value}

    def get_info(self):
        return self.info

    def calculate(self, size_=Optional[int]):
        return np.ones(shape=(size_,)) * self.info['value']


@typechecked
class DataNode:
    """ **Data Node object in causal DAGs**

    Use this class to create a node with the samples being read from an external data file.
    If you want to construct a causal DAG, please use the :func:`~pyparcs.cdag.graph_objects.Graph` class instead.

    Parameters
    ----------
    name : str, optional
        name of the node. Optional unless the node is used in a graph
    csv_dir : str
        CSV file directory
    """

    def __init__(self,
                 csv_dir: Union[str, Path],
                 name: str):
        self.name = name
        self.samples = pd.read_csv(csv_dir)[name]
        self.info = {'node_type': 'data', 'size': len(self.samples)}

    def get_info(self):
        return self.info

    @typechecked
    def calculate(self, sampled_errors: pd.Series) -> np.ndarray:
        validate_error_term(sampled_errors, self.name)
        ind = np.floor(sampled_errors * len(self.samples))
        return self.samples.iloc[ind].values


@typechecked
class Edge:
    """ **Edge object in causal DAGs**

    Use this class to create an edge, independent of any graphs.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph` class instead.
    Edge object receives an array, and maps it based on the edge function.
    If ``do_correction = True`` Then batch normalization parameters are set upon the next data batch,
    and be used in further transformations. :ref:`(read more) <edge_doc>`

    Parameters
    ----------
    name : str, optional
        a name in the form of 'X->Y' where X and Y are parent and chile dones respectively.
    do_correction : bool, default=True
        if ``True``, then batch normalization is done. Parameters of normalization are stored upon first run.
        if ``True``, then the length of the first batch must be ``>1000``.
    function_name : str
        Selected from the available edge functions. call ``graph_objects.EDGE_FUNCTIONS`` to see the list.
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
    >>> from pyparcs.cdag.graph_objects import Edge
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
                     f'function_name should be in f{EDGE_FUNCTIONS_KEYS}, got {function_name} instead')
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

        This method maps a given input array based on set `edge_function` and `do_correction` values.

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


@typechecked
class Graph:
    """**Causal DAG class**

    This class creates a causal DAG for the specified nodes and edges. A graph is equipped with sampling
    and intervention methods. For a comprehensive tutorial, see the :ref:`get started <create_the_first_graph>` doc.

    Parameters
    ----------
    nodes : list of dicts
        List of dictionaries whose keys are kwargs of the :func:`~pyparcs.cdag.graph_objects.Node` object.
    edges : list of dicts
        List of dictionaries whose keys are kwargs of the :func:`~pyparcs.cdag.graph_objects.Edge` object.
    dummy_node_prefix : str
        the prefix in the graph description file which identifies dummy nodes: dummy nodes will be suppressed
        from the output

    Attributes
    ----------
    nodes, edges : dicts
        dict of Graph nodes and edges, in the form of `{'node_name': Node()}` and `{'edge_name': Edge()}`

    cache : tuple of samples and error terms
        If caching is enabled upon sampling, then the result is cached in this attribute
    """

    def __init__(self,
                 nodes: List[dict],
                 edges: List[dict],
                 dummy_node_prefix: str = 'dummy_'):
        self.nodes: dict = {
            kwargs['name']: Node(**kwargs) if 'output_distribution' in kwargs
            else DetNode(**kwargs) if 'function' in kwargs
            else DataNode(**kwargs) if 'csv_dir' in kwargs
            else ConstNode(**kwargs)
            for kwargs in nodes
        }
        dummy_len = len(dummy_node_prefix)
        self.dummy_names = [n for n in self.nodes if n[:dummy_len] == dummy_node_prefix]
        self.node_types = {
            name: self.nodes[name].info['node_type'] for name in self.nodes
        }
        self.edges = {kwargs['name']: Edge(**kwargs) for kwargs in edges}
        self.parent_sets = {}
        self.adj_matrix = None
        self._set_adj_matrix()
        self.cache = {}

        # one-time sample to setup corrections
        # TODO: don't do it if no correction=True
        self.sample(size=500)

    def get_info(self, report_type: str = 'raw', info_dir: Optional[Union[str, Path]] = None):
        """ **getting nodes and edges information**

        This method gives the graph nodes and edges information

        Parameters
        ----------
        report_type : {'raw', 'md'}
            Type of the report. If `raw`, then returns a raw dict of nodes and edges info.
            If `md` then writes a markdown report in the ``info_dir`` directory.
        info_dir : str
            Directory of the report, if `type='md'`

        Returns
        -------
        None
            If `type='md'`
        info : dict
            If `type='raw'`

        """
        parcs_assert(type in REPORT_TYPES, DistributionError, f'type should be in {REPORT_TYPES}, got {type} instead')
        info = {
            'nodes': {n: self.nodes[n].get_info() for n in self.nodes},
            'edges': {e: self.edges[e].get_info() for e in self.edges}
        }
        if report_type == 'raw':
            return info
        elif report_type == 'md':
            with open(info_dir, 'w') as file:
                file.write(info_md_parser())
            return None

    def _set_adj_matrix(self):
        num_n = len(self.nodes)
        n_names = sorted(list(self.nodes.keys()))
        self.adj_matrix = pd.DataFrame(
            np.zeros(shape=(num_n, num_n)),
            index=n_names, columns=n_names
        )
        for n in self.nodes:
            self.parent_sets[n] = [edge.split('->')[0] for edge in self.edges if edge.split('->')[1] == n]
            self.adj_matrix.loc[self.parent_sets[n], n] = 1

            if self.node_types[n] == 'data' and len(self.parent_sets[n]) != 0:
                raise ValueError('node {} is DataNode but has parents in graph'.format(n))

    def _single_sample_round(self, node_name: str, data: pd.DataFrame, sampled_errors: pd.DataFrame):
        # transform parents by edges
        inputs = pd.DataFrame({
            parent: self.edges['{}->{}'.format(parent, node_name)].map(array=data[parent].values)
            for parent in self.parent_sets[node_name]
        })
        # calculate node
        parents = sorted(list(self.adj_matrix[self.adj_matrix[node_name] == 1].index))

        return self.nodes[node_name].calculate(inputs, parents, sampled_errors[node_name])

    def _single_det_round(self, node_name: str, data: pd.DataFrame):
        # transform parents by edges
        inputs = pd.DataFrame({
            parent: self.edges['{}->{}'.format(parent, node_name)].map(array=data[parent].values)
            for parent in self.parent_sets[node_name]
        })
        # calculate node
        parents = sorted(list(self.adj_matrix[self.adj_matrix[node_name] == 1].index))

        return self.nodes[node_name].calculate(inputs, parents)

    def _single_const_round(self, node_name: str, size_: int):
        return self.nodes[node_name].calculate(size_)

    def _single_data_round(self, node_name: str, sampled_errors: pd.DataFrame):
        return self.nodes[node_name].calculate(sampled_errors=sampled_errors[node_name])

    def _calc_non_interventions(self, node_name: str, data: pd.DataFrame, sampled_errors: pd.DataFrame):
        if self.node_types[node_name] == 'stochastic':
            return self._single_sample_round(
                node_name=node_name, data=data, sampled_errors=sampled_errors
            )
        elif self.node_types[node_name] == 'deterministic':
            return self._single_det_round(node_name=node_name, data=data)
        elif self.node_types[node_name] == 'constant':
            return self._single_const_round(node_name=node_name, size_=len(sampled_errors))
        elif self.node_types[node_name] == 'data':
            return self._single_data_round(node_name=node_name, sampled_errors=sampled_errors)
        else:
            raise TypeError

    def _get_errors(self, use_sampled_errors: bool, size: Optional[int], sampled_errors: Optional[pd.DataFrame],
                    full_data: bool = False):
        data_nodes = [n for n in self.node_types if self.node_types[n] == 'data']
        if full_data:
            parcs_assert(
                len(data_nodes) > 0,
                GraphError,
                'full_data option works only if graph has data node'
            )
            parcs_assert(
                size is None,
                GraphError,
                'with full_data option, size parameter must not be given'
            )
            # read size
            size = self.nodes[data_nodes[0]].get_info()['size']
        if not use_sampled_errors:
            parcs_assert(
                size is not None,
                GraphError,
                'either specify `size` or reuse sampled errors'
            )
            # sample new errors
            sampled_errors = pd.DataFrame(
                np.random.uniform(size=(size, len(self.adj_matrix))),
                columns=self.adj_matrix.columns
            )
            # if full data: return linspace index for all data nodes
            if full_data:
                sampled_errors[data_nodes[0]] = np.linspace(0, 1, size)
            # unify the data nodes.
            for i in range(1, len(data_nodes)):
                sampled_errors[data_nodes[i]] = sampled_errors[data_nodes[0]].values
        else:  # use sampled error
            parcs_assert(
                size is None,
                GraphError,
                'size must not be given when reusing errors'
            )
            # check if data nodes are unified
            for i in range(1, len(data_nodes)):
                parcs_assert(
                    all(sampled_errors[data_nodes[i]].values == sampled_errors[data_nodes[0]]),
                    DistributionError,
                    'sampled errors for data nodes are inconsistent.'
                )
        return sampled_errors

    def sample(self, size: Optional[int] = None, cache_name: Optional[str] = None,
               sampled_errors: Optional[pd.DataFrame] = None,
               return_errors: bool = False, cache_sampling: bool = False, use_sampled_errors: bool = False,
               full_data: bool = False):
        """**Sample from observational distribution**

        This method samples from the distribution that is modeled by the graph (with no intervention)
        this method reads either the `size` or `sampled_errors` which then set the size to length of sampled errors.
        To read more about sampling procedure and the meaning of error terms, :ref:`see here <sampling_error_terms>`

        Parameters
        ----------
        size : int
            number of samples. If ``use_sampled_errors=False`` then this method is used.
            Otherwise, this argument will be ignored.
            If the graph contains at least one data node, size will be ignored
        return_errors : bool, default=False
            If ``True`` then method returns ``samples, errors``.
        cache_sampling : bool, default=True
            If ``True``, then the ``cache_name`` must be given.
            The samples will be cached then in the attribute ``cache``
        cache_name : str
            If ``cache_sampling=True`` then this argument is necessary.
            The cached results are stored in ``Graph.cache[cache_name]``
        use_sampled_errors : bool, default=False
            If ``True``, the ``sampled_errors`` arg must be given. The ``size`` argument will be ignored.
        sampled_errors : pd.DataFrame
            The result of other sample calls with ``return_errors=True``.
        full_data : bool, default=False
            Is read only if at least one data node exists in the Graph. In this case, `True` means the sample size
            is equal to the length of the data node csv file, and all rows of the file are returned (only once).

        Returns
        -------
        samples : pd.DataFrame
            If ``return_errors=False``. The return will be a pandas data frame with columns equal to node names
            while each row is a sample
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. both objects are pandas data frames and similar to the previous item.

        Raises
        ------
        ValueError
            if sample size is bigger than the length of data in a data node and `with_replacement=False`.
        """
        data = pd.DataFrame([])

        # errors
        sampled_errors = self._get_errors(
            use_sampled_errors=use_sampled_errors, size=size, sampled_errors=sampled_errors,
            full_data=full_data
        )

        for node_name in topological_sort(self.adj_matrix):
            data[node_name] = self._calc_non_interventions(
                node_name=node_name, data=data, sampled_errors=sampled_errors
            )

        data.drop(self.dummy_names, axis=1, inplace=True)
        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data

    def do(self, size: int, interventions: dict, cache_name: Optional[str] = None, use_sampled_errors: bool = False,
           sampled_errors: Optional[pd.DataFrame] = None, return_errors: bool = False, cache_sampling: bool = False):
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s) on the main graph.
        This method accepts fixed interventions (see the ``interventions`` parameter).

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors, return_errors, cache_sampling, cache_name
            see :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        interventions : dict
            dictionary of interventions in the form of ``{'<node_to_be_intervened>': <intervention_value>}``


        Returns
        -------
        samples : pd.DataFrame
            If ``return_errors=False``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        """
        for i in interventions:
            assert i not in self.dummy_names, 'cannot intervene on dummy node {}'.format(i)
        data = pd.DataFrame([])
        sampled_errors = self._get_errors(
            use_sampled_errors=use_sampled_errors, size=size, sampled_errors=sampled_errors
        )

        for node_name in topological_sort(self.adj_matrix):
            if node_name not in interventions:
                array = self._calc_non_interventions(
                    node_name=node_name, data=data, sampled_errors=sampled_errors
                )
            else:
                array = np.ones(shape=(size,)) * interventions[node_name]
            data[node_name] = array

        data.drop(self.dummy_names, axis=1, inplace=True)
        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data

    def do_functional(self, size: int, intervene_on: str, inputs: List[str], func: Callable,
                      use_sampled_errors: bool = False, sampled_errors: Optional[pd.DataFrame] = None,
                      return_errors: bool = False, cache_sampling: bool = False, cache_name: Optional[str] = None):
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s) on the main graph.
        This method accepts interventions defined by a function on a subset of nodes.
        (see the ``intervene_on, inputs, func`` parameters).

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors, return_errors, cache_sampling, cache_name
            see :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        intervene_on : str
            name of the node subjected to intervention
        inputs : list of str
            list of node names as inputs of the functional intervention. the function reads the inputs in the same order
            as defined in the list
        func : callable
            A callable function with the same arguments as the input list, returning a value.

        Returns
        -------
        samples : pd.DataFrame
            If ``return_errors=False``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        """
        assert intervene_on not in self.dummy_names, 'cannot intervene on dummy node {}'.format(intervene_on)
        data = pd.DataFrame([])
        sampled_errors = self._get_errors(
            use_sampled_errors=use_sampled_errors, size=size, sampled_errors=sampled_errors
        )

        # TODO: we need to make sure z_list doesn't have items from descendant
        # solution: do all sampling first, then sample again the node with intrv, and the downstreams too.
        for node_name in topological_sort(self.adj_matrix):
            if node_name != intervene_on:
                array = self._calc_non_interventions(
                    node_name=node_name, data=data, sampled_errors=sampled_errors
                )
            else:
                # TODO: warn if z is child of X_0
                assert True
                array = data[inputs].apply(lambda x: func(*x.values), axis=1)
            data[node_name] = array

        data.drop(self.dummy_names, axis=1, inplace=True)
        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data

    def do_self(self, size: int, func: Callable, intervene_on: str,
                use_sampled_errors: bool = False, sampled_errors: Optional[pd.DataFrame] = None,
                return_errors: bool = False,
                cache_sampling: bool = False, cache_name: Optional[str] = None):
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s) on the main graph.
        This method accepts interventions defined as intervening on a node based on its observed value.

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors, return_errors, cache_sampling, cache_name
            see :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        intervene_on : str
            name of the node subjected to intervention
        func : callable
            A callable function with 1 argument, returning a transformed value.

        Returns
        -------
        samples : pd.DataFrame
            If ``return_errors=False``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        """
        assert intervene_on not in self.dummy_names, 'cannot intervene on dummy node {}'.format(intervene_on)
        data = pd.DataFrame([])
        sampled_errors = self._get_errors(
            use_sampled_errors=use_sampled_errors, size=size, sampled_errors=sampled_errors
        )

        for node_name in topological_sort(self.adj_matrix):
            data[node_name] = self._calc_non_interventions(
                node_name=node_name, data=data, sampled_errors=sampled_errors
            )
            if intervene_on == node_name:
                data[node_name] = data[node_name].apply(func)

        data.drop(self.dummy_names, axis=1, inplace=True)
        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data
