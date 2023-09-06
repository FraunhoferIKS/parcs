#  Copyright (c) 2023. Fraunhofer-Gesellschaft zur Foerderung der angewandten Forschung e.V.
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
from functools import wraps
from typing import Optional, List, Callable, Tuple
from typeguard import typechecked
from functools import wraps
import numpy as np
from pandas import Series, DataFrame
import warnings
from pyparcs.core.description import Description
from pyparcs.api.graph_objects import Edge, NODE_OBJECT_DICT
from pyparcs.api.utils import topological_sort
from pyparcs.core.exceptions import (parcs_assert, GraphError, DistributionError,
                                     InterventionError, DescriptionError)


def drop_dummy_nodes(func):
    """drop dummy nodes from the sampled datasets in Graph"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        data, error = func(self, *args, **kwargs)
        data.drop([node for node in self.nodes if self.desc.is_dummy(node)],
                  axis=1, inplace=True)
        return data, error

    return wrapper


@typechecked
class Graph:
    """**Causal DAG class**

    This class creates a causal DAG for a description. A graph is equipped with
    sampling and intervention methods. For a comprehensive tutorial, see the :ref:`get started
    <create_the_first_graph>` doc.

    Parameters
    ----------
    desc: Description
        the PARCS description object
    warning_level: str
        warning level for the checking of support compliance

    Attributes
    ----------
    nodes, edges : dicts
        dict of Graph nodes and edges, in the form of `{'node_name': Node()}` and
        `{'edge_name': Edge()}`
    """

    def __init__(self, desc: Description, warning_level: str = 'error'):
        # unwrap and create info for the graph
        self.desc = desc.unwrap()

        # create node objects
        self.nodes = {
            node: NODE_OBJECT_DICT[self.desc.node_types[node]](**specs)
            for (node, specs) in self.desc.nodes.items()
        }
        self.edges = {edge: Edge(**specs) for (edge, specs) in self.desc.edges.items()}

        if self.desc.has_correction:
            # burn out samples to tune correction
            self.sample(size=500)

        self.warning_level = warning_level

    def _check_support_compliance(self, node_name: str, values: Series):
        """**check each node's compliance with its distribution support after intervention**
        This method checks for each node whether the observed value (including intervened value)
        complies with its distribution support

        Parameters
        ----------
        node_name: str
            the name of the node for which compliance must be checked
        values: pandas Series
            the values of the node, under compliance check

        Raises
        ------
        UserWarning
            if self.warning_level == 'warning'
        InterventionError
            if self.warning_level == 'error'
        """
        try:
            assert self.nodes[node_name].output_distribution.validate_support(values)
        except AssertionError as exc:
            msg = (f"the intervention on {node_name} doesn't comply with the support of the "
                   f"output distribution {self.nodes[node_name].info['output_distribution']}")
            if self.warning_level == 'warning':
                warnings.warn(msg)
            elif self.warning_level == 'error':
                raise InterventionError(msg) from exc
            else:
                pass

    def _get_node_inputs(self, node_name, data):
        return DataFrame({
            parent: self.edges[f'{parent}->{node_name}'].map(array=data[parent].values)
            for parent in self.desc.parents_list[node_name]
        })

    def _get_errors(self,
                    use_sampled_errors: bool,
                    size: Optional[int],
                    sampled_errors: Optional[DataFrame],
                    full_data: bool = False):
        parcs_assert(size != 0,
                     GraphError,
                     'size cannot be zero.')
        is_size = False if size is None else True
        # check the correctness of size, use_sampled_errors and full_data parameters
        parcs_assert(is_size + full_data + use_sampled_errors == 1,
                     GraphError,
                     'invalid size, use_sample_errors, full_data combination: at least, '
                     'one and not more than one should be given. given parameters are:'
                     f'size={size}, use_sample_errors={use_sampled_errors}, full_data={full_data}')

        data_nodes = [n for (n, type_) in self.desc.node_types.items() if type_ == 'data']
        if full_data:
            parcs_assert(len(data_nodes) > 0,
                         GraphError,
                         'full_data option is relevant only if graph has at least one data node')
            parcs_assert(size is None,
                         GraphError,
                         'with full_data option, size parameter must not be given')
            # read size
            size = self.nodes[data_nodes[0]].get_info()['size']

        if not use_sampled_errors:
            # sample new errors
            sampled_errors = DataFrame(
                np.random.uniform(size=(size, len(self.desc.sorted_node_list))),
                columns=self.desc.sorted_node_list
            )
            # if full data: return linspace index for all data nodes
            if full_data:
                sampled_errors[data_nodes[0]] = np.linspace(0, 1, size)
            # unify the data nodes.
            for i in range(1, len(data_nodes)):
                sampled_errors[data_nodes[i]] = sampled_errors[data_nodes[0]].values
        else:
            # check if data nodes are unified
            for i in range(1, len(data_nodes)):
                parcs_assert(
                    all(sampled_errors[data_nodes[i]].values == sampled_errors[data_nodes[0]]),
                    DistributionError,
                    'sampled errors for data nodes are inconsistent.'
                )
        return sampled_errors

    @drop_dummy_nodes
    def sample(self,
               size: Optional[int] = None,
               use_sampled_errors: bool = False,
               sampled_errors: Optional[DataFrame] = None,
               full_data: bool = False) -> Tuple[DataFrame, DataFrame]:
        """**Sample from observational distribution**

        This method samples from the distribution that is modeled by the graph
        (with no intervention) this method reads either the `size` or `sampled_errors`
        which then set the size to length of sampled errors.

        Parameters
        ----------
        size : int
            number of samples. If ``use_sampled_errors=False`` then this method is used.
            Otherwise, this argument will be ignored.
            If the graph contains at least one data node, size will be ignored
        use_sampled_errors : bool, default=False
            If ``True``, the ``sampled_errors`` arg must be given. The ``size`` argument
            will be ignored.
        sampled_errors : DataFrame
            The result of other sample calls with ``return_errors=True``.
        full_data : bool, default=False
            Is read only if at least one data node exists in the Graph. In this case, `True`
            means the sample size
            is equal to the length of the data node csv file, and all rows of the file are returned
            (only once).

        Returns
        -------
        samples, errors : DataFrame

        Raises
        ------
        ValueError
            if sample size is bigger than the length of data in a data node and
            `with_replacement=False`.
        """
        data = DataFrame([])

        # errors
        sampled_errors = self._get_errors(use_sampled_errors=use_sampled_errors,
                                          sampled_errors=sampled_errors,
                                          full_data=full_data,
                                          size=size)

        for node_name in self.desc.topo_sort:
            data[node_name] = self.nodes[node_name].calculate(
                data=self._get_node_inputs(node_name, data),
                parents=self.desc.parents_list[node_name],
                errors=sampled_errors[node_name],
                size_=len(sampled_errors)
            )

        return data, sampled_errors

    @drop_dummy_nodes
    def do(self,
           size: int, interventions: dict, use_sampled_errors: bool = False,
           sampled_errors: Optional[DataFrame] = None) -> Tuple[DataFrame, DataFrame]:
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s)
        on the main graph.
        This method accepts fixed interventions (see the ``interventions`` parameter).

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors
            see :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        interventions : dict
            dictionary of interventions in the form of
            ``{'<node_to_be_intervened>': <intervention_value>}``


        Returns
        -------
        samples, errors : DataFrame
            If ``return_errors=True``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        """
        parcs_assert(all(not self.desc.is_dummy(node) for node in interventions),
                     GraphError,
                     'intervention not allowed on dummy nodes.')

        data = DataFrame([])
        sampled_errors = self._get_errors(
            use_sampled_errors=use_sampled_errors, size=size, sampled_errors=sampled_errors
        )

        for node_name in self.desc.topo_sort:
            if node_name in interventions:
                array = Series(np.ones(shape=(size,)) * interventions[node_name])
                if self.desc.node_types[node_name] == 'stochastic':
                    self._check_support_compliance(node_name, array)
            else:
                array = self.nodes[node_name].calculate(
                    data=self._get_node_inputs(node_name, data),
                    parents=self.desc.parents_list[node_name],
                    errors=sampled_errors[node_name],
                    size_=len(sampled_errors)
                )
            data[node_name] = array

        return data, sampled_errors

    @drop_dummy_nodes
    def do_functional(self, size: int, intervene_on: str, inputs: List[str], func: Callable,
                      use_sampled_errors: bool = False,
                      sampled_errors: Optional[DataFrame] = None
                      ) -> Tuple[DataFrame, DataFrame]:
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s)
        on the main graph.
        This method accepts interventions defined by a function on a subset of nodes.
        (see the ``intervene_on, inputs, func`` parameters).

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors
            see :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        intervene_on : str
            name of the node subjected to intervention
        inputs : list of str
            list of node names as inputs of the functional intervention.
            the function reads the inputs in the same order as defined in the list
        func : callable
            A callable function with the same arguments as the input list, returning a value.

        Returns
        -------
        samples, errors : DataFrame
        """
        parcs_assert(not self.desc.is_dummy(intervene_on),
                     GraphError,
                     'intervention not allowed on dummy nodes.')
        data = DataFrame([])
        sampled_errors = self._get_errors(
            use_sampled_errors=use_sampled_errors, size=size, sampled_errors=sampled_errors
        )

        # obtain the new order of nodes by modifying the edges
        new_adj = self.desc.adj_matrix.copy()
        # delete previous parents of the intervened-on
        new_adj.loc[:, intervene_on] = 0
        # add new parents of the intervened-on
        new_adj.loc[inputs, intervene_on] = 1
        # get new order
        try:
            new_topo_sort = topological_sort(new_adj)
        except DescriptionError as exc:
            raise InterventionError('do_functional creates loops in the graph. '
                                    'new parents of the not cannot be the descendants '
                                    'of the node that you intervene on.') from exc
        for node_name in new_topo_sort:
            if node_name == intervene_on:
                array = data[inputs].apply(lambda x: func(*x.values), axis=1)
                if self.desc.node_types[node_name] == 'stochastic':
                    self._check_support_compliance(node_name, array)
            else:
                array = self.nodes[node_name].calculate(
                    data=self._get_node_inputs(node_name, data),
                    parents=self.desc.parents_list[node_name],
                    errors=sampled_errors[node_name],
                    size_=len(sampled_errors)
                )

            data[node_name] = array

        return data, sampled_errors

    @drop_dummy_nodes
    def do_self(self, func: Callable, intervene_on: str, size: Optional[int] = None,
                use_sampled_errors: bool = False,
                sampled_errors: Optional[DataFrame] = None) -> Tuple[DataFrame, DataFrame]:
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s)
        on the main graph. This method accepts interventions defined as intervening on a node
        based on its observed value.

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors
            see :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        intervene_on : str
            name of the node subjected to intervention
        func : callable
            A callable function with 1 argument, returning a transformed value.

        Returns
        -------
        samples : DataFrame
            If ``return_errors=False``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        samples, errors : DataFrame, DataFrame
            If ``return_errors=True``. See :func:`~pyparcs.cdag.graph_objects.Graph.sample`
        """
        parcs_assert(not self.desc.is_dummy(intervene_on),
                     GraphError,
                     'intervention not allowed on dummy nodes.')
        data = DataFrame([])
        sampled_errors = self._get_errors(
            use_sampled_errors=use_sampled_errors, size=size, sampled_errors=sampled_errors
        )

        for node_name in self.desc.topo_sort:
            data[node_name] = self.nodes[node_name].calculate(
                data=self._get_node_inputs(node_name, data),
                parents=self.desc.parents_list[node_name],
                errors=sampled_errors[node_name],
                size_=len(sampled_errors)
            )
            if intervene_on == node_name:
                data[node_name] = data[node_name].apply(func)
                if self.desc.node_types[node_name] == 'stochastic':
                    self._check_support_compliance(node_name, data[node_name])

        return data, sampled_errors
