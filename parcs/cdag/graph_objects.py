import numpy as np
import pandas as pd
from parcs.cdag import mapping_functions
from parcs.cdag.utils import topological_sort, EdgeCorrection
from parcs.cdag.output_distributions import OUTPUT_DISTRIBUTIONS


class Node:
    """ **Node object in causal DAGs**

    Use this class to create a node, independent of any graphs.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph` class instead.
    Node can be sampled by passing the data and error terms (``pd.DataFrame``) to the ``.calculate()`` method.

    Parameters
    ----------
    name : str, optional
        name of the node. Optional unless the node is used in a graph
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
                 name=None,
                 output_distribution=None,
                 do_correction=False,
                 correction_config={},
                 dist_params_coefs=None):
        # basic attributes
        self.info = {
            'name': name,
            'output_distribution': output_distribution
        }
        self.output_distribution = OUTPUT_DISTRIBUTIONS[output_distribution](
            correction_config=correction_config,
            do_correction=do_correction,
            coefs=dist_params_coefs
        )

    def calculate(self, data, parents, errors):
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


class DetNode:
    """ **Deterministic Node object in causal DAGs**

    Use this class to create a node, independent of any graphs.
    If you want to construct a causal DAG, please use the :func:`~parcs.cdag.graph_objects.Graph` class instead.
    Deterministic node yields the output by calling the `.calculate()` method.

    Parameters
    ----------
    name : str, optional
        name of the node. Optional unless the node is used in a graph
    function : callable
        the user-defined function that represents the node.

    Examples
    --------
    >>> from parcs.cdag.graph_objects import DetNode
    >>> import pandas as pd
    >>> n_0 = DetNode(name='N_0', func=lambda a,b: a+b)
    >>> data = pd.DataFrame([[1, 2], [2, 2], [3, 3]], columns=('N_1', 'N_2'))
    >>> n_0.calculate(data, ['N_1', 'N_2'])
    array([3, 4, 6])
    """
    def __init__(self,
                 name=None,
                 function=None):
        self.info = {'name': name}
        self.function = function

    def calculate(self, data, parents):
        return np.array([
            self.function(r[0], r[1]) for r in data[parents].values
        ])


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
    >>> import numpy as np
    >>> from parcs.cdag.graph_objects import Edge
    >>> # a standard Sigmoid activation
    >>> edge = Edge(
    ...     function_name='sigmoid',
    ...     function_params={},
    ...     do_correction=False
    ... )
    >>> x = np.array([-10, -1, 0, 1, 10])
    >>> x_mapped = np.round( edge.map(x), 2)
    >>> x_mapped
    array([0.  , 0.12, 0.5 , 0.88, 1.  ])
    """
    def __init__(self,
                 name=None,
                 do_correction=False,
                 function_name=None,
                 function_params=None):
        self.name = name
        self.parent, self.child = self.name.split('->')
        self.do_correction = do_correction
        if self.do_correction:
            self.corrector = EdgeCorrection()

        self.edge_function = {
            'name': function_name,
            'function': mapping_functions.EDGE_FUNCTIONS[function_name],
            'params': function_params
        }

    def map(self, array=None):
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


class Graph:
    """**Causal DAG class**

    This class creates a causal DAG for the specified nodes and edges. A graph is equipped with sampling
    and intervention methods. For a comprehensive tutorial, see the :ref:`get started <create_the_first_graph>` doc.

    Parameters
    ----------
    nodes : list of dicts
        List of dictionaries whose keys are kwargs of the :func:`~parcs.cdag.graph_objects.Node` object.
    edges : list of dicts
        List of dictionaries whose keys are kwargs of the :func:`~parcs.cdag.graph_objects.Edge` object.

    Attributes
    ----------
    nodes, edges : dicts
        dict of Graph nodes and edges, in the form of `{'node_name': Node()}` and `{'edge_name': Edge()}`

    cache : tuple of samples and error terms
        If caching is enabled upon sampling, then the result is cached in this attribute
    """
    def __init__(self,
                 nodes=None,
                 edges=None):
        self.nodes = {
            kwargs['name']: Node(**kwargs) if 'output_distribution' in kwargs else DetNode(**kwargs)
            for kwargs in nodes
        }
        self.edges = {kwargs['name']: Edge(**kwargs) for kwargs in edges}
        self.parent_sets = {}
        self.adj_matrix = None
        self._set_adj_matrix()
        self.cache = {}

        # one-time sample to setup corrections
        # TODO: don't do it if no correction=True
        self.sample(size=500)

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

    def _single_sample_round(self, use_sampled_errors=False, node_name=None, size=None, data=None, sampled_errors=None):
        if not use_sampled_errors:
            # sample errors
            sampled_errors[node_name] = np.random.uniform(0, 1, size=size)
        # transform parents by edges
        inputs = pd.DataFrame({
            parent: self.edges['{}->{}'.format(parent, node_name)].map(array=data[parent].values)
            for parent in self.parent_sets[node_name]
        })
        # calculate node
        parents = sorted(list(self.adj_matrix[self.adj_matrix[node_name] == 1].index))

        return self.nodes[node_name].calculate(inputs, parents, sampled_errors[node_name])

    def _single_det_round(self, node_name, data):
        # transform parents by edges
        inputs = pd.DataFrame({
            parent: self.edges['{}->{}'.format(parent, node_name)].map(array=data[parent].values)
            for parent in self.parent_sets[node_name]
        })
        # calculate node
        parents = sorted(list(self.adj_matrix[self.adj_matrix[node_name] == 1].index))

        return self.nodes[node_name].calculate(inputs, parents)

    def _node_type(self, node_name):
        node = self.nodes[node_name]
        if isinstance(node, Node):
            return 'stoch'
        elif isinstance(node, DetNode):
            return 'det'
        else:
            return TypeError

    def _calc_non_interventions(self, node_name=None, data=None, size=None,
                                use_sampled_errors=None, sampled_errors=None):
        if self._node_type(node_name) == 'stoch':
            return self._single_sample_round(
                use_sampled_errors=use_sampled_errors, node_name=node_name, size=size, data=data,
                sampled_errors=sampled_errors
            )
        elif self._node_type(node_name) == 'det':
            return self._single_det_round(node_name=node_name, data=data)
        else:
            raise ValueError


    def sample(self, size=None, return_errors=False, cache_sampling=False, cache_name=None,
               use_sampled_errors=False, sampled_errors=None):
        """**Sample from observational distribution**

        This method samples from the distribution that is modeled by the graph (with no intervention)
        this method reads either the `size` or `sampled_errors` which then set the size to length of sampled errors.
        To read more about sampling procedure and the meaning of error terms, :ref:`see here <sampling_error_terms>`

        Parameters
        ----------
        size : int
            number of samples. If ``use_sampled_errors=False`` then this method is used.
            Otherwise, this argument will be ignored.
        return_errors : bool, default=False
            If ``True`` then method returns ``samples, errors``.
        cache_sampling : bool, default=True
            If ``True``, then the ``cache_name`` must be given. The samples will be cached then in the attribute ``cache``
        cache_name : str
            If ``cache_sampling=True`` then this argument is necessary.
            The cached results are stored in ``Graph.cache[cache_name]``
        use_sampled_errors : bool, default=False
            If ``True``, the ``sampled_errors`` arg must be given. The ``size`` argument will be ignored.
        sampled_errors : pd.DataFrame
            The result of other sample calls with ``return_errors=True``.

        Returns
        -------
        samples : pd.DataFrame
            If ``return_errors=False``. The return will be a pandas data frame with columns equal to node names
            while each row is a sample
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. both objects are pandas data frames and similar to the previous item.
        """
        # TODO: check if size is none and use_sample_errors=False: raise error
        data = pd.DataFrame([])
        self._set_adj_matrix()
        if not use_sampled_errors:
            sampled_errors = pd.DataFrame([])

        for node_name in topological_sort(self.adj_matrix):
            data[node_name] = self._calc_non_interventions(
                node_name=node_name, size=size, data=data,
                use_sampled_errors=use_sampled_errors, sampled_errors=sampled_errors
            )

        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data

    def do(self, size=None, interventions=None, use_sampled_errors=False, sampled_errors=None,
           return_errors=False, cache_sampling=False, cache_name=None):
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s) on the main graph.
        This method accepts fixed interventions (see the ``interventions`` parameter).

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors, return_errors, cache_sampling, cache_name
            see :func:`~parcs.cdag.graph_objects.Graph.sample`
        interventions : dict
            dictionary of interventions in the form of ``{'<node_to_be_intervened>': <intervention_value>}``

        Returns
        -------
        samples : pd.DataFrame
            If ``return_errors=False``. See :func:`~parcs.cdag.graph_objects.Graph.sample`
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. See :func:`~parcs.cdag.graph_objects.Graph.sample`
        """
        data = pd.DataFrame([])
        self._set_adj_matrix()
        if not use_sampled_errors:
            sampled_errors = pd.DataFrame([])

        for node_name in topological_sort(self.adj_matrix):
            if node_name not in interventions:
                array = self._calc_non_interventions(
                    node_name=node_name, size=size, data=data,
                    use_sampled_errors=use_sampled_errors, sampled_errors=sampled_errors
                )
            else:
                array = np.ones(shape=(size,)) * interventions[node_name]
            data[node_name] = array

        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data

    def do_functional(self, size=None, intervene_on=None, inputs=None, func=None,
                      use_sampled_errors=False, sampled_errors=None,
                      return_errors=False, cache_sampling=False, cache_name=None):
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s) on the main graph.
        This method accepts interventions defined by a function on a subset of nodes.
        (see the ``intervene_on, inputs, func`` parameters).

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors, return_errors, cache_sampling, cache_name
            see :func:`~parcs.cdag.graph_objects.Graph.sample`
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
            If ``return_errors=False``. See :func:`~parcs.cdag.graph_objects.Graph.sample`
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. See :func:`~parcs.cdag.graph_objects.Graph.sample`
        """
        data = pd.DataFrame([])
        self._set_adj_matrix()
        if not use_sampled_errors:
            sampled_errors = pd.DataFrame([])

        # TODO: we need to make sure z_list doesn't have items from descendant
        # solution: do all sampling first, then sample again the node with intrv, and the downstreams too.
        for node_name in topological_sort(self.adj_matrix):
            if node_name != intervene_on:
                array = self._calc_non_interventions(
                    node_name=node_name, size=size, data=data,
                    use_sampled_errors=use_sampled_errors, sampled_errors=sampled_errors
                )
            else:
                # TODO: warn if z is child of X_0
                assert True
                array = data[inputs].apply(lambda x: func(*x.values), axis=1)
            data[node_name] = array

        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data

    def do_self(self, size=None, func=None, intervene_on=None,
                use_sampled_errors=False, sampled_errors=None, return_errors=False,
                cache_sampling=False, cache_name=None):
        """**sample from interventional distribution**
        This methods sample from an interventional distribution which is modeled by intervention(s) on the main graph.
        This method accepts interventions defined as intervening on a node based on its observed value.

        Parameters
        ----------
        size, use_sampled_errors, sampled_errors, return_errors, cache_sampling, cache_name
            see :func:`~parcs.cdag.graph_objects.Graph.sample`
        intervene_on : str
            name of the node subjected to intervention
        func : callable
            A callable function with 1 argument, returning a transformed value.

        Returns
        -------
        samples : pd.DataFrame
            If ``return_errors=False``. See :func:`~parcs.cdag.graph_objects.Graph.sample`
        samples, errors : pd.DataFrame, pd.DataFrame
            If ``return_errors=True``. See :func:`~parcs.cdag.graph_objects.Graph.sample`
        """
        data = pd.DataFrame([])
        self._set_adj_matrix()
        if not use_sampled_errors:
            sampled_errors = pd.DataFrame([])

        for node_name in topological_sort(self.adj_matrix):
            data[node_name] = self._single_sample_round(
                use_sampled_errors=use_sampled_errors, node_name=node_name, size=size, data=data,
                sampled_errors=sampled_errors
            )
            if intervene_on == node_name:
                data[node_name] = data[node_name].apply(func)

        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data


def m_graph_convert(data: pd.DataFrame, missingness_prefix='R_', indicator_is_missed=0):
    len_prefix = len(missingness_prefix)
    # take Rs: it starts with prefix, and subtracting the prefix gives the name of another node
    r_columns = [
        i for i in data.columns if i[:len_prefix] == missingness_prefix and i[len_prefix:] in data.columns
    ]

    x_columns = set(data.columns) - set(r_columns)
    # masking
    for r in r_columns:
        x = r[len_prefix:]
        data[x][data[r]==indicator_is_missed] = np.nan

    return data[x_columns]