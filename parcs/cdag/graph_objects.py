import numpy as np
import pandas as pd
from parcs.cdag import mapping_functions
from parcs.cdag.utils import topological_sort, EdgeCorrection
from parcs.cdag.output_distributions import OUTPUT_DISTRIBUTIONS


class Node:
    """ **Node object in causal DAGs**
            Use this class to create a node, independent of any graphs.
            If you want to construct a causal DAG, please use the ``parcs.cdag.graph_objects.BaseGraph`` class instead.
        Node can be sampled by passing the data (``pd.DataFrame``) to the ``.sample()`` method.
            if the node is source in graph (parents are not in columns of data), then ``size`` is used to sample with
            random distribution parameters

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

            Examples
            --------


            """
    def __init__(self,
                 name=None,
                 output_distribution=None,
                 do_correction=False,
                 correction_config=None,
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
        """ **samples the node**

        If data is empty, the ``size`` is used to sample according to coefficient vector *biases*.
        If data is non-empty, then ``size`` is ignored

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


class Edge:
    """ **Edge object in causal DAGs**

    Use this class to create an edge, independent of any graphs.
    If you want to construct a causal DAG, please use the ``parcs.cdag.graph_objects.BaseGraph`` class instead.
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
    def __init__(self,
                 nodes=None,
                 edges=None):

        self.nodes = {kwargs['name']: Node(**kwargs) for kwargs in nodes}
        self.edges = {kwargs['name']: Edge(**kwargs) for kwargs in edges}
        self.parent_sets = {}
        self.adj_matrix = None
        self._set_adj_matrix()
        self.cache = {}

    def _set_adj_matrix(self):
        num_n = len(self.nodes)
        n_names = self.nodes.keys()
        self.adj_matrix = pd.DataFrame(
            np.zeros(shape=(num_n, num_n)),
            index=n_names, columns=n_names
        )
        for n in self.nodes:
            self.parent_sets[n] = [edge.split('->')[0] for edge in self.edges if edge.split('->')[1] == n]
            self.adj_matrix.loc[self.parent_sets[n], n] = 1

    def sample(self, size=200, return_errors=False, cache_sampling=False, cache_name=None):
        data = pd.DataFrame([])
        sampled_errors = pd.DataFrame([])
        self._set_adj_matrix()

        for node_name in topological_sort(self.adj_matrix):
            # sample errors
            sampled_errors[node_name] = np.random.uniform(0, 1, size=size)
            # transform parents by edges
            inputs = pd.DataFrame({
                parent: self.edges['{}->{}'.format(parent, node_name)].map(array=data[parent].values)
                for parent in self.parent_sets[node_name]
            })
            # calculate node
            data[node_name] = self.nodes[node_name].calculate(inputs, sampled_errors[node_name])
        if cache_sampling:
            self.cache[cache_name] = (data, sampled_errors)
        if return_errors:
            return data, sampled_errors
        else:
            return data


if __name__ == '__main__':
    g = Graph(
        nodes=[
            {
                'name': 'x0', 'parents': [], 'output_distribution': 'gaussian',
                'dist_params_coefs': {
                    'mu_': {'bias': 0, 'linear': np.array([]), 'interactions': np.array([])},
                    'sigma_': {'bias': 1, 'linear': np.array([]), 'interactions': np.array([])}
                }
            },
            {
                'name': 'x1', 'parents': ['x0'], 'output_distribution': 'gaussian',
                'dist_params_coefs': {
                    'mu_': {'bias': 0, 'linear': np.array([1]), 'interactions': np.array([])},
                    'sigma_': {'bias': 1, 'linear': np.array([0]), 'interactions': np.array([])}
                }
            },
            {
                'name': 'x2', 'parents': ['x0', 'x1'], 'output_distribution': 'gaussian',
                'dist_params_coefs': {
                    'mu_': {'bias': 0, 'linear': np.array([1, 1]), 'interactions': np.array([0])},
                    'sigma_': {'bias': 1, 'linear': np.array([0, 0]), 'interactions': np.array([0])}
                }
            },
        ],
        edges=[
            {'parent': 'x0', 'child': 'x1', 'function_name': 'identity', 'function_params': {}},
            {'parent': 'x0', 'child': 'x2', 'function_name': 'identity', 'function_params': {}},
            {'parent': 'x1', 'child': 'x2', 'function_name': 'identity', 'function_params': {}}
        ]
    )
    data, errors = g.sample(size=500, cache_sampling=True, cache_name='exp', return_errors=True)
    from matplotlib import pyplot as plt
    plt.scatter(data['x0'], data['x1'], c=data['x2'])
    plt.show()
