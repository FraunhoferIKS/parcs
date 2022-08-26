import numpy as np
import pandas as pd
from itertools import product
from rad_sim.scm import mapping_functions
from rad_sim.scm.utils import topological_sort, EdgeCorrection
from rad_sim.scm.output_distributions import GaussianDistribution, BernoulliDistribution


OUTPUT_DISTRIBUTIONS = {
    'gaussian': GaussianDistribution,
    'bernoulli': BernoulliDistribution
}

EDGE_FUNCTIONS = {
    'identity': mapping_functions.edge_binary_identity,
    'sigmoid': mapping_functions.edge_sigmoid,
    'gaussian_rbf': mapping_functions.edge_gaussian_rbf
}


class Node:
    def __init__(self,
                 name=None,
                 parents=None,
                 output_distribution=None,
                 do_correction=True,
                 dist_configs={},
                 dist_params_coefs=None):
        """ **Node object in causal DAGs**

        A node is defined by its names and parents, and an output distribution along with its parameters.
        After being set, the node can be sampled by fetching data (``pd.DataFrame``) to the ``.sample()`` method.
        if the node is source in graph (parents are not in columns of data), then ``size`` is used to sample with
        random distribution parameters

        Parameters
        ----------
        name : str, optional
            name of the node. Optional unless the node is used in a graph
        parents : list[str]
            name of the node's parents, to be searched in data header
        output_distribution : str
            Selected from the available distributions. call ``graph_objects.OUTPUT_DISTRIBUTIONS`` to see the list.
        do_correction : bool, default=True
            **tbc**
        dist_configs : dict
            **tbc**
        dist_params_coefs : dict
            **tbc**


        """
        # basic attributes
        self.info = {
            'name': name,
            'output_distribution': output_distribution,
            'parents': parents
        }
        self.output_distribution = OUTPUT_DISTRIBUTIONS[output_distribution](
            **dist_configs,
            do_correction=do_correction,
            coefs=dist_params_coefs
        )

    def sample(self, data, size):
        return self.output_distribution.calculate_output(
            data[self.info['parents']].values,
            size
        )


class Edge:
    """ **Edge object in causal DAGs**

    An edge is defined by its child-parent nodes, and the edge function (the node names are only for clarification.
    they are not used in edge methods).
    It receives the data from parent node, and maps it based on the edge function.
    If ``do_correction = True`` Then batch normalization parameters are set upon the next data batch, and be used
    in further transformations.

    Parameters
    ----------
    parent : str, optional
        name of the parent node. Optional if edge is not in a graph
    child : str, optional
        name of the child node. Optional if edge is not in a graph
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
        `name`, `function` and `params` of the edge function

    Examples
    --------
    >>> import numpy as np
    >>> from rad_sim.scm.graph_objects import Edge
    >>> # a standard Sigmoid activation
    >>> edge = Edge(
    ...     function_name='sigmoid',
    ...     function_params={}
    ... )
    >>> x = np.array([-10, -1, 0, 1, 10])
    >>> x_mapped = np.round( edge.map(x), 2)
    >>> x_mapped
    array([0.  , 0.12, 0.5 , 0.88, 1.  ])
    """
    def __init__(self,
                 parent=None,
                 child=None,
                 do_correction=True,
                 function_name=None,
                 function_params=None):
        self.parent = parent
        self.child = child
        self.do_correction = do_correction
        if self.do_correction:
            self.corrector = EdgeCorrection()

        self.edge_function = {
            'name': function_name,
            'function': EDGE_FUNCTIONS[function_name],
            'params': function_params
        }

    def map(self, array=None):
        """ **maps an input array**

        This method maps a given input array based on set `edge_function` and `do_correction` values.

        Parameters
        ----------
        array : 1D numpy array, optional
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


class BaseGraph:
    def __init__(self):
        self.nodes = {}
        self.edges = {}
        self.adj_matrix = pd.DataFrame([])
        self.data = {}

    def set_nodes(self, nodes_list=None):
        self.nodes = {
            item['name']: Node(
                name=item['name'], parents=item['parents']
            ).set_state_function(
                function_name=item['state_function']
            ).set_output_function(
                function_name=item['output_function']
            ).set_state_params(
                params=item['state_params']
            ).set_output_params(
                params=item['output_params']
            ) for item in nodes_list
        }
        return self

    def set_edges(self, adj_matrix: pd.DataFrame = None, function_specs: dict = None):
        self.adj_matrix = adj_matrix
        for node_pair in product(adj_matrix.index, adj_matrix.columns):
            try:
                info = adj_matrix.loc[node_pair[0], node_pair[1]]
                edge_symbol = '{} -> {}'.format(node_pair[0], node_pair[1])
                assert info != 0
                self.edges[edge_symbol] = Edge(
                    parent=node_pair[0], child=node_pair[1]
                ).set_function(
                    function_name=function_specs[edge_symbol]['function_name']
                ).set_function_params(
                    function_params=function_specs[edge_symbol]['function_params']
                )
            except AssertionError:
                continue
        return self

    def sample(self, size=None):
        assert size is not None, 'Specify size for sample'
        for node in topological_sort(adj_matrix=self.adj_matrix):
            v = self.nodes[node]
            inputs = pd.DataFrame({
                p: self.edges['{} -> {}'.format(p, v.name)].map(
                    array=self.nodes[p].value['output']
                ) for p in v.parents
            })
            v.calc_state(inputs=inputs, size=size)
            v.calc_output()
        self.data = pd.DataFrame({v: self.nodes[v].value['output'] for v in self.nodes})
        return self.data
