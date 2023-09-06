"""Providing the Description object for PARCS Graphs"""

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
from itertools import product
from functools import wraps
from pyparcs.api.parsers import (description_parser, is_partial,
                                 outline_splitter, augment_line, infer_missing_edges,
                                 stochastic_node_synthesizer, edge_synthesizer)
from pyparcs.api.temporal_parsers import temporal_outline_parser
from pyparcs.api.utils import get_adj_matrix, topological_sort, digest_outline_input
from pyparcs.core.exceptions import RandomizerError, DescriptionError, parcs_assert
from pyparcs.core.guideline import Guideline
from pyparcs.api.randomization import *


def _check_partial(func):
    """Decorator to check is_partial for the description after each method, if needed"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self.is_partial = is_partial(self.nodes, self.edges)
        return result

    return wrapper


def _update_outline(func):
    """Decorator to update the outline after the method randomize_parameters"""

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        result = func(self, *args, **kwargs)
        self._update_outline()
        return result

    return wrapper


def _validate_tag(tag_type, tag_order):
    """Decorator for checking the validity of the tags

    Parameters
    ----------
    tag_type: str
        What the tag must start with
    tag_order: int
        In case the user doesn't pass tag as a kwarg, what is the position
        of the tag parameter
    """

    def decorator(func):
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            try:
                if 'tag' in kwargs:
                    tag = kwargs['tag']
                else:
                    tag = args[tag_order]
                parcs_assert(tag is None or tag[0] == tag_type,
                             RandomizerError,
                             f'Error in tag {tag}. It must start with {tag_type}')
            except IndexError:
                pass
            finally:
                result = func(self, *args, **kwargs)
            return result

        return wrapper

    return decorator


@typechecked
class Description:
    """description objects for PARCS graphs

    Description object stores, parses and controls the user-given
    description of the simulation graph. It will then be passed
    to Graph objects to instantiate a graph.

    Parameters
    ----------
    outline : dict or path
        a dictionary or a yml file provided by a path which describes
        the nodes and edges of a graph
    infer_edges : bool, default = False
        If a node `A` appears in the parameters of a node `B` but
        there is no `A->B` edge, `infer_edges=True` creates an
        identity edge to complete the description. If you want to
        declare all the edges manually, set to `False` to catch
        unwanted skipping of the edges.

    Attributes
    ----------
    outline: dict
        The user-given outline
    nodes, edges: list of dict
        The list of nodes and edge info dicts which will be consumed
        by :class:`pyparcs.Graph` to instantiate a graph
    """

    def __init__(self,
                 outline: Union[dict, str],
                 infer_edges: bool = False):
        # == USER INPUTS ==
        # store the dict format of the given outlines
        self.outline = digest_outline_input(outline)
        # flag for if user needs inferring implicit edges
        self.infer_edges = infer_edges

        # == MAIN ATTRIBUTES ==
        self.nodes, self.edges = description_parser(self.outline,
                                                    infer_edges=self.infer_edges)
        self._setup_attributes()

    def _setup_attributes(self):
        """early setup of description attributes

        Setup is methodized because it happens other than at the beginning
        """
        self.node_types = {n: specs.pop('node_type') for (n, specs) in self.nodes.items()}
        self.node_tags = {n: specs.pop('tags') for (n, specs) in self.nodes.items()}
        self.edge_tags = {e: specs.pop('tags') for (e, specs) in self.edges.items()}
        self.has_correction = any(
            self.node_types[n] == 'stochastic' and
            self.nodes[n]['do_correction']
            for n in self.nodes
        ) or any(self.edges[e]['do_correction'] for e in self.edges)

        self.sorted_node_list = sorted(list(self.nodes.keys()))
        self.parents_list = {
            node: sorted([e.split('->')[0] for e in self.edges if e.split('->')[1] == node])
            for node in self.nodes
        }
        self.adj_matrix = get_adj_matrix(self.sorted_node_list, self.parents_list)
        self.is_partial = is_partial(self.nodes, self.edges)
        self.topo_sort = topological_sort(self.adj_matrix)

        parcs_assert(
            all(self.node_types[node] in ['deterministic', 'stochastic']
                or len(self.parents_list[node]) == 0
                for node in self.nodes),
            DescriptionError,
            "No node type other than deterministic and stochastic cannot have parents"
        )

        if self.infer_edges and self.is_partial:
            warnings.warn(
                "'infer_edges' is True for a partially-specified description: "
                "infer_edges work by checking if an edge exists, "
                "when a node is found in the description of the other node. "
                "this search procedure is a string search. Thus, no edge is "
                "inferred when a parameter is '?' or a function/distribution "
                "is 'random'."
            )

    def is_dummy(self, node):
        tags = self.node_tags[node]
        return any(tag == 'D' for tag in tags)

    def unwrap(self):
        parcs_assert(not self.is_partial,
                     DescriptionError,
                     "Partially-specified descriptions cannot build a PARCS Graph."
                     "Consider randomizing.")
        return self

    @_update_outline
    @_check_partial
    @_validate_tag('P', 1)
    @typechecked
    def randomize_parameters(self, guideline: Guideline, tag: Union[str, None] = None):
        """Parameter Randomizer for Graphs

        Takes incomplete node and edge lists and fills in the missing parts.
        It is done based on a passed guideline dict.

        Parameter randomizer can resolve '?' coefs and 'random' distributions/functions

        Parameters
        ----------
        guideline : Guideline
            Randomization guideline
        tag : str, optional
            A tag to be addressed by the current randomizer. If `None`, then addresses
            lines with no randomization tags.
            This tag must start with 'P'

        Returns
        -------
        self
        """
        # edges:
        for name, edge in self.edges.items():
            # skip if not eligible
            if not is_eligible(tag, self.edge_tags[name]):
                continue
            # randomize function
            randomize_edge_function(edge, guideline)
            # randomize parameters
            randomize_edge_function_parameters(edge, guideline)

        # nodes:
        for name, node in self.nodes.items():
            # skip if node is not stochastic, or the line is not eligible
            if ('output_distribution' not in node) or \
                    (not is_eligible(tag, self.node_tags[name])):
                continue
            # randomize distribution
            randomize_node_distribution(node, guideline)
            # randomize params
            randomize_node_distribution_parameters(node, self.parents_list[name], guideline)
        return self

    @_validate_tag('C', 3)
    def randomize_connection_to(self,
                                outline: Union[dict, str],
                                guideline: Guideline,
                                infer_edges: bool = False,
                                tag: Union[str, None] = None,
                                mask: Union[pd.DataFrame, None] = None,
                                apply_limit: bool = False):
        """Connect the description to a child outline

        This method connects the current description to another graph,
        outlined by the new outline.

        As the result of this randomization, new nodes and edges will be added
        to the current description.

        Parameters
        ----------
        outline: dict or Path
            The child graph that receives edge from the description nodes
        guideline: dict or Path
            The randomization guideline
        infer_edges: bool, default = False
            if to infer edge for the child outline
        tag: str or None, default = None
            starting with 'C'. If given, then only the tagged nodes are
            allowed to have an outgoing edge. If None, and some nodes
            are tagged with 'C' tags, then those tagged nodes won't
            have edges
        mask: pd.DataFrame or None, default = None
            A mask of the shape (num_parent_nodes, num_child_nodes) that is applied
            to the sampled connection edges. If None, no mask is applied.
        apply_limit: bool, default = False
            In the child outline, parameters can be marked with a "!" sign,
            and hence are blocked to have any incoming edge. By setting
            this argument to `True`, this blocking will take place

        Returns
        -------
        self
        """
        # initial checking of the child outline
        try:
            description_parser({k: v.replace('!', '') for k, v in outline.items()},
                               infer_edges=infer_edges)
        except Exception as exc:
            raise DescriptionError('child outline is invalid') from exc

        new_outline = outline.copy()
        # first, resolve the infer_edge locally for the child outline
        child_nodes, child_edges = outline_splitter(new_outline)
        child_edges = infer_missing_edges(child_nodes, child_edges)
        new_outline = {**child_nodes, **child_edges}

        parcs_assert(len(set(self.nodes) & set(child_nodes)) == 0,
                     DescriptionError,
                     'parent and child descriptions must not have common node names')
        # create random adjacency
        try:
            adj_matrix = random_connection_adj_matrix(
                self.sorted_node_list,
                child_nodes,
                guideline.sample_values('graph.density'),
                mask
            )
        except AssertionError as exc:
            raise DescriptionError('The index and columns of the mask must correspond to parent '
                                   'and child node names respectively.') from exc
        # apply the tag restriction
        for node, tags in self.node_tags.items():
            # if node has C tag when rand doesn't, or node doesn't have tag when rand does
            if (tag and tag not in tags) or (not tag and any(t[0] == 'C' for t in tags)):
                # cancel the outgoing edges
                adj_matrix.loc[node, :] = 0

        # create extra terms child by child
        for node, line in child_nodes.items():
            # prospective edges
            incoming = adj_matrix[adj_matrix[node] == 1].index.to_list()
            # added terms to the nodes
            addition = get_new_terms(incoming)
            # augment the line
            is_added, augmented_line = augment_line(line, addition, limit=apply_limit)
            new_outline[node] = augmented_line
            if is_added:
                # new edges
                new_outline = {**new_outline, **{f'{p}->{node}': 'random' for p in incoming}}
        # in order to parse the new outline, we need to put the parent names in the
        # outline temporarily
        new_outline.update({node: 'constant(1)' for node in self.nodes})
        # parse the new outline and add to the existing description
        new_nodes, new_edges = description_parser(new_outline,
                                                  infer_edges=False)
        # delete the temp nodes that where added before parsing
        for node in self.nodes:
            del new_nodes[node]
        self.nodes.update(new_nodes)
        self.edges.update(new_edges)

        # we popped tags and types for existing nodes and edges. we need to add them
        for node, tags in self.node_tags.items():
            self.nodes[node]['tags'] = tags
        for edge, tags in self.edge_tags.items():
            self.edges[edge]['tags'] = tags
        for node, node_type in self.node_types.items():
            self.nodes[node]['node_type'] = node_type

        # re-initiating the description with added nodes and edges
        self._setup_attributes()
        self.randomize_parameters(guideline)

        return self

    def _update_outline(self):
        """
        This function uses the description's attributes: nodes, edges, parent_list, node_tags and edge_tags
        to update another description's attribute: outline.
        """
        # create node dict
        for node_name, node_dict in self.nodes.items():
            if self.node_types[node_name] == 'stochastic':
                self.outline[node_name] = stochastic_node_synthesizer(
                    node_dict, self.parents_list[node_name], self.node_tags[node_name]
                )

        # create edge dict
        for edge_name, edge_dict in self.edges.items():
            self.outline[edge_name] = edge_synthesizer(edge_dict,
                                                       self.edge_tags[edge_name])


class RandomDescription(Description):
    """Spawns a random description from scratch

    This class gives a description file according to a defined guideline
    which describes the number of nodes, sparsity of edges, etc.

    Parameters
    ----------
    guideline : dict or path
            A standard randomization guideline dictionary
    """

    def __init__(self, guideline: Guideline, node_prefix: str = 'Z'):
        # BUILD THE SKELETON GRAPH
        # pick the nodes
        node_names = [f'{node_prefix}_{i}'
                      for i in range(guideline.sample_values('graph.num_nodes'))]

        node_outline = {node: 'random' for node in node_names}
        # pick the edges
        adj = random_adj_matrix(node_names,
                                density=guideline.sample_values('graph.density'))
        edge_outline = {f'{p}->{c}': 'random' for (p, c) in product(node_names, repeat=2)
                        if adj.loc[p, c] == 1}
        # setup the skeleton outline
        outline = {**node_outline, **edge_outline}
        super().__init__(outline)
        self.randomize_parameters(guideline)


class TemporalDescription(Description):
    """Temporal Description class

    This class provides descriptions with temporal nodes and edges.

    Parameters
    ----------
    outline: dict or path
        the temporal outline, with `n_timesteps` key and `_{}` timestep subscripts
    n_timesteps: int, default = 3
        number of time steps to generate the graph
    infer_edges: bool, default = False
        same as normal Description
    """

    def __init__(self,
                 outline: Union[dict, str],
                 n_timesteps: int,
                 infer_edges: bool = False):
        self.n_timesteps = n_timesteps
        self.flattened_outline = temporal_outline_parser(outline, n_timesteps)
        super().__init__(self.flattened_outline,
                         infer_edges=infer_edges)
