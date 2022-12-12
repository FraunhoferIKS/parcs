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

from parcs.exceptions import *
from parcs.graph_builder.parsers import graph_file_parser
from parcs.cdag.graph_objects import Graph
import pytest

def test_full_data_with_no_data_nodes_raise_error():
    '''raises GraphError if `full_data=True` for a graph with no data node'''
    nodes, edges = graph_file_parser('./tests/misc/description_files/gdf_1.yml')
    g = Graph(nodes, edges)
    with pytest.raises(GraphError):
        g.sample(size=10, full_data=True)

def test_full_data_with_size_raise_error():
    '''raises GraphError if `size` is given when `full_data=True`'''
    nodes, edges = graph_file_parser('./tests/misc/description_files/gdf_2.yml')
    g = Graph(nodes, edges)
    with pytest.raises(GraphError):
        g.sample(size=10, full_data=True)

def test_no_size_no_reusing_raise_error():
    '''raises GraphError if `size` is not given and `use_sampled_errors=False`'''
    nodes, edges = graph_file_parser('./tests/misc/description_files/gdf_1.yml')
    g = Graph(nodes, edges)
    g.sample(size=10, return_errors=True)
    with pytest.raises(GraphError):
        g.sample(use_sampled_errors=False)

def test_size_while_reusing_raise_error():
    '''raises GraphError if `size` is given while `use_sampled_errors=True`'''
    nodes, edges = graph_file_parser('./tests/misc/description_files/gdf_1.yml')
    g = Graph(nodes, edges)
    d, e = g.sample(size=10, return_errors=True)
    with pytest.raises(GraphError):
        g.sample(use_sampled_errors=True, sampled_errors=e, size=10)

def test_inconsistent_data_error_raise_error():
    '''raises DataError if sampled errors for data nodes are inconsistent'''
    nodes, edges = graph_file_parser('./tests/misc/description_files/gdf_2.yml')
    g = Graph(nodes, edges)
    d, e = g.sample(size=10, return_errors=True)
    # sabotage error
    e.loc[0, 'Z_1'] = 0.9
    with pytest.raises(DataError):
        g.sample(use_sampled_errors=True, sampled_errors=e)