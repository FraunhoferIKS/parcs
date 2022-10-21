from parcs.exceptions import *
from parcs.graph_builder.parsers import graph_file_parser
from parcs.cdag.graph_objects import Graph
import pytest

def test_full_data_with_no_data_nodes_raise_error():
    '''raises GraphError if `full_data=True` for a graph with no data node'''
    nodes, edges = graph_file_parser('./tests/description_files/gdf_1.yml')
    g = Graph(nodes, edges)
    with pytest.raises(GraphError):
        g.sample(size=10, full_data=True)

def test_full_data_with_size_raise_error():
    '''raises GraphError if `size` is given when `full_data=True`'''
    nodes, edges = graph_file_parser('./tests/description_files/gdf_2.yml')
    g = Graph(nodes, edges)
    with pytest.raises(GraphError):
        g.sample(size=10, full_data=True)

def test_no_size_no_reusing_raise_error():
    '''raises GraphError if `size` is not given and `use_sampled_errors=False`'''
    nodes, edges = graph_file_parser('./tests/description_files/gdf_1.yml')
    g = Graph(nodes, edges)
    g.sample(size=10, return_errors=True)
    with pytest.raises(GraphError):
        g.sample(use_sampled_errors=False)

def test_size_while_reusing_raise_error():
    '''raises GraphError if `size` is given while `use_sampled_errors=True`'''
    nodes, edges = graph_file_parser('./tests/description_files/gdf_1.yml')
    g = Graph(nodes, edges)
    d, e = g.sample(size=10, return_errors=True)
    with pytest.raises(GraphError):
        g.sample(use_sampled_errors=True, sampled_errors=e, size=10)

def test_inconsistent_data_error_raise_error():
    '''raises DataError if sampled errors for data nodes are inconsistent'''
    nodes, edges = graph_file_parser('./tests/description_files/gdf_2.yml')
    g = Graph(nodes, edges)
    d, e = g.sample(size=10, return_errors=True)
    # sabotage error
    e.loc[0, 'Z_1'] = 0.9
    with pytest.raises(DataError):
        g.sample(use_sampled_errors=True, sampled_errors=e)