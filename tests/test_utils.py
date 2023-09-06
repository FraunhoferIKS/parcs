import pytest
import numpy as np
from data.utils_data import GraphUtilsData
from pyparcs.api import utils


class TestGraphUtils:
    """Tests utils related to graphs"""
    @staticmethod
    @pytest.mark.parametrize('node_list,parents_list,adj_matrix',
                             GraphUtilsData.get_adj_matrix)
    def tests_get_adj_matrix(node_list, parents_list, adj_matrix):
        """Tests creation of adjacency matrix"""
        adjm = utils.get_adj_matrix(node_list, parents_list)
        assert adjm.equals(adj_matrix)

    @staticmethod
    @pytest.mark.parametrize('adj_matrix,sorted_list',
                             GraphUtilsData.topological_sort)
    def tests_topological_sorting(adj_matrix, sorted_list):
        """Tests the topological sorting of adjacency matrices"""
        assert utils.topological_sort(adj_matrix) == sorted_list

    @staticmethod
    @pytest.mark.parametrize('data,result',
                             GraphUtilsData.get_interactions_values)
    def tests_get_interactions_values(data, result):
        """Tests calculation of interactions values for an array"""
        assert np.array_equal(utils.get_interactions_values(data),
                              result)

    @staticmethod
    @pytest.mark.parametrize('data_len,interactions_len',
                             GraphUtilsData.get_interactions_length)
    def tests_get_interactions_length(data_len, interactions_len):
        """Tests calculation of the length of the interactions for an array"""
        assert utils.get_interactions_length(data_len) == interactions_len

    @staticmethod
    @pytest.mark.parametrize('names,interactions_names',
                             GraphUtilsData.get_interactions_names)
    def tests_get_interactions_length(names, interactions_names):
        """Tests calculation of the names for the interaction list"""
        assert utils.get_interactions_names(names) == interactions_names

    @staticmethod
    @pytest.mark.parametrize('array,coef,result',
                             GraphUtilsData.dot_prod)
    def tests_dot_prod(array, coef, result):
        """Tests dot product function"""
        assert np.array_equal(utils.dot_prod(array, coef),
                              result)
