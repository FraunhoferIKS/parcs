import pytest
from data.mapping_functions_data import EdgeFunctionsData
from pyparcs.api.mapping_functions import *


class TestEdgeFunctions:
    @staticmethod
    @pytest.mark.parametrize('array', EdgeFunctionsData.identity_data)
    def test_edge_identity(array):
        """Tests identity edge function"""
        assert np.array_equal(array, edge_identity(array))

    @staticmethod
    @pytest.mark.parametrize('array,params,expected_array',
                             EdgeFunctionsData.sigmoid_data)
    def test_edge_sigmoid(array, params, expected_array):
        """Tests sigmoid edge function"""
        assert np.array_equal(expected_array,
                              edge_sigmoid(array,
                                           alpha=params['alpha'],
                                           beta=params['beta'],
                                           gamma=params['gamma'],
                                           tau=params['tau']))

    @staticmethod
    @pytest.mark.parametrize('array,params,expected_array',
                             EdgeFunctionsData.gaussian_rbf_data)
    def test_edge_gaussian_rbf(array, params, expected_array):
        """Tests gaussian RBF edge function"""
        assert np.array_equal(expected_array,
                              edge_gaussian_rbf(array,
                                                alpha=params['alpha'],
                                                beta=params['beta'],
                                                gamma=params['gamma'],
                                                tau=params['tau']))

    @staticmethod
    @pytest.mark.parametrize('array,params,expected_array',
                             EdgeFunctionsData.arctan_data)
    def test_edge_arctan(array, params, expected_array):
        """Tests arctan edge function"""
        assert np.array_equal(expected_array,
                              edge_arctan(array,
                                          alpha=params['alpha'],
                                          beta=params['beta'],
                                          gamma=params['gamma']))
