import os
import pytest
from pyparcs import Graph
from pyparcs.temporal import TemporalDescription


class TestTemporalDescription:
    """Tests the description for temporal outlines"""
    @staticmethod
    def test_temporal_parsing():
        """Tests normal behavior of temporal parsers"""
        temporal_outline = {'A': 'bernoulli(p_=0.5)',
                            'B_{t}': 'normal(mu_=A+B_{t-1}, sigma_=1)',
                            'B_{0}': 'constant(0)',
                            'A->B_{t}': 'identity()',
                            'B_{t-1}->B_{t}': 'identity()'}
        desc = TemporalDescription(temporal_outline, 3)

        assert set(desc.sorted_node_list) == {'A', 'B_0', 'B_1', 'B_2', 'B_3'}
        assert set(desc.edges) == {'A->B_1', 'A->B_2', 'A->B_3',
                                   'B_0->B_1', 'B_1->B_2', 'B_2->B_3'}

    @staticmethod
    def test_infer_edges_in_temporal():
        """Tests the behavior of infer_edges in temporal description"""
        temporal_outline = {'A': 'bernoulli(p_=0.5)',
                            'B_{t}': 'normal(mu_=A+B_{t-1}, sigma_=1)',
                            'B_{0}': 'constant(0)'}
        desc = TemporalDescription(temporal_outline, 3, infer_edges=True)

        assert set(desc.sorted_node_list) == {'A', 'B_0', 'B_1', 'B_2', 'B_3'}
        assert set(desc.edges) == {'A->B_1', 'A->B_2', 'A->B_3',
                                   'B_0->B_1', 'B_1->B_2', 'B_2->B_3'}