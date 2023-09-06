from pyparcs import Description, Guideline, RandomDescription
import pandas as pd
import numpy as np
import pytest


class TestDescription:
    """Tests the base Description class"""

    @staticmethod
    def tests_attributes():
        """Tests if Description attributes have been set up correctly"""
        outline = {'A': 'bernoulli(p_=0.2), correction[]',
                   'B': 'normal(mu_=A+2, sigma_=1), tags[G1, P2, C3]',
                   'C': 'constant(2), tags[D]',
                   'D': 'data(./dir/file.csv, D)',
                   'A->B': 'identity(), correction[], tags[L2]'}
        desc = Description(outline)

        # outline is fully specified
        assert not desc.is_partial
        # types
        assert desc.node_types == {'A': 'stochastic', 'B': 'stochastic',
                                   'C': 'constant', 'D': 'data'}
        # tags
        assert desc.edge_tags == {'A->B': ['L2']}
        assert desc.node_tags == {'A': [], 'B': ['G1', 'P2', 'C3'], 'C': ['D'], 'D': []}
        assert desc.parents_list == {'A': [], 'B': ['A'], 'C': [], 'D': []}
        assert desc.sorted_node_list == ['A', 'B', 'C', 'D']
        assert desc.adj_matrix.equals(pd.DataFrame([[0, 1, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0],
                                                    [0, 0, 0, 0]],
                                                   index=('A', 'B', 'C', 'D'),
                                                   columns=('A', 'B', 'C', 'D')))
        assert (not desc.is_dummy('A') and
                not desc.is_dummy('B') and
                not desc.is_dummy('D') and
                desc.is_dummy('C'))

    @staticmethod
    def test_infer_edges():
        """Tests if can infer non-existing edges from the node parents"""
        outline = {'A': 'bernoulli(p_=0.3)',
                   'B': 'normal(mu_=A+1, sigma_=2)',
                   'C': 'bernoulli(p_=A+B), correction[]',
                   'D': 'normal(mu_=0, sigma_=1)'}
        desc = Description(outline, infer_edges=True)
        # must infer A->B, A->C and B->C edges
        assert set(desc.edges.keys()) == {'A->B', 'A->C', 'B->C'}
        assert all(edge['function_name'] == 'identity' for edge in desc.edges.values())

    @staticmethod
    def test_randomize_parameters():
        """Tests randomization of parameters"""
        desc_outline = {'A': 'bernoulli(p_=?)',
                        'B': 'normal(mu_=A+?, sigma_=2)',
                        'C': 'bernoulli(p_=A+B), correction[]',
                        'D': 'normal(mu_=1, sigma_=?)',
                        'A->B': 'identity()', 'A->C': 'identity()', 'B->C': 'identity()',
                        'A->D': 'identity()', 'B->D': 'identity()'}
        guide_outline = {
            'nodes': {
                'bernoulli': {'p_': [['f-range', 0, 1], 5, 7]},
                'normal': {'mu_': [['f-range', 110, 111], 6, 8],
                           'sigma_': [['f-range', 10, 11], 20, 21]}
            }
        }
        desc = Description(desc_outline)
        guideline = Guideline(guide_outline)
        desc.randomize_parameters(guideline)

        # A & B biases, and sigma of D
        a_p_ = desc.nodes['A']['dist_params_coefs']['p_']
        assert 0 < a_p_['bias'] < 1
        b_mu_ = desc.nodes['B']['dist_params_coefs']['mu_']
        assert (110 < b_mu_['bias'] < 111 and
                b_mu_['linear'] == [1] and
                b_mu_['interactions'] == [0])
        d_sigma_ = desc.nodes['D']['dist_params_coefs']['sigma_']
        assert (10 < d_sigma_['bias'] < 11 and
                np.array_equal(d_sigma_['linear'], [20, 20]) and
                np.array_equal(d_sigma_['interactions'], [21, 21, 21]))
        assert not desc.is_partial

    @staticmethod
    def test_randomize_tags():
        """Tests if can randomize according to randomization tag 'P'."""
        desc_outline = {'A': 'bernoulli(p_=?), tags[P1]',
                        'B': 'normal(mu_=A+?, sigma_=2), tags[P2]',
                        'C': 'bernoulli(p_=A+B), correction[]',
                        'D': 'normal(mu_=1, sigma_=?)',
                        'A->B': 'identity()', 'A->C': 'identity()', 'B->C': 'identity()',
                        'A->D': 'identity()', 'B->D': 'identity()'}
        guide_outline = {
            'nodes': {
                'bernoulli': {'p_': [['f-range', 0, 1], 5, 7]},
                'normal': {'mu_': [['f-range', 110, 111], 6, 8],
                           'sigma_': [['f-range', 10, 11], 20, 21]}
            }
        }
        desc = Description(desc_outline)
        guideline = Guideline(guide_outline)
        # must only randomize D because it doesn't have any tag
        desc.randomize_parameters(guideline)
        a_p_ = desc.nodes['A']['dist_params_coefs']['p_']
        assert a_p_['bias'] == '?'
        b_mu_ = desc.nodes['B']['dist_params_coefs']['mu_']
        assert b_mu_['bias'] == '?' and b_mu_['linear'] == [1]
        d_sigma_ = desc.nodes['D']['dist_params_coefs']['sigma_']
        assert (10 < d_sigma_['bias'] < 11 and
                np.array_equal(d_sigma_['linear'], [20, 20]) and
                np.array_equal(d_sigma_['interactions'], [21, 21, 21]))
        # is_partial must be True because of A and B
        assert desc.is_partial
        # Must randomize A only, for its P1 tag
        desc.randomize_parameters(guideline, 'P1')
        a_p_ = desc.nodes['A']['dist_params_coefs']['p_']
        assert 0 < a_p_['bias'] < 1
        b_mu_ = desc.nodes['B']['dist_params_coefs']['mu_']
        assert b_mu_['bias'] == '?'
        # is_partial must be True because of B
        assert desc.is_partial
        # Must randomize B for its P2 tag
        desc.randomize_parameters(guideline, 'P2')
        b_mu_ = desc.nodes['B']['dist_params_coefs']['mu_']
        assert 110 < b_mu_['bias'] < 111
        # is_partial must be False after completion
        assert not desc.is_partial

    @staticmethod
    def test_randomize_connection_base():
        """Tests randomize connection for a 2-node to 2-node connection"""
        parent_outline = {'A': 'bernoulli(p_=0.2)',
                          'B': 'normal(mu_=0, sigma_=1)',
                          'A->B': 'identity()'}
        child_outline = {'C': 'bernoulli(p_=0.99)',
                         'D': 'bernoulli(p_=0.3)'}
        guide_outline = {
            'nodes': {
                'bernoulli': {'p_': [['f-range', 0, 1], 5, 7]}
            },
            'edges': {'identity': None},
            'graph': {'density': 1}
        }

        desc = Description(parent_outline)
        guideline = Guideline(guide_outline)
        desc.randomize_connection_to(child_outline, guideline)

        assert desc.sorted_node_list == ['A', 'B', 'C', 'D']
        # including connection edges
        assert set(desc.edges.keys()) == {'A->B', 'A->C', 'A->D', 'B->C', 'B->D'}
        # new edges must be sampled from the given guideline
        c_p_ = desc.nodes['C']['dist_params_coefs']['p_']
        assert (c_p_['bias'] == 0.99 and
                np.array_equal(c_p_['linear'], [5, 5]) and
                np.array_equal(c_p_['interactions'], [7, 7, 7]))
        d_p_ = desc.nodes['D']['dist_params_coefs']['p_']
        assert (d_p_['bias'] == 0.3 and
                np.array_equal(d_p_['linear'], [5, 5]) and
                np.array_equal(d_p_['interactions'], [7, 7, 7]))

    @staticmethod
    def test_randomize_connection_with_limit():
        """Tests if apply_limit will exclude the parameter with the '!' sign."""
        parent_outline = {'A': 'bernoulli(p_=0.2)',
                          'B': 'normal(mu_=0, sigma_=1)',
                          'A->B': 'identity()'}
        child_outline = {'C': 'bernoulli(!p_=0.99)',
                         'D': 'bernoulli(p_=0.3)'}
        guide_outline = {
            'nodes': {
                'bernoulli': {'p_': [['f-range', 0, 1], 5, 7]}
            },
            'edges': {'identity': None},
            'graph': {'density': 1}
        }

        desc = Description(parent_outline)
        guideline = Guideline(guide_outline)
        desc.randomize_connection_to(child_outline, guideline, apply_limit=True)

        assert desc.sorted_node_list == ['A', 'B', 'C', 'D']
        assert set(desc.edges.keys()) == {'A->B', 'A->D', 'B->D'}
        # C must remain the same, as its parameter was marked with !
        c_p_ = desc.nodes['C']['dist_params_coefs']['p_']
        assert (c_p_['bias'] == 0.99 and
                c_p_['linear'] == [] and
                c_p_['interactions'] == [])
        d_p_ = desc.nodes['D']['dist_params_coefs']['p_']
        assert (d_p_['bias'] == 0.3 and
                np.array_equal(d_p_['linear'], [5, 5]) and
                np.array_equal(d_p_['interactions'], [7, 7, 7]))

    @staticmethod
    def test_randomize_connection_mask():
        """Tests if applying a mask will exclude certain edges."""
        parent_outline = {'A': 'bernoulli(p_=0.2)',
                          'B': 'normal(mu_=0, sigma_=1)',
                          'A->B': 'identity()'}
        child_outline = {'C': 'bernoulli(p_=0.99)',
                         'D': 'bernoulli(p_=0.3)'}
        guide_outline = {
            'nodes': {
                'bernoulli': {'p_': [['f-range', 0, 1], 5, 7]}
            },
            'edges': {'identity': None},
            'graph': {'density': 1}
        }

        desc = Description(parent_outline)
        guideline = Guideline(guide_outline)
        # scrambled index name to test undesired sensitivity to the name orders
        desc.randomize_connection_to(child_outline,
                                     guideline,
                                     mask=pd.DataFrame([[1, 1],
                                                        [0, 1]],
                                                       index=('B', 'A'),
                                                       columns=('C', 'D')))
        # edge A->C must be suppressed
        assert set(desc.edges.keys()) == {'A->B', 'A->D', 'B->C', 'B->D'}

    @staticmethod
    @pytest.mark.parametrize('outline, guideline, expected_outline', [
        # general term with 0 bias (test if bias is omitted as expected)
        (
                {'A': 'normal(mu_=?, sigma_=?)',
                 'B': 'normal(mu_=?, sigma_=?)',
                 'Y': 'bernoulli(p_=?), correction[target_mean=0.5]',
                 'A->Y': 'random', 'B->Y': 'identity()'},
                Guideline({'nodes': {'normal': {'mu_': [2, 0, 0],
                                     'sigma_': [0.5, 0, 0]},
                           'bernoulli': {'p_': [0, 2, -1.9]}},
                           'edges': {'identity': None}}),
                {'A': 'normal(mu_=2, sigma_=0.5)',
                 'B': 'normal(mu_=2, sigma_=0.5)',
                 'Y': 'bernoulli(p_=2A+2B-1.9A^2-1.9AB-1.9B^2), correction[target_mean=0.5]',
                 'A->Y': 'identity(), correction[]', 'B->Y': 'identity()'}
        ),
        # test "-1" coefficient
        (
                {'A': 'normal(mu_=?, sigma_=?)',
                 'B': 'normal(mu_=?, sigma_=?)',
                 'Y': 'bernoulli(p_=?), correction[target_mean=0.5]',
                 'A->Y': 'identity()', 'B->Y': 'identity()'},
                Guideline({'nodes': {'normal': {'mu_': [2, 0, 0],
                                                'sigma_': [0.5, 0, 0]},
                                     'bernoulli': {'p_': [2, -1, 0]}}}),
                {'A': 'normal(mu_=2, sigma_=0.5)',
                 'B': 'normal(mu_=2, sigma_=0.5)',
                 'Y': 'bernoulli(p_=2-A-B), correction[target_mean=0.5]',
                 'A->Y': 'identity()',
                 'B->Y': 'identity()'}
        ),
        # test "1" coefficient
        (
                {'A': 'normal(mu_=?, sigma_=?)',
                 'B': 'normal(mu_=?, sigma_=?)',
                 'Y': 'bernoulli(p_=?), correction[target_mean=0.5]',
                 'A->Y': 'identity()', 'B->Y': 'identity()'},
                Guideline({'nodes': {'normal': {'mu_': [2, 0, 0],
                                                'sigma_': [0.5, 0, 0]},
                                     'bernoulli': {'p_': [2, 1, 1]}}}),
                {'A': 'normal(mu_=2, sigma_=0.5)',
                 'B': 'normal(mu_=2, sigma_=0.5)',
                 'Y': 'bernoulli(p_=2+A+B+A^2+AB+B^2), correction[target_mean=0.5]',
                 'A->Y': 'identity()',
                 'B->Y': 'identity()'}
        ),
        # non stochastic nodes
        (
                {'A': 'random', 'B': 'constant(2)'},
                Guideline({'nodes': {'bernoulli': {'p_': [0.2, 0, 0]}}}),
                {'A': 'bernoulli(p_=0.2), correction[]', 'B': 'constant(2)'}
        ),
        # still partially specified after randomization
        # -- all random node
        (
                {'A': 'random, tags[P1]'},
                Guideline({'nodes': {'bernoulli': {'p_': [2, 1, 1]}}}),
                {'A': 'random'}
        )
    ])
    def test_updated_outline_after_randomization(outline, guideline, expected_outline):
        description = Description(outline)
        description.randomize_parameters(guideline)
        assert description.outline == expected_outline


class TestRandomDescription:
    @staticmethod
    def tests_description_making():
        guide_outline = {
            'nodes': {
                'bernoulli': {'p_': [['f-range', 0, 1], 5, 7]}
            },
            'edges': {'identity': None},
            'graph': {'density': 1, 'num_nodes': 4}
        }
        guideline = Guideline(guide_outline)
        desc = RandomDescription(guideline, node_prefix='H')

        assert desc.sorted_node_list == ['H_0', 'H_1', 'H_2', 'H_3']
        # because density = 1
        assert len(desc.edges) == 6
