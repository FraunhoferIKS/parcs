import pytest
from pyparcs import Guideline, GuidelineIterator


class TestGuideline:
    """Tests Guideline methods"""
    @staticmethod
    def tests_sample_keys_method():
        outline = {
            'nodes': {'dist1': {}, 'dist2': {}},
            'edges': {'func1': {}, 'func2': {}}
        }
        guideline = Guideline(outline)
        # node sample
        assert guideline.sample_keys(which='nodes') in ['dist1', 'dist2']
        # edge sample
        assert guideline.sample_keys(which='edges') in ['func1', 'func2']

    @staticmethod
    def tests_sample_values_method():
        outline = {
            'nodes': {'bernoulli': {'p_': [['f-range', 0.3, 0.4], 10, 20]}},
            'edges': {'sigmoid': {'alpha': ['f-range', -2, -1], 'beta': 0, 'gamma': 9, 'tau': 11}}
        }
        guideline = Guideline(outline)
        assert 0.3 < guideline.sample_values('nodes.bernoulli.p_.0') < 0.4
        assert guideline.sample_values('nodes.bernoulli.p_.1') == 10
        assert guideline.sample_values('nodes.bernoulli.p_.2') == 20

        assert -2 < guideline.sample_values('edges.sigmoid.alpha') < -1
        assert guideline.sample_values('edges.sigmoid.beta') == 0
        assert guideline.sample_values('edges.sigmoid.gamma') == 9
        assert guideline.sample_values('edges.sigmoid.tau') == 11


class TestGuidelineIterator:
    """Tests guideline iterator"""
    @staticmethod
    def test_choice_iterator():
        """Tests returns for three directive types"""
        g_outline = {'nodes': {'bernoulli': {'p_': [['f-range', 0, 2],
                                                    ['i-range', 0, 4],
                                                    ['choice', 2, 3, 4]]}}}
        iterator = GuidelineIterator(g_outline)

        for guideline, i in zip(iterator.get_guidelines('nodes.bernoulli.p_.bias', steps=0.5),
                                [0, 0.5, 1, 1.5]):
            assert guideline.sample_values('nodes.bernoulli.p_.0') == i

        for guideline, i in zip(iterator.get_guidelines('nodes.bernoulli.p_.linear', steps=1),
                                [0, 1, 2, 3]):
            assert guideline.sample_values('nodes.bernoulli.p_.1') == i

        for guideline, i in zip(iterator.get_guidelines('nodes.bernoulli.p_.interactions'),
                                [2, 3, 4]):
            assert guideline.sample_values('nodes.bernoulli.p_.2') == i
