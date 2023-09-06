import os
import pytest
from pyparcs.api.parsers import *
from pyparcs import Description
from pyparcs import Description, Guideline
from data.parsers_data import *


@pytest.mark.parametrize('line,addition,limit,expected',
                         MiscData.augment_line_data)
def test_augment_line(line, addition, limit, expected):
    assert augment_line(line, addition, limit) == expected


@pytest.mark.parametrize('line,tags', MiscData.tag_parser_data)
def test_tag_parser(line, tags):
    assert set(tags_parser(line)) == tags


@pytest.mark.parametrize('line', MiscData.tag_parser_erroneous_data)
def test_tag_parser(line):
    with pytest.raises(DescriptionError):
        tags_parser(line)


class TestTermParser:
    """
    Term parser parses the individual terms in an equation such as 2XY, -0.5Z, etc.
    """
    @staticmethod
    @pytest.mark.parametrize('term,vars,exp_pars,exp_coef',
                             TermParserData.term_parser_data)
    def test_parse_terms_correctly(term, vars, exp_pars, exp_coef):
        """Tests normal behavior"""
        pars, coef = term_parser(term, vars)
        assert sorted(pars) == sorted(exp_pars)
        assert coef == exp_coef

    @staticmethod
    @pytest.mark.parametrize('term,vars,err',
                             TermParserData.term_parser_erroneous_data)
    def test_parse_terms_raise_correct_error(term, vars, err):
        """Tests raising the correct error"""
        with pytest.raises(err):
            term_parser(term, vars)


class TestEquationParser:
    """
    This parser parses equations which are made of terms, e.g. '2X + 3Y - X^2 + 1'
    """
    @staticmethod
    @pytest.mark.parametrize('eq,vars,output', EquationParserData.eq_parser)
    def test_parse_equations(eq, vars, output):
        """Tests normal behavior"""
        for gen, correct in zip(equation_parser(eq, vars), output):
            assert set(gen[0]) == set(correct[0])  # parents are correct
            assert gen[1] == correct[1]  # coefficient is correct

    @staticmethod
    @pytest.mark.parametrize('eq,vars,err',
                             EquationParserData.eq_parser_erroneous_data)
    def test_parse_equations_raise_error(eq, vars, err):
        """Tests raising the correct error"""
        with pytest.raises(err):
            equation_parser(eq, vars)


class TestNodeParser:
    """
    This parser parses lines of description files to give config dicts for nodes.
    """
    @staticmethod
    @pytest.fixture(scope='class')
    def write_custom_function_py():
        """Fixture for deterministic nodes"""
        # setup
        with open('./customs.py', 'w') as script:
            script.write("def custom_function(data): return data['A'] + data['B']")
        # test
        yield True
        # teardown
        os.remove('./customs.py')

    @staticmethod
    @pytest.mark.parametrize('line,parents,dict_output',
                             NodeParserData.const_node_data)
    def test_parse_constant_node(line, parents, dict_output):
        """Tests constant node parsing"""
        assert node_parser(line, parents) == dict_output

    @staticmethod
    @pytest.mark.parametrize('line,parents',
                             NodeParserData.const_node_erroneous_data)
    def test_parse_constant_node_raise_error(line, parents):
        """Tests constant node raises error"""
        with pytest.raises(DescriptionError):
            node_parser(line, parents)

    @staticmethod
    @pytest.mark.parametrize('line,parents,dist,param_coefs,do_correction,correction_config',
                             NodeParserData.stochastic_node_data)
    def test_parse_stochastic_node(line, parents, dist, param_coefs,
                                   do_correction, correction_config):
        """Tests stochastic node parsing"""
        # distribution
        assert (out := node_parser(line, parents))['output_distribution'] == dist
        # params
        assert set(out['dist_params_coefs'].keys()) == set(param_coefs.keys())
        # coefs
        for param in out['dist_params_coefs'].keys():
            for coef_type in ['bias', 'linear', 'interactions']:
                assert out['dist_params_coefs'][param][coef_type] == param_coefs[param][coef_type]
        # correction
        assert out['do_correction'] == do_correction
        assert out['correction_config'] == correction_config

    @staticmethod
    @pytest.mark.parametrize('line,parents',
                             NodeParserData.stochastic_node_erroneous_data)
    def test_parse_stochastic_node_raises_error(line, parents):
        """Tests stochastic node raises error"""
        with pytest.raises(DescriptionError):
            node_parser(line, parents)

    @staticmethod
    def test_parse_deterministic_node(write_custom_function_py):
        """Tests deterministic node parsing"""
        out = node_parser('deterministic(customs.py, custom_function)',
                          ['A', 'B'])
        assert 'function' in out.keys()
        assert out['function'].__name__ == 'custom_function'

    @staticmethod
    def test_parse_deterministic_node_raises_error(write_custom_function_py):
        """Tests deterministic node raises error"""
        with pytest.raises(ExternalResourceError):
            node_parser('deterministic(non_existing.py, custom_function)',
                        ['A', 'B'])
        with pytest.raises(ExternalResourceError):
            node_parser('deterministic(customs.py, non_existing_function)',
                        ['A', 'B'])

    @staticmethod
    def test_parse_data_node():
        """Tests data node parsing"""
        out = node_parser('data(./some_data.csv, A)', [])
        assert out['csv_dir'] == './some_data.csv' and out['col'] == 'A'

    @staticmethod
    def test_parse_data_node_raise_error():
        with pytest.raises(DescriptionError):
            """Tests data node raises error"""
            node_parser('data(./some_data.csv, V)', ['A'])

    @staticmethod
    def test_parse_random_node():
        """Tests random node parsing"""
        out = node_parser('random', ['A', 'B'])
        assert out['output_distribution'] == '?' and out['do_correction']

    @staticmethod
    @pytest.mark.parametrize('line,parents,expected_config',
                             NodeParserData.do_correction_data)
    def test_do_correction(line, parents, expected_config):
        assert node_parser(line, parents)['do_correction'] is True

    @staticmethod
    @pytest.mark.parametrize('line,parents',
                             NodeParserData.do_correction_erroneous_data)
    def test_do_correction_raises_error(line, parents):
        """Tests do correction raises error"""
        with pytest.raises(DescriptionError):
            node_parser(line, parents)


class TestEdgeParser:
    @staticmethod
    @pytest.mark.parametrize('line,func_name,func_params,do_correction',
                             EdgeParserData.edge_parser_data)
    def test_parse_edge(line, func_name, func_params, do_correction):

        assert func_name == (out := edge_parser(line))['function_name']
        assert func_params == out['function_params']
        assert do_correction == out['do_correction']

    @staticmethod
    def test_parse_random_edge():
        out = edge_parser('random')
        assert out['function_name'] == '?' and out['do_correction'] is True

    @staticmethod
    def test_parse_edge_correction():
        out = edge_parser('identity(), correction[]')
        assert out['function_name'] == 'identity'
        assert out['function_params'] == {}
        assert out['do_correction'] is True

    @staticmethod
    @pytest.mark.parametrize('line', EdgeParserData.edge_parser_erroneous_data)
    def test_parse_edge_raises_error(line):
        with pytest.raises(DescriptionError):
            edge_parser(line)


class TestDescriptionParser:
    @staticmethod
    @pytest.mark.parametrize('nodes_sublist, edges_sublist, expected_edges_sublist',
                             DescriptionParserData.infer_edges_data)
    def test_infer_edges(nodes_sublist, edges_sublist, expected_edges_sublist):
        edges_sublist = infer_missing_edges(nodes_sublist, edges_sublist)
        assert edges_sublist == expected_edges_sublist

    @staticmethod
    @pytest.mark.parametrize('outline', DescriptionParserData.infer_edges_erroneous_data)
    def tests_infer_edges_raises_error(outline):
        with pytest.raises(DescriptionError):
            Description(outline, infer_edges=True)


class TestSynthesizers:
    @staticmethod
    @pytest.mark.parametrize('parsed,synthesized', SynthesizerData.term_synthesizer_data)
    def test_term_synthesizer(parsed, synthesized):
        assert term_synthesizer(*parsed) == synthesized

    @staticmethod
    @pytest.mark.parametrize('parsed,synthesized', SynthesizerData.equation_synthesizer_data)
    def test_equation_synthesizer(parsed, synthesized):
        assert equation_synthesizer(parsed) == synthesized

    @staticmethod
    @pytest.mark.parametrize('node,parents,tags,synthesized', SynthesizerData.node_synthesizer_data)
    def test_node_synthesizer(node, parents, tags, synthesized):
        assert stochastic_node_synthesizer(node, parents, tags) == synthesized

    @staticmethod
    @pytest.mark.parametrize('edge,tags,synthesized', SynthesizerData.edge_synthesizer_data)
    def test_edge_synthesizer(edge, tags, synthesized):
        assert edge_synthesizer(edge, tags) == synthesized
