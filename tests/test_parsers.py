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
import numpy as np
import os
import pytest
from pyparcs.graph_builder.parsers import *
from pyparcs.exceptions import *


class TestTermParser:
    """
    Term parser parses the individual terms in an equation such as 2XY, -0.5Z, etc.
    """

    @staticmethod
    @pytest.mark.parametrize('term,vars,exp_pars,exp_coef', [
        # bias terms
        ('0', ['A', 'B', 'Z_1'], [], 0),
        ('1.5', ['A', 'B', 'Z_1'], [], 1.5),
        ('3', ['A', 'B', 'Z_1'], [], 3),
        ('-0', ['A', 'B', 'Z_1'], [], 0),
        ('-0.6', ['A', 'B', 'Z_1'], [], -0.6),
        ('-2', ['A', 'B', 'Z_1'], [], -2),
        # linear terms
        ('B', ['A', 'B', 'Z_1'], ['B'], 1),
        ('-0.3Z_1', ['A', 'B', 'Z_1'], ['Z_1'], -0.3),
        ('1.7A', ['A', 'B', 'Z_1'], ['A'], 1.7),
        # interaction terms
        ('ABZ_1', ['A', 'B', 'Z_1'], ['A', 'B', 'Z_1'], 1),
        ('2Z_1A', ['A', 'B', 'Z_1'], ['A', 'Z_1'], 2),
        ('-0.3BA', ['A', 'B', 'Z_1'], ['A', 'B'], -0.3),
        # quadratic terms
        ('A^2', ['A', 'B', 'Z_1'], ['A', 'A'], 1),
        ('1.6Z_1^2', ['A', 'B', 'Z_1'], ['Z_1', 'Z_1'], 1.6),
        ('-3B^2', ['A', 'B', 'Z_1'], ['B', 'B'], -3)
    ])
    def test_parse_terms_correctly(term, vars, exp_pars, exp_coef):
        """
        Tests whether the outputs are correct when inputs are correct.

        The test parameters:
            - term: the string term which the function receives
            - vars: the list of possible parents that the function must look for
            - exp_pars: first output - expected present parents in the term
            - exp_coef: second output - expected multiplicative factor of the term
        """
        pars, coef = term_parser(term, vars)
        assert sorted(pars) == sorted(exp_pars)
        assert coef == exp_coef

    @staticmethod
    @pytest.mark.parametrize('term,vars,err', [
        # bias terms
        ('J', ['A', 'B', 'Z_1'], DescriptionFileError),  # not existing parent
        ('AZ_1A', ['A', 'B', 'Z_1'], DescriptionFileError),  # parent duplicate
        ('AA', ['A', 'B', 'Z_1'], DescriptionFileError),  # parent duplicate
        ('2B^2A', ['A', 'B', 'Z_1'], DescriptionFileError),  # invalid quadratic term
        ('2AB^2', ['A', 'B', 'Z_1'], DescriptionFileError),  # invalid quadratic term
        ('2B^3', ['A', 'B', 'Z_1'], DescriptionFileError)  # invalid power
    ])
    def test_parse_terms_raise_correct_error(term, vars, err):
        """
        Tests whether an error is raised in case of invalid inputs.

        The test parameters:
            - term: the string term which the function receives
            - vars: the list of possible parents that the function must look for
            - err: the error it must return
        """
        with pytest.raises(err):
            term_parser(term, vars)


class TestEquationParser:
    """
    This parser parses equations which are made of terms, e.g. '2X + 3Y - X^2 + 1'
    """

    @staticmethod
    @pytest.mark.parametrize('eq,vars,output', [
        ('2+A-2.8B', ['A', 'B'], [([], 2), (['A'], 1.0), (['B'], -2.8)]),  # example equation, no space
        ('1', ['A', 'B'], [([], 1)]),  # only bias
        ('A^2-2AB', ['A', 'B'], [(['A', 'A'], 1), (['A', 'B'], -2.0)])  # quadratic terms with space
    ])
    def test_parse_equations(eq, vars, output):
        """
        Tests whether the outputs are correct when inputs are correct.

        The test parameters:
            - eq: the string term which the function receives
            - vars: the list of possible parents that the function must look for
            - output: the parsed equation
        """
        terms = equation_parser(eq, vars)
        for gen, correct in zip(sorted(terms), sorted(output)):
            assert set(gen[0]) == set(correct[0])  # parents are correct
            assert gen[1] == correct[1]  # coefficient is correct

    @staticmethod
    @pytest.mark.parametrize('eq,vars,err', [
        # duplicate terms
        ('2A + 3A', ['A', 'B'], DescriptionFileError),
        ('2AB + 3BA', ['A', 'B'], DescriptionFileError),
        ('B + 2A^2 - A^2', ['A', 'B'], DescriptionFileError),
        # non-existing parents
        ('B + 2A^2', ['B'], DescriptionFileError),
        # non-standard symbols
        ('A + B * 3', ['B'], DescriptionFileError),

    ])
    def test_parse_equations_raise_error(eq, vars, err):
        """
        Tests whether an error is raised in case of invalid inputs.

        The test parameters:
            - term: the string term which the function receives
            - vars: the list of possible parents that the function must look for
            - the error it must return
        """
        with pytest.raises(err):
            equation_parser(eq, vars)


class TestNodeParser:
    """
    This parser parses lines of description files to give config dicts for nodes.
    """

    @staticmethod
    @pytest.mark.parametrize('line,parents,dict_output', [
        ('constant(2)', ['A', 'B'], {'value': 2}),
        ('constant(-0.3)', ['A', 'B'], {'value': -0.3}),
        ('constant(0)', ['A', 'B'], {'value': 0}),
    ])
    def test_parse_constant_node(line, parents, dict_output):
        assert node_parser(line, parents) == dict_output

    @staticmethod
    @pytest.mark.parametrize('line,parents', [
        ('constant(A)', ['A', 'B']),
        ('constant()', ['A', 'B']),
    ])
    def test_parse_constant_node_raise_error(line, parents):
        with pytest.raises(DescriptionFileError):
            node_parser(line, parents)

    @staticmethod
    @pytest.mark.parametrize('line,parents,dist,param_coefs,do_correction,correction_config', [
        ('bernoulli(p_=2A+B^2)', ['A', 'B'], 'bernoulli',
         {'p_': {'bias': 0, 'linear': [2, 0], 'interactions': [0, 0, 1]}}, False, {}),
        ('gaussian(mu_=1-0.3AB, sigma_=2)', ['A', 'B'], 'gaussian',
         {'mu_': {'bias': 1, 'linear': [0, 0], 'interactions': [0, -0.3, 0]},
          'sigma_': {'bias': 2, 'linear': [0, 0], 'interactions': [0, 0, 0]}}, False, {}),
        ('uniform(mu_=4B, diff_=A^2)', ['A', 'B'], 'uniform',
         {'mu_': {'bias': 0, 'linear': [0, 4], 'interactions': [0, 0, 0]},
          'diff_': {'bias': 0, 'linear': [0, 0], 'interactions': [1, 0, 0]}}, False, {}),
        ('lognormal(mu_=A+B, sigma_=A)', ['A', 'B'], 'lognormal',
         {'mu_': {'bias': 0, 'linear': [1, 1], 'interactions': [0, 0, 0]},
          'sigma_': {'bias': 0, 'linear': [1, 0], 'interactions': [0, 0, 0]}}, False, {}),
        ('poisson(lambda_=B^2+1)', ['A', 'B'], 'poisson',
         {'lambda_': {'bias': 1, 'linear': [0, 0], 'interactions': [0, 0, 1]}}, False, {}),
        ('exponential(lambda_=-AB)', ['A', 'B'], 'exponential',
         {'lambda_': {'bias': 0, 'linear': [0, 0], 'interactions': [0, -1, 0]}}, False, {}),
        # parentless nodes: only test one distribution since logic is the same
        ('bernoulli(p_=2)', [], 'bernoulli',
         {'p_': {'bias': 2, 'linear': [], 'interactions': []}}, False, {}),
        # partially randomized cases
        ('bernoulli(?)', ['A'], 'bernoulli',
         {'p_': {'bias': '?', 'linear': '?', 'interactions': '?'}}, False, {}),
        ('gaussian(mu_=?, sigma_=2A)', ['A', 'B'], 'gaussian',
         {'mu_': {'bias': '?', 'linear': '?', 'interactions': '?'},
          'sigma_': {'bias': 0, 'linear': [2, 0], 'interactions': [0, 0, 0]}}, False, {}),
        ('gaussian(mu_=?, sigma_=?)', ['A', 'B'], 'gaussian',
         {'mu_': {'bias': '?', 'linear': '?', 'interactions': '?'},
          'sigma_': {'bias': '?', 'linear': '?', 'interactions': '?'}}, False, {}),
        ('lognormal(?)', ['A', 'B'], 'lognormal',
         {'mu_': {'bias': '?', 'linear': '?', 'interactions': '?'},
          'sigma_': {'bias': '?', 'linear': '?', 'interactions': '?'}}, False, {}),
    ])
    def test_parse_stochastic_node(line, parents, dist, param_coefs, do_correction, correction_config):
        out = node_parser(line, parents)
        # distribution
        assert out['output_distribution'] == dist
        # params
        assert set(out['dist_params_coefs'].keys()) == set(param_coefs.keys())
        # coefs
        for param in out['dist_params_coefs'].keys():
            for coef_type in ['bias', 'linear', 'interactions']:
                assert np.array_equal(out['dist_params_coefs'][param][coef_type], param_coefs[param][coef_type])
        # correction
        assert out['do_correction'] == do_correction
        assert out['correction_config'] == correction_config

    @staticmethod
    @pytest.mark.parametrize('line,parents', [
        ('fakedist(p_=2A+B^2)', ['A', 'B']),  # wrong distribution name
        ('bernoulli(mu_=2A+B^2)', ['A', 'B']),  # wrong parameter name
        ('gaussian(mu_=2A+B^2, mu_=2, sigma_=3)', ['A', 'B']),  # duplicate params
        ('exponential(lambda_=2A+B^2)', []),  # wrong parents
        ('poisson(lambda_=B^2)', ['A']),  # wrong parents
        ('poisson(lambda_=B^2+?A)', ['A'])  # wrong randomization
    ])
    def test_parse_stochastic_node_raises_error(line, parents):
        with pytest.raises(DescriptionFileError):
            node_parser(line, parents)

    @staticmethod
    @pytest.fixture(scope='class')
    def write_custom_function_py():
        # setup
        with open('./customs.py', 'w') as script:
            script.write("def custom_function(data): return data['A'] + data['B']")
        # test
        yield True
        # teardown
        os.remove('./customs.py')

    @staticmethod
    def test_parse_deterministic_node(write_custom_function_py):
        out = node_parser('deterministic(customs.py, custom_function)', ['A', 'B'])
        assert 'function' in out.keys()
        assert out['function'].__name__ == 'custom_function'

    @staticmethod
    def test_parse_deterministic_node_raises_error(write_custom_function_py):
        with pytest.raises(ExternalResourceError):
            node_parser('deterministic(non_existing.py, custom_function)', ['A', 'B'])
        with pytest.raises(ExternalResourceError):
            node_parser('deterministic(customs.py, non_existing_function)', ['A', 'B'])

    @staticmethod
    def test_parse_data_node():
        out = node_parser('data(./some_data.csv)', [])
        assert out == {'csv_dir': './some_data.csv'}

    @staticmethod
    def test_parse_data_node_raise_error():
        with pytest.raises(DescriptionFileError):
            node_parser('data(./some_data.csv)', ['A'])

    @staticmethod
    def test_parse_random_stochastic_node():
        out = node_parser('random', ['A', 'B'])
        assert out == {'output_distribution': '?', 'do_correction': True}

    @staticmethod
    @pytest.mark.parametrize('line,parents,expected_config', [
        ('bernoulli(p_=A), correction[target_mean=0.3, lower=0, upper=1]', ['A'], {
            'target_mean': 0.3, 'lower': 0, 'upper': 1}),
        ('bernoulli(p_=A), correction[]', ['A'], {})
    ])
    def test_do_correction(line, parents, expected_config):
        out = node_parser(line, parents)
        assert out['do_correction'] is True

    @staticmethod
    @pytest.mark.parametrize('line,parents', [
        ('bernoulli(p_=A), correction[lower=1, upper=2+X]', ['A']),  # non-float values
    ])
    def test_do_correction_raises_error(line, parents):
        """
        This functionality is only 'parsing'. for correctness of the params etc., we will have
        other tests e.g. in SigmoidCorrection section
        """
        with pytest.raises(DescriptionFileError):
            node_parser(line, parents)


class TestEdgeParser:
    @staticmethod
    @pytest.mark.parametrize('line,func_name,func_params,do_correction', [
        # normal
        ('identity()', 'identity', {}, False),
        ('sigmoid(alpha=2, beta=1, gamma=0, tau=1)', 'sigmoid',
         {'alpha': 2, 'beta': 1, 'gamma': 0, 'tau': 1}, False),
        ('gaussian_rbf(alpha=2, beta=1, gamma=0, tau=1)', 'gaussian_rbf',
         {'alpha': 2, 'beta': 1, 'gamma': 0, 'tau': 1}, False),
        ('arctan(alpha=2, beta=1, gamma=0)', 'arctan',
         {'alpha': 2, 'beta': 1, 'gamma': 0}, False),
        # partially randomized
        ('sigmoid(alpha=?, beta=1, gamma=?, tau=1), correction[]', 'sigmoid',
         {'alpha': '?', 'beta': 1, 'gamma': '?', 'tau': 1}, True),
        ('gaussian_rbf(?)', 'gaussian_rbf',
         {'alpha': '?', 'beta': '?', 'gamma': '?', 'tau': '?'}, False)
    ])
    def test_parse_edge(line, func_name, func_params, do_correction):
        out = edge_parser(line)
        assert func_name == out['function_name']
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
    @pytest.mark.parametrize('line', [
        # normal
        'fakeedge(alpha=1)',  # wrong edge function name
        'identity(alpha=1)',  # wrong params
        'gaussian_rbf(alpha=2, beta=1)',  # incomplete params
        'gaussian_rbf(alpha=2, beta=1, gamma=2, tau=A)',  # incomplete params
    ])
    def test_parse_edge_raises_error(line):
        with pytest.raises(DescriptionFileError):
            edge_parser(line)


class TestGraphFileParser:
    """
    In this step, we are certain of node and edge parsers. We only need to make sure the respective lines
    in the description file have been parsed with the correct parsers, therefore we only check
    the function/distribution names.

    Parser only cares for the correct parsing. consistency of the nodes and edges will be checked in graph
    """
    @staticmethod
    @pytest.fixture(params=[
        [("A: bernoulli(p_=0.2)\n"
          "B: gaussian(mu_=2A, sigma_=1)\n"
          "A->B: identity()"),
         # nodes
         {'A': 'bernoulli', 'B': 'gaussian'},
         # edges
         {'A->B': 'identity'}],
        [("A: bernoulli(p_=0.2)\n"
          "B: gaussian(mu_=1, sigma_=1)"),
         # nodes
         {'A': 'bernoulli', 'B': 'gaussian'},
         # edges
         {}],
        ["A: bernoulli(p_=0.2)",
         # nodes
         {'A': 'bernoulli'},
         # edges
         {}],
    ])
    def setup_gdf(request):
        # setup
        desc = request.param[0]
        nodes = request.param[1]
        edges = request.param[2]

        file_name = 'gdf.yml'
        with open(file_name, 'w') as f:
            f.write(desc)
        # test
        yield file_name, nodes, edges
        # teardown
        os.remove(file_name)

    @staticmethod
    def test_parses_gdf_correctly(setup_gdf):
        nodes, edges = graph_file_parser(setup_gdf[0])
        # Nodes
        # 1. names
        assert len(nodes) == len(setup_gdf[1])
        extracted_node_names = [i['name'] for i in nodes]
        assert set(extracted_node_names) == set(setup_gdf[1].keys())
        # 2. distributions
        for node in nodes:
            assert setup_gdf[1][node['name']] == node['output_distribution']

        # Edges
        assert len(edges) == len(setup_gdf[2])
        extracted_edge_names = [i['name'] for i in edges]
        assert set(extracted_edge_names) == set(setup_gdf[2].keys())
        # 2. distributions
        for edge in edges:
            assert setup_gdf[2][edge['name']] == edge['function_name']

    @staticmethod
    @pytest.fixture(params=[
        # edge has node name which doesn't exist
        ("A: bernoulli(p_=0.2)\n"
         "B: gaussian(mu_=2A, sigma_=1)\n"
         "A->C: identity()"),
        # node names other than Letter+underscore+number
        "A*&: random",  # characters other than letter + number + underscore
        "A-: random",  # characters other than letter + number + underscore
        "1_n: random",  # starting with number
        "A_: random",  # ending with underscore
        "_A: random",  # starting with underscore
        # bad yml file
        ("A; bernoulli(p_=0.2)\n\t\t"
         "B: gaussian(mu_=2A, sigma_=1)\n"
         "A->B: identity()")
    ])
    def setup_wrong_gdf(request):
        # setup
        desc = request.param

        file_name = 'gdf.yml'
        with open(file_name, 'w') as f:
            f.write(desc)
        # test
        yield file_name
        # teardown
        os.remove(file_name)

    @staticmethod
    def test_parses_gdf_raises_error(setup_wrong_gdf):
        with pytest.raises(DescriptionFileError):
            graph_file_parser(setup_wrong_gdf)
