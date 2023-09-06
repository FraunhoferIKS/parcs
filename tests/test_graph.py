import pytest
from pyparcs.api.graph_objects import Node
from pyparcs import Description, Graph
from pyparcs.api.mapping_functions import *
import pandas as pd
from pyparcs.core.exceptions import InterventionError
from data.graph_data import InterventionData

class TestGraph:
    @staticmethod
    def tests_simple_run():
        # PARCS GRAPH
        outline = {'Z_1': 'normal(mu_=0, sigma_=1)',
                   'Z_2': 'bernoulli(p_=0.3)',
                   'Z_3': 'lognormal(mu_=Z_1+Z_2+2Z_1^2, sigma_=1+Z_2^2)',
                   'Z_1->Z_3': 'identity()',
                   'Z_2->Z_3': 'sigmoid(alpha=1, beta=0, gamma=0, tau=1)'}
        desc = Description(outline)
        graph = Graph(desc)
        samples, errors = graph.sample(10)

        # MANUAL GRAPH
        z_1 = Node(output_distribution='normal',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [], 'interactions': []},
                       'sigma_': {'bias': 1, 'linear': [], 'interactions': []}},
                   do_correction=False)
        z_2 = Node(output_distribution='bernoulli',
                   dist_params_coefs={
                       'p_': {'bias': 0.3, 'linear': [], 'interactions': []}},
                   do_correction=False)
        z_3 = Node(output_distribution='lognormal',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [1, 1], 'interactions': [2, 0, 0]},
                       'sigma_': {'bias': 1, 'linear': [0, 0], 'interactions': [0, 0, 1]}},
                   do_correction=False)
        data = pd.DataFrame([], columns=('Z_1', 'Z_2', 'Z_3'))
        data['Z_1'] = z_1.calculate(data, [], errors['Z_1'])
        data['Z_2'] = z_2.calculate(data, [], errors['Z_2'])
        input_data = pd.DataFrame({
            'Z_1': edge_identity(data['Z_1'].values),
            'Z_2': edge_sigmoid(data['Z_2'].values, alpha=1, beta=0, gamma=0, tau=1)})
        data['Z_3'] = z_3.calculate(input_data, ['Z_1', 'Z_2'], errors['Z_3'])
        assert samples.equals(data)

    def tests_with_correction(self):
        # PARCS GRAPH
        outline = {'Z_1': 'normal(mu_=0, sigma_=1)',
                   'Z_2': 'bernoulli(p_=Z_1^2), correction[]',
                   'Z_3': 'exponential(lambda_=Z_1-Z_2), correction[lower=0, upper=10]',
                   'Z_1->Z_2': 'identity()',
                   'Z_1->Z_3': 'gaussian_rbf(alpha=1, beta=0, gamma=1, tau=2)',
                   'Z_2->Z_3': 'arctan(alpha=1, beta=0, gamma=0)'}
        desc = Description(outline)
        graph = Graph(desc)
        samples, errors = graph.sample(10)

        # MANUAL GRAPH
        z_1 = Node(output_distribution='normal',
                   dist_params_coefs={
                       'mu_': {'bias': 0, 'linear': [], 'interactions': []},
                       'sigma_': {'bias': 1, 'linear': [], 'interactions': []}},
                   do_correction=False)
        z_2 = Node(output_distribution='bernoulli',
                   dist_params_coefs={
                       'p_': {'bias': 0, 'linear': [0], 'interactions': [1]}},
                   do_correction=True,
                   correction_config={'lower': 0, 'upper': 1})
        z_3 = Node(output_distribution='exponential',
                   dist_params_coefs={
                       'lambda_': {'bias': 0, 'linear': [1, -1], 'interactions': [0, 0, 0]}},
                   do_correction=True,
                   correction_config={'lower': 0, 'upper': 10})

        data = pd.DataFrame([], columns=('Z_1', 'Z_2', 'Z_3'))
        data['Z_1'] = z_1.calculate(data, [], errors['Z_1'])
        data['Z_2'] = z_2.calculate(data, ['Z_1'], errors['Z_2'])
        input_data = pd.DataFrame({
            'Z_1': edge_gaussian_rbf(data['Z_1'].values, alpha=1, beta=0, gamma=1, tau=2),
            'Z_2': edge_arctan(data['Z_2'].values, alpha=1, beta=0, gamma=0)})
        data['Z_3'] = z_3.calculate(input_data, ['Z_1', 'Z_2'], errors['Z_3'])
        assert samples.equals(data)

    @staticmethod
    def tests_dummy_tags():
        outline = {'A': 'bernoulli(p_=0.3), tags[D]',
                   'B': 'normal(mu_=A+2, sigma_=1)'}
        graph = Graph(Description(outline, infer_edges=True))
        samples, errors = graph.sample(10)

        assert set(samples.columns) == {'B'} and set(errors.columns) == {'A', 'B'}


class TestIntervention:
    @staticmethod
    @pytest.mark.parametrize('outline,parent,child', InterventionData.functional_do_data)
    def test_functional_do(outline, parent, child):
        desc = Description(outline, infer_edges=True)
        graph = Graph(desc)
        samples, _ = graph.do_functional(size=100,
                                         intervene_on=child,
                                         inputs=[parent],
                                         func=lambda i: i + 1)
        assert samples[child].equals(samples[parent] + 1)

    @staticmethod
    @pytest.mark.parametrize('outline,parents,child',
                             InterventionData.functional_do_two_parent_data)
    def test_functional_do_two_parents(outline, parents, child):
        desc = Description(outline, infer_edges=True)
        graph = Graph(desc)
        samples, _ = graph.do_functional(size=100,
                                         intervene_on=child,
                                         inputs=parents,
                                         func=lambda i, j: i + j)
        assert samples[child].equals(samples[parents[0]] + samples[parents[1]])

    @staticmethod
    @pytest.mark.parametrize('outline,parent,child',
                             InterventionData.functional_do_loop_error_data)
    def test_functional_do_raise_errors(outline, parent, child):
        desc = Description(outline, infer_edges=True)
        graph = Graph(desc)
        with pytest.raises(InterventionError):
            graph.do_functional(size=100,
                                intervene_on=child,
                                inputs=[parent],
                                func=lambda i: i+1)

    @staticmethod
    @pytest.mark.parametrize('description', [
        ({'A': 'constant(3)'})
    ])
    def test_intervene_on_constant(description):
        desc = Description(description)
        graph = Graph(desc, warning_level='error')
        graph.do(size=10, interventions={'A': -1})

    @staticmethod
    @pytest.mark.parametrize('description', InterventionData.do_erroneous_data)
    def test_do_raises_error(description):
        with pytest.raises(InterventionError):
            desc = Description(description)
            graph = Graph(desc, warning_level='error')
            graph.do(size=10, interventions={'A': -1})

    @staticmethod
    @pytest.mark.parametrize('description',
                             InterventionData.support_error_data)
    def test_do_functional_raises_error(description):
        with pytest.raises(InterventionError):
            desc = Description(description)
            graph = Graph(desc, warning_level='error')
            graph.do_functional(
                size=3,
                intervene_on='A', inputs=['B'],
                func=lambda b: b - 1
            )
        with pytest.raises(InterventionError):
            desc = Description(description)
            graph = Graph(desc, warning_level='error')
            graph.do_self(size=10, func=lambda a: a - 100, intervene_on='A')
