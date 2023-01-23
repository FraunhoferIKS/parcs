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
import pytest
import numpy as np
from scipy import stats as dists
from pyparcs.cdag.output_distributions import *
from pyparcs.cdag.utils import *


class TestUtils:
    @staticmethod
    @pytest.mark.parametrize('data,coef,result', [
        (np.array([[2], [1]]), {'bias': 0, 'linear': [1], 'interactions': [1]}, [6, 2]),
        (np.array([[2, 1], [1, 1]]), {'bias': 0, 'linear': [1, 1], 'interactions': [0, 0, 2]}, [5, 4]),
        (np.array([]), {'bias': 2.7, 'linear': [], 'interactions': []}, 2.7),
        (np.array([[1, 2], [3, 4]]), {'bias': 0, 'linear': [0, 0], 'interactions': [1, 1, 1]}, [7, 37]),
    ])
    def test_dot_product(data, coef, result):
        assert np.array_equal(dot_prod(data, coef), result)


class TestBaseClasses:
    @staticmethod
    @pytest.mark.parametrize('name,data,coef', [
        ('p_1', np.array([[2], [1]]), {'bias': 1, 'linear': [-1], 'interactions': [2]}),
        ('p_2', np.array([[2, 1], [1, 1]]), {'bias': 0, 'linear': [1, 1], 'interactions': [0, 0, 2]}),
        ('p_1', np.array([]), {'bias': 2.7, 'linear': [], 'interactions': []}),
    ])
    def test_param_class(name, data, coef):
        """
        the dot_product function is already tested, so we rely on it in this unit test
        """
        param = DistParam(name=name, coef=coef, corrector=None)
        assert param.name == name
        assert np.array_equal(param.calculate(data), dot_prod(data, coef))

    @staticmethod
    @pytest.mark.parametrize('icdf,params,coefs,errors,data', [
        (dists.bernoulli.ppf, ['p'], {'p': {'bias': 1, 'linear': [], 'interactions': []}}, [0.2, 0.3], np.array([])),
        (dists.norm.ppf, ['loc', 'scale'], {
            'loc': {'bias': 0, 'linear': [1, 0], 'interactions': [2, 0, 0]},
            'scale': {'bias': 1, 'linear': [0, 0], 'interactions': [0, 0, 0]}
        }, [0.2, 0.3, 0.5], np.array([[1, 2], [2, 4], [1, 5]])),
    ])
    def test_parcs_dist_class_no_corrector(icdf, params, coefs, errors, data):
        dist = PARCSDistribution(icdf=icdf, params=params, coefs=coefs, correctors={p: None for p in params})
        assert np.array_equal(
            dist.calculate(data, errors),
            icdf(errors, **{p: dot_prod(data, coefs[p]) for p in params})
        )


class TestPARCSDistributions:
    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', [
        ({'p_': {'bias': 0.2, 'linear': [], 'interactions': []}}, [0.2, 0.3], np.array([])),
        ({'p_': {'bias': 0, 'linear': [2], 'interactions': [0]}}, [0.1, 0.9], np.array([[0.1], [0.3]])),
        ({'p_': {'bias': 0, 'linear': [1, 2], 'interactions': [0.2, 0.1, 0.4]}},
         [0.3, 0.1], np.array([[0.1, 0.05], [0.03, 0.02]])),
    ])
    def test_bernoulli_distribution(coefs, errors, data):
        dist = BernoulliDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        assert np.array_equal(
            out,
            dists.bernoulli.ppf(errors, p=dot_prod(data, coefs['p_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data,correction_config', [
        ({'p_': {'bias': 0.2, 'linear': [], 'interactions': []}},
         [0.2, 0.8],
         np.array([]),
         {'lower': 0, 'upper': 1, 'target_mean': None}),
        ({'p_': {'bias': 2, 'linear': [1, 3], 'interactions': [-1, 2, 2.5]}},
         np.linspace(0, 1, 9),
         np.random.uniform(size=(9, 2)),
         {'lower': 0, 'upper': 1, 'target_mean': None}),
    ])
    def test_bernoulli_correction(coefs, errors, data, correction_config):
        dist = BernoulliDistribution(coefs=coefs, do_correction=True, correction_config=correction_config)
        out = dist.calculate(data, errors)

        sigmoid_correction = SigmoidCorrection(**correction_config)
        assert np.array_equal(
            out,
            dists.bernoulli.ppf(
                errors,
                p=sigmoid_correction.transform(dot_prod(data, coefs['p_']))
            )
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', [
        ({'mu_': {'bias': 0, 'linear': [], 'interactions': []},
          'diff_': {'bias': 1, 'linear': [], 'interactions': []}}, [0.2, 0.3], np.array([])),
        ({'mu_': {'bias': 0, 'linear': [1, -0.5], 'interactions': [1, 0, 2]},
          'diff_': {'bias': 3, 'linear': [0, 0], 'interactions': [-1, -2, 0]}}, [0.01], np.array([[1, 4]])),
    ])
    def test_uniform_distribution(coefs, errors, data):
        dist = UniformDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        mu = dot_prod(data, coefs['mu_'])
        diff = dot_prod(data, coefs['diff_'])
        assert np.array_equal(
            out,
            mu + (np.array(errors) - 0.5) * diff
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', [
        ({'mu_': {'bias': 0, 'linear': [], 'interactions': []},
          'sigma_': {'bias': 1, 'linear': [], 'interactions': []}}, [0.2, 0.3], np.array([])),
        ({'mu_': {'bias': 0, 'linear': [1, -0.5], 'interactions': [1, 0, 2]},
          'sigma_': {'bias': 3, 'linear': [0, 0], 'interactions': [1, 2, 0]}}, [0.01], np.array([[1, 4]])),
    ])
    def test_normal_distribution(coefs, errors, data):
        dist = GaussianNormalDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)

        assert np.array_equal(
            out,
            dists.norm.ppf(errors, dot_prod(data, coefs['mu_']), dot_prod(data, coefs['sigma_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', [
        ({'lambda_': {'bias': 0.2, 'linear': [], 'interactions': []}}, [0.2, 0.3], np.array([])),
        ({'lambda_': {'bias': 0, 'linear': [2], 'interactions': [0]}}, [0.1, 0.9], np.array([[0.1], [0.3]])),
        ({'lambda_': {'bias': 0, 'linear': [1, 2], 'interactions': [3, 0.1, 4]}},
         [0.3, 0.1], np.array([[0.1, 0.5], [0.3, 2]])),
    ])
    def test_exponential_distribution(coefs, errors, data):
        dist = ExponentialDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        assert np.array_equal(
            out,
            dists.expon.ppf(errors, dot_prod(data, coefs['lambda_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', [
        ({'lambda_': {'bias': 0.2, 'linear': [], 'interactions': []}}, [0.2, 0.3], np.array([])),
        ({'lambda_': {'bias': 0, 'linear': [2], 'interactions': [0]}}, [0.1, 0.9], np.array([[0.1], [0.3]])),
        ({'lambda_': {'bias': 0, 'linear': [1, 2], 'interactions': [3, 0.1, 4]}},
         [0.3, 0.1], np.array([[0.1, 0.5], [0.3, 2]])),
    ])
    def test_poisson_distribution(coefs, errors, data):
        dist = PoissonDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        assert np.array_equal(
            out,
            dists.poisson.ppf(errors, dot_prod(data, coefs['lambda_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', [
        ({'mu_': {'bias': 0, 'linear': [], 'interactions': []},
          'sigma_': {'bias': 1, 'linear': [], 'interactions': []}}, [0.2, 0.7], np.array([])),
        ({'mu_': {'bias': 0, 'linear': [1, -0.5], 'interactions': [1, 0, 2]},
          'sigma_': {'bias': 3, 'linear': [0, 0], 'interactions': [1, 2, 0]}}, [0.1], np.array([[1, 4]])),
    ])
    def test_lognormal_distribution(coefs, errors, data):
        dist = LogNormalDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)

        assert np.array_equal(
            out,
            dists.lognorm.ppf(errors, loc=dot_prod(data, coefs['mu_']), s=dot_prod(data, coefs['sigma_']))
        )