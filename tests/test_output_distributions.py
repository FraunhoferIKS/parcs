import pytest
from data.output_distributions_data import BaseClassTestData, DistributionsData
from pyparcs.api.output_distributions import *
from pyparcs.api.utils import *


class TestBaseClasses:
    @staticmethod
    @pytest.mark.parametrize('name,data,coef', BaseClassTestData.dist_param)
    def test_param_class(name, data, coef):
        """Tests DistParam functionality
        the dot_product function is already tested, so we rely on it in this unit test
        """
        param = DistParam(name=name, coef=coef, corrector=None)
        assert param.name == name
        assert np.array_equal(param.calculate(data), dot_prod(data, coef))

    @staticmethod
    @pytest.mark.parametrize('icdf,params,coefs,errors,data', BaseClassTestData.parcs_dist)
    def test_parcs_dist_class(icdf, params, coefs, errors, data):
        dist = PARCSDistribution(icdf=icdf, params=params, coefs=coefs, correctors={p: None for p in params})
        assert np.array_equal(
            dist.calculate(data, errors),
            icdf(errors, **{p: dot_prod(data, coefs[p]) for p in params})
        )


class TestDistributions:
    """Tests existing distributions, inheriting from PARCSDistribution"""
    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', DistributionsData.bernoulli_data)
    def test_bernoulli_distribution(coefs, errors, data):
        """Tests Bernoulli distribution"""
        dist = BernoulliDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        assert np.array_equal(
            out,
            dists.bernoulli.ppf(errors, p=dot_prod(data, coefs['p_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data,correction_config',
                             DistributionsData.bernoulli_correction_data)
    def test_bernoulli_correction(coefs, errors, data, correction_config):
        """Tests Bernoulli distribution with correction"""
        dist = BernoulliDistribution(coefs=coefs,
                                     do_correction=True,
                                     correction_config=correction_config)
        out = dist.calculate(data, errors)
        sigmoid_correction = SigmoidCorrection(**correction_config)
        assert np.array_equal(
            out,
            dists.bernoulli.ppf(errors,
                                p=sigmoid_correction.transform(dot_prod(data,
                                                                        coefs['p_'])))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', DistributionsData.uniform_data)
    def test_uniform_distribution(coefs, errors, data):
        """Tests uniform distribution"""
        dist = UniformDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        mu = dot_prod(data, coefs['mu_'])
        diff = dot_prod(data, coefs['diff_'])
        assert np.array_equal(out,
                              mu + (np.array(errors) - 0.5) * diff)

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', DistributionsData.normal_data)
    def test_normal_distribution(coefs, errors, data):
        """Tests Gaussian normal distribution"""
        dist = GaussianNormalDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)

        assert np.array_equal(
            out,
            dists.norm.ppf(errors, dot_prod(data, coefs['mu_']), dot_prod(data, coefs['sigma_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', DistributionsData.exponential_data)
    def test_exponential_distribution(coefs, errors, data):
        """Tests exponential distribution"""
        dist = ExponentialDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        assert np.array_equal(
            out,
            dists.expon.ppf(errors, dot_prod(data, coefs['lambda_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', DistributionsData.poisson_data)
    def test_poisson_distribution(coefs, errors, data):
        """Tests Poisson distribution"""
        dist = PoissonDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)
        assert np.array_equal(
            out,
            dists.poisson.ppf(errors, dot_prod(data, coefs['lambda_']))
        )

    @staticmethod
    @pytest.mark.parametrize('coefs,errors,data', DistributionsData.lognormal_data)
    def test_lognormal_distribution(coefs, errors, data):
        """Tests lognormal distribution"""
        dist = LogNormalDistribution(coefs=coefs, do_correction=False)
        out = dist.calculate(data, errors)

        assert np.array_equal(
            out,
            dists.lognorm.ppf(errors,
                              loc=dot_prod(data, coefs['mu_']),
                              s=dot_prod(data, coefs['sigma_']))
        )