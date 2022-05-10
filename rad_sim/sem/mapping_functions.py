import numpy as np
from scipy.stats import beta, gamma
from scipy.special import expit
from sklearn.preprocessing import PolynomialFeatures

# TODO: std_V and var_V should be checked, both in paper and code
# TODO: adding noise in the binary output in paper is wrong
# TODO: should gamma noise += (min + 1) to fix support to (0, inf) (in paper)
# TODO: no-parent nodes launched only by Gaussian normal distribution
# TODO: Catch Overflow warning for exponent


def get_function_params_list(function=None):
    lookup = {
        'edge': {
            'sigmoid': ('alpha', 'beta', 'gamma', 'tau', 'percentiles', 'rho'),
            'gaussian_rbf': ('alpha', 'beta', 'gamma', 'tau', 'percentiles', 'rho'),
            'beta_noise': ('rho', ),
            'binary_identity': ()
        },
        'state': {
        },
        'output': {
            'gaussian_noise': ('rho', ),
            'gamma_noise': ('rho', ),
            'bernoulli': ('mean_', 'gamma', 'percentiles', 'rho')
        }
    }
    return lookup[function]


def get_output_function_options(output_type=None):
    lookup = {
        'binary': ('bernoulli', ),
        'continuous': ('gaussian_noise', 'gamma_noise'),
        'categorical': ('multinomial', )
    }
    return lookup[output_type]


def edge_empty(**kwargs):
    assert kwargs
    print('edge function not implemented')
    raise ValueError


def edge_sigmoid(array=None,
                 alpha=1.0, beta=0.0, gamma=1, tau=1,
                 percentiles=10, rho=0.2):
    # 1,2. SCALE ARRAY
    V = scale_V(array=array, percentiles=percentiles)

    # 3. PERTURB V
    noise = np.random.normal(loc=0, scale=rho*np.std(V), size=len(V))
    V += noise

    # 4. CALCULATE A (negative for using expit instead of numpy exponential)
    expon = -((-1)**gamma) * 2 * alpha * ((V - beta)**tau)

    return expit(expon)


def edge_gaussian_rbf(array=None,
                      alpha=1, beta=0, gamma=0, tau=2,
                      percentiles=10, rho=0.2):
    # 1,2. SCALE ARRAY
    V = scale_V(array=array, percentiles=percentiles)

    # 3. PERTURB V
    noise = np.random.normal(loc=0, scale=rho*np.std(V), size=len(V))
    V += noise

    # 4. CALCULATE A
    expon = -alpha * ((V - beta)**tau)

    return gamma + ((-1)**gamma) * np.exp(expon)


def edge_binary_beta(array=None,
                     rho=None):
    # define errors
    e_0 = beta(1, 100-99*rho)
    e_1 = beta(100-99*rho, 1)

    # perturb the binary input
    return np.array([e_0.rvs(1)[0] if i == 0 else e_1.rvs(1)[0] for i in array])


def edge_binary_identity(array=None):
    return array


def state_linear(inputs=None, parents_order=None, coefs=None):
    # assert on the number of coefs
    assert len(parents_order) == len(coefs)
    data = inputs[parents_order]
    return np.dot(
        data.values,
        coefs
    )


def state_poly2(inputs=None, parents_order=None, coefs=None):
    poly2 = PolynomialFeatures(
        degree=2,
        interaction_only=False,
        include_bias=True
    )
    data = poly2.fit_transform(inputs[parents_order].values)

    assert data.shape[1] == len(coefs)
    return np.dot(
        data,
        coefs
    )


def state_poly1_interactions(inputs=None, parents_order=None, coefs=None):
    poly2 = PolynomialFeatures(
        degree=2,
        interaction_only=True,
        include_bias=True
    )
    data = poly2.fit_transform(inputs[parents_order].values)

    assert data.shape[1] == len(coefs)
    return np.dot(
        data,
        coefs
    )


def state_empty(**kwargs):
    assert kwargs
    print('state function not implemented')
    raise ValueError


def output_gaussian_noise(array=None,
                          rho=0.2):
    noise = np.random.normal(loc=0, scale=rho * np.std(array), size=len(array))
    return array + noise


def output_gamma_noise(array=None,
                       rho=0.2):
    std2 = np.var(array)
    array += abs(min(array)) + 0.01

    return np.array([
        gamma((i ** 2) / (rho * std2), scale=(rho * std2) / i).rvs(1)[0]
        for i in array
    ])


def output_bernoulli(array=None,
                     mean_=None,
                     gamma=None,
                     percentiles=10, rho=0.2):
    # scale
    V = edge_sigmoid(array=array, gamma=gamma, percentiles=percentiles, rho=rho)
    # FIX MEAN
    try:
        assert mean_ is not None
        # 1. current mean
        E_V = np.mean(V)

        # 2. Case:
        if mean_ < E_V:
            a = mean_ / E_V
            b = 0
        elif mean_ > E_V:
            a = (1-mean_) / (max(V) - E_V)
            b = 1 - a * max(V)
        else:
            a = 1
            b = 0
        # 3. apply
        V = a * V + b
    except AssertionError:
        pass

    return [np.random.choice([0, 1], p=[1-i, i]) for i in V]


def output_multinomial(array=None,
                       centers=None,
                       gamma=None,
                       percentiles=10,
                       rho=0.2):
    # scale
    V = edge_sigmoid(array=array, gamma=gamma, percentiles=percentiles, rho=rho)

    # FIX MEAN
    return [
        np.random.choice(
            [i for i in range(len(centers))],
            p=np.nan_to_num(1/np.abs(v-centers)/(1/np.abs(v-centers)).sum(), nan=1)
        ) for v in V
    ]


def output_empty(**kwargs):
    assert kwargs
    print('output function not implemented')
    raise ValueError


# === HELPERS ===
def scale_V(array=None, percentiles=None):
    # percentiles
    P_n = np.percentile(array, q=percentiles)
    P_100n = np.percentile(array, q=100-percentiles)
    # scale
    return -1 + 2 * (array - P_n) / (P_100n - P_n)


if __name__ == '__main__':
    arr = np.array(range(100))

    # v = edge_sigmoid_new(array=arr)
    v = edge_gaussian_rbf(array=arr)

    from matplotlib import pyplot as plt
    plt.scatter(arr, v)
    plt.show()
