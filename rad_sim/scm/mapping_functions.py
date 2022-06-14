import numpy as np
from scipy.special import expit


def edge_empty(**kwargs):
    assert kwargs
    print('edge function not implemented')
    raise ValueError


def edge_sigmoid(array=None,
                 alpha=1.0, beta=0.0, gamma=1, tau=1,
                 percentiles=10, rho=0.2):
    # 1,2. SCALE ARRAY
    V = scale_V(array=array)

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
    V = scale_V(array=array)

    # 3. PERTURB V
    noise = np.random.normal(loc=0, scale=rho*np.std(V), size=len(V))
    V += noise

    # 4. CALCULATE A
    expon = -alpha * ((V - beta)**tau)

    return gamma + ((-1)**gamma) * np.exp(expon)


def edge_binary_identity(array=None):
    return array


# === HELPERS ===
def scale_V(array=None):
    return (array - array.mean()) / array.std()


if __name__ == '__main__':
    arr = np.array(range(100))

    # v = edge_sigmoid_new(array=arr)
    v = edge_gaussian_rbf(array=arr)

    from matplotlib import pyplot as plt
    plt.scatter(arr, v)
    plt.show()
