import numpy as np
from scipy.special import expit


def brownian(sample_size=None, time_points=None, rho=1):
    # sample step values
    steps = np.random.normal(size=sample_size)
    # make X
    x = np.array([
        [np.random.normal(step, rho) for _ in range(time_points)]
        for step in steps
    ])
    x = x.cumsum(axis=1)

    # make Y = f (brownian motion mean)
    bernoulli_prob = expit(steps)
    y = np.array([
        np.random.choice([0, 1], p=[prob, 1-prob]) for prob in bernoulli_prob
    ])
    return x, y


def additive_model(sample_size=None, time_points=None):
    def trend(slope=None, intercept=None, t=None):
        return slope * t + intercept

    def seasonal(w=None, phi=None, magnitude=None, t=None):
        return magnitude * np.sin(w * t + phi)

    def noise(sigma=0.5):
        return np.random.normal(0, sigma)

    # sample params
    slope = np.random.normal(0, 0.2, size=sample_size)
    intercept = np.random.uniform(0, 1, size=sample_size)
    w = np.random.uniform(2*np.pi/10, 2*np.pi/5, size=sample_size)
    phi = np.random.normal(0, np.pi/2, size=sample_size)
    magnitude = np.random.uniform(1, 3, size=sample_size)

    # make X
    x = np.array([
        trend(t=t, slope=slope, intercept=intercept) +
        seasonal(t=t, w=w, phi=phi, magnitude=magnitude) +
        noise()
        for t in range(time_points)
    ])

    # make Y = f(slope, w)
    bias = np.ones(shape=(sample_size,))*3*np.pi/10
    latent = np.dstack([slope, w, bias])[0]
    weights = np.array([2, 1, -1])
    bernoulli_prob = expit(np.dot(latent, weights))
    y = np.array([
        np.random.choice([0, 1], p=[prob, 1 - prob]) for prob in bernoulli_prob
    ])

    return x.transpose(), y


def fourier_series(sample_size=None, time_points=None, num_sines=4):
    # sample frequencies
    freqs = np.random.uniform(2*np.pi/20, 2*np.pi/3, size=(sample_size, num_sines))
    # sample phi
    phi = np.random.normal(0, np.pi/3, size=(sample_size, num_sines))
    # sample magnitudes
    magnitudes = np.random.uniform(1, 5, size=(sample_size, num_sines))

    # make X
    x = np.array([
        magnitudes * np.sin(t*freqs+phi)
        for t in range(time_points)
    ]).sum(axis=2).transpose()

    # make Y = f(freqs)
    bernoulli_prob = expit(freqs.mean(axis=1)-np.pi/4)
    y = np.array([
        np.random.choice([0, 1], p=[prob, 1 - prob]) for prob in bernoulli_prob
    ])

    return x, y