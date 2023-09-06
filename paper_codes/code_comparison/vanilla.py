from numpy.random import choice, normal, uniform
from numpy import arctan, stack, dot
from scipy.special import expit
from scipy.stats import bernoulli
from pandas import DataFrame

# configs
N_SAMPLES = 100
DENSITY = 0.8

# create Y_1
mu, sigma = uniform(-2, 2), 1
y_1 = normal(mu, sigma, size=N_SAMPLES)

# create Y_2
alpha, beta = uniform(1, 5), uniform(-0.8, 0.8)

linear = (uniform(-2, 2) if choice([0, 1], p=[0.5, 0.5]) else uniform(5, 10))
y1_y2_input = arctan(alpha * (y_1 - beta))
p_bernoulli = expit(y1_y2_input * linear)
y_2 = bernoulli.ppf(uniform(0, 1, size=N_SAMPLES),
                    p_bernoulli)

# create Y_3
dist = choice(['normal', 'bernoulli'], p=[0.5, 0.5])
y3_inputs = stack([y_1, y_2]).transpose()
if dist == 'normal':
    mu_linear = (uniform(-3, 1, size=2) if choice([0, 1], p=[0.5, 0.5]) else uniform(1, 3, size=2))
    mu_bias = uniform(-2, 2, size=N_SAMPLES)
    sigma_linear, sigma_bias = uniform(-1, 0, size=2), 1
    y_3 = normal(mu_bias + dot(y3_inputs, mu_linear),
                 expit(dot(y3_inputs, sigma_linear) + sigma_bias))
else:
    p_linear = (uniform(-2, 2, size=2) if choice([0, 1], p=[0.5, 0.5]) else uniform(5, 10, size=2))
    p_bias = (uniform(-3, 1) if choice([0, 1], p=[0.5, 0.5]) else uniform(1, 3))
    y_3 = bernoulli.ppf(uniform(0, 1, size=N_SAMPLES),
                        expit(p_bias + dot(y3_inputs, p_linear)))

samples = DataFrame({'Y_1': y_1, 'Y_2': y_2, 'Y_3': y_3})
