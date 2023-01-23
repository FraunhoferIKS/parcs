import numpy as np
import pandas as pd

def real_power(base, power):
    return np.sign(base) * np.abs(base)**power

def ones_power(data):
    return pd.DataFrame(np.ones(len(data)) * data["dp"][1], columns = ["U"])

def ones_ones(data):
    return pd.DataFrame(np.ones(len(data)), columns = ["U"])

def X1(data):
    return np.exp(data['e1']) - np.exp(0.5)

def X2(data):
    return np.sign(data['sb'][0]) * data['b'][0] * real_power(data['X1'], data['P']) + np.exp(data['e2']) - np.exp(0.5)

def X3(data):
    return np.sign(data['sb'][1]) * data['b'][1] * real_power(data['X1'], data['P']) + np.sign(data['sb'][2]) * data['b'][2] * real_power(data['X2'], data['P']) + np.exp(data['e3']) - np.exp(0.5)

def X4(data):
    return np.sign(data['sb'][3]) * data['b'][3] * real_power(data['X1'], data['P']) + np.sign(data['sb'][4]) * data['b'][4] * real_power(data['X2'], data['P']) + np.sign(data['sb'][5]) * data['b'][5] * real_power(data['X3'], data['P']) + np.exp(data['e4']) - np.exp(0.5)

def X5(data):
    return np.sign(data['sb'][6]) * data['b'][6] * real_power(data['X1'], data['P']) + np.sign(data['sb'][7]) * data['b'][7] * real_power(data['X2'], data['P']) + np.sign(data['sb'][8]) * data['b'][8] * real_power(data['X3'], data['P']) + np.sign(data['sb'][9]) * data['b'][9] * real_power(data['X4'], data['P']) + np.exp(data['e5']) - np.exp(0.5)