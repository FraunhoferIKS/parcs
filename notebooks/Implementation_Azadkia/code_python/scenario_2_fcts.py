import numpy as np
import pandas as pd
from scipy.stats import norm

def gausss(data):
    return pd.DataFrame(np.ones(len(data)) * norm.cdf(data["e"][1]), columns = ["U"])
def arctand2(data):
    return np.arctan(data['d2'])
def arctand3(data):
    return np.arctan(data['d3'])
def arctand4(data):
    return np.arctan(data['d4'])
def arctand5(data):
    return np.arctan(data['d5'])
def arctand6(data):
    return np.arctan(data['d6'])
def arctand7(data):
    return np.arctan(data['d7'])