import numpy as np


def log_sin(data):
    """some custom functions"""
    return 1 + 2 * np.log(data['A']+1) + 3 * np.sin(data['B']**2)
